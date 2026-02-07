import torch
import torch.nn as nn
from .base import BasePruner
from .engine.hessian_free import SNOWSEngine
import numpy as np
from .optimizer import lagrangian_optimization 

class PDTPruner(BasePruner):
    def __init__(self, model, config, args=None, topology_groups=None):
        super().__init__(model, config)
        
        strat_cfg = config.get('strategy', {})
        
        # --- 인자 설정 우선순위 로직 ---
        self.group_selection_ratio = strat_cfg.get('group_selection_ratio', 1.0)
        self.final_keep_ratio = strat_cfg.get('channel_keep_ratio', 0.2)
        self.min_survival_ratio = strat_cfg.get('min_survival_ratio', 0.1)

        if args:
            if hasattr(args, 'group_selection_ratio') and args.group_selection_ratio is not None:
                self.group_selection_ratio = args.group_selection_ratio
            if hasattr(args, 'channel_keep_ratio') and args.channel_keep_ratio is not None:
                self.final_keep_ratio = args.channel_keep_ratio
            if hasattr(args, 'min_survival_ratio') and args.min_survival_ratio is not None:
                self.min_survival_ratio = args.min_survival_ratio

        self.ema_decay = strat_cfg.get('ema_decay', 0.95)
        self.lambda_h = strat_cfg.get('lambda_h', 0.005)
        self.k_horizon = strat_cfg.get('k_horizon', 25)
        self.engine = SNOWSEngine(n_iter=strat_cfg.get('hessian_iter', 10))

        self.topology_groups = topology_groups
        self.layers_dict = nn.ModuleDict()
        
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                self.layers_dict[name.replace('.', '_')] = m
        
        self.layers = list(self.layers_dict.values())
        self._check_buffers()
        
        print(f"\n[Pruner Init] Strict Conv-Topology Mode Activated")
        print(f"[*] Applied Min Survival Guarantee: {self.min_survival_ratio*100:.1f}%")
        print(f"[*] Only defined groups (1-{len(self.topology_groups) if self.topology_groups else 0}) will be pruned.")

    def _check_buffers(self):
        for m in self.layers:
            n_f = m.weight.shape[0]
            if not hasattr(m, 'mask'):
                m.register_buffer("mask", torch.ones(n_f, device=self.device))
            if not hasattr(m, 'grad_ema'):
                m.register_buffer("grad_ema", torch.zeros(n_f, device=self.device))
            if not hasattr(m, 'hessian_score'):
                m.register_buffer("hessian_score", torch.zeros(n_f, device=self.device))

    def apply_mask_to_weights(self, optimizer=None):
        with torch.no_grad():
            for m in self.layers:
                mask = m.mask
                m_view = mask.view(-1, 1, 1, 1) if m.weight.dim() == 4 else mask.view(-1, 1)
                m.weight.data.mul_(m_view)
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.mul_(mask)

    def update_ema_and_mask_grad(self):
        with torch.no_grad():
            for m in self.layers:
                if hasattr(m, 'weight') and m.weight.grad is not None:
                    m_view = m.mask.view(-1, 1, 1, 1) if m.weight.dim()==4 else m.mask.view(-1, 1)
                    m.weight.grad.mul_(m_view)
                    g = m.weight.grad.pow(2).reshape(m.weight.shape[0], -1).mean(1)
                    m.grad_ema.mul_(self.ema_decay).add_(g, alpha=1 - self.ema_decay)

    def step_pruning(self, loss, current_epoch, total_epochs):
        all_modules = dict(self.model.named_modules())
        def find_layer(name):
            if name in all_modules: return all_modules[name]
            return all_modules.get(name.replace('_', '.'))

        # 1. 타겟 생존율 계산
        progress = current_epoch / total_epochs
        total_target_keep_ratio = 1.0 - (progress * (1.0 - self.final_keep_ratio))
        
        # 2. 토폴로지 그룹 수집
        group_info_list = []
        if self.topology_groups:
            for idx, group in enumerate(self.topology_groups):
                score_layers = [find_layer(ln) for ln in group 
                                if isinstance(find_layer(ln), (nn.Conv2d, nn.Linear))]
                if not score_layers: continue
                w_g = torch.mean(torch.stack([m.grad_ema.mean() for m in score_layers])).item()
                group_info_list.append({
                    'id': idx+1, 
                    'layers': score_layers, 
                    'w_g': w_g, 
                    'names': group
                })

        if not group_info_list: return

        # 3. 중요도 정렬 및 타겟 선정
        sorted_groups = sorted(group_info_list, key=lambda x: x['w_g'])
        num_targets = int(len(sorted_groups) * self.group_selection_ratio)
        target_group_ids = [g['id'] for g in sorted_groups[:num_targets]]

        # --- [4] Hessian 계산 및 2단계 정규화 ---
        target_params = [m.weight for g in sorted_groups[:num_targets] for m in g['layers']]
        hv_list = self.engine.get_k_step_hessian_selective(loss, target_params, self.k_horizon)

        target_unit_scores = []
        target_unit_costs = []
        target_unit_metadata = []

        hv_idx = 0
        for g in sorted_groups[:num_targets]:
            mask = g['layers'][0].mask
            alive_indices = torch.where(mask > 0.5)[0].cpu().numpy()
            total_n = mask.numel()

            for m in g['layers']:
                hv = hv_list[hv_idx]
                h_energy = hv.pow(2).reshape(hv.shape[0], -1).mean(1)
                if h_energy.max() > h_energy.min():
                    h_energy = (h_energy - h_energy.min()) / (h_energy.max() - h_energy.min() + 1e-8)
                m.hessian_score.copy_(h_energy)
                hv_idx += 1

            if (len(alive_indices) / total_n) <= self.min_survival_ratio:
                if g['id'] in target_group_ids: target_group_ids.remove(g['id'])
                continue

            if len(alive_indices) > 0:
                for i in alive_indices:
                    s_gc_list = [m.hessian_score[i].item() for m in g['layers'] if i < m.hessian_score.size(0)]
                    s_gc = sum(s_gc_list)/len(s_gc_list) if s_gc_list else 0.0
                    
                    # [비교 포인트] PDT 수식: Grad EMA * (Hessian Score * Lambda)
                    raw_score = g['w_g'] * (s_gc * self.lambda_h)
                    
                    target_unit_scores.append(raw_score) 
                    target_unit_costs.append(sum(m.weight.nelement()/m.weight.shape[0] for m in g['layers']))
                    target_unit_metadata.append((g, i))

        # --- [5] 프루닝 최적화 및 마스크 적용 ---
        current_sparsity = self.get_current_sparsity() / 100.0
        current_alive_ratio = 1.0 - current_sparsity
        pruned_count = 0
        cutoff_threshold = 0.0

        if current_alive_ratio > total_target_keep_ratio and target_unit_scores:
            incremental_keep_ratio = total_target_keep_ratio / current_alive_ratio
            total_budget = np.sum(target_unit_costs) * incremental_keep_ratio
            optimal_mask_flags = lagrangian_optimization(np.array(target_unit_scores), np.array(target_unit_costs), total_budget)
            dead_scores = np.array(target_unit_scores)[optimal_mask_flags == 0]
            if len(dead_scores) > 0: cutoff_threshold = np.max(dead_scores)

            with torch.no_grad():
                for idx, is_alive in enumerate(optimal_mask_flags):
                    if not is_alive:
                        group_obj, channel_idx = target_unit_metadata[idx]
                        for ln in group_obj['names']:
                            layer_obj = find_layer(ln)
                            if layer_obj is not None and hasattr(layer_obj, 'mask'):
                                if channel_idx < layer_obj.mask.size(0):
                                    layer_obj.mask[channel_idx] = 0.0
                        pruned_count += 1

        # --- [6] 결과 출력 ---
        print(f"\n{'='*30} PDT Pruning: Epoch {current_epoch} {'='*30}")
        print(f" [*] Cut-off Threshold: {cutoff_threshold:.6f} | Pruned: {pruned_count}")
        print(f" {'Group ID':<10} | {'Alive/Total':>12} | {'Sparsity':>8} | {'Hessian(avg)':>12} | {'Status'}")
        print(f" {'-'*87}")
        
        group_info_list.sort(key=lambda x: x['id'])
        for g in group_info_list:
            m = g['layers'][0]
            total, alive = m.mask.numel(), int(m.mask.sum().item())
            sparsity = (1 - alive/total) * 100
            h_avg = g['layers'][0].hessian_score.mean().item() if g['id'] in target_group_ids else 0.0
            
            status = "TARGET" if g['id'] in target_group_ids else "FIXED"
            if (alive/total) <= self.min_survival_ratio: status = "MIN-SURV"
            print(f" Group {g['id']:2d}      | {alive:4d}/{total:4d}      | {sparsity:>7.1f}% | {h_avg:>12.6f} | [{status}]")
        print(f"{'='*89}\n")
        # --- 학회용 리소스 분석 출력 ---
        eff = self.get_model_efficiency()
        print(f"\n[Scientific Metrics - Epoch {current_epoch}]")
        print(f" 🟢 Model Size: {eff['orig_mb']:.2f} MB -> {eff['curr_mb']:.2f} MB (Reduction: {eff['orig_mb'] - eff['curr_mb']:.2f} MB)")
        print(f" 🔵 Sparsity (Params/FLOPs): {eff['sparsity']:.2f} %")
        print(f" 🟡 Theoretical Speedup: {eff['speedup']:.2f}x")
        # 실제 측정치 (추론 모드 기준 아님, 현재 학습 세션 기준)
        print(f" 🟠 Current GPU Mem (Allocated): {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"{'='*89}\n")
        torch.cuda.empty_cache()

    def get_current_sparsity(self):
        total_p = sum(m.mask.numel() for m in self.layers)
        active_p = sum(m.mask.sum().item() for m in self.layers)
        return (1.0 - (active_p / total_p)) * 100.0 if total_p > 0 else 0.0

    def get_model_efficiency(self, example_inputs=None):
        """해외 학회용: FLOPs, Latency(Proxy), Memory, Size를 이론적으로 계산"""
        total_params = 0
        remaining_params = 0
        total_flops = 0
        remaining_flops = 0
        
        # 1. Parameter & FLOPs 계산 (VGG 기준)
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # 원본 수치
                n_p = m.weight.numel()
                total_params += n_p
                
                # FLOPs 근사 (H * W * C_in * C_out * K * K)
                # 여기서는 파라미터 감소 비율을 FLOPs 감소 비율의 프록시로 사용
                if hasattr(m, 'mask'):
                    keep_ratio = m.mask.sum().item() / m.mask.numel()
                    remaining_params += n_p * keep_ratio
                    # Conv의 경우 입력/출력이 같이 줄어들면 제곱으로 줄어들지만, 
                    # 마스킹 단계에서는 출력 채널 감소 비율을 기준으로 선형 근사하여 보수적 측정
                    remaining_flops += n_p * keep_ratio 
                else:
                    remaining_params += n_p
                    remaining_flops += n_p

        # 2. 지표 산출
        orig_size_mb = (total_params * 4) / (1024**2)
        curr_size_mb = (remaining_params * 4) / (1024**2)
        sparsity = (1 - remaining_params/total_params) * 100
        
        # 3. Latency & Energy (학술적 프록시 모델링)
        # 실제 Latency는 하드웨어 종속적이므로, FLOPs 감소량에 기반한 이론적 가속도를 출력
        theoretical_speedup = total_params / (remaining_params + 1e-8)
        
        return {
            'orig_mb': orig_size_mb,
            'curr_mb': curr_size_mb,
            'sparsity': sparsity,
            'speedup': theoretical_speedup
        }


# ==============================================================================
# HAP (Hessian-Aware Pruning) 비교 실험용 Pruner
# ==============================================================================
class HAPPruner(PDTPruner):
    def step_pruning(self, loss, current_epoch, total_epochs):
        all_modules = dict(self.model.named_modules())
        def find_layer(name):
            if name in all_modules: return all_modules[name]
            return all_modules.get(name.replace('_', '.'))

        # 1. 현재 전체 목표 Sparsity 계산
        progress = current_epoch / total_epochs
        total_target_sparsity = progress * (1.0 - self.final_keep_ratio)
        
        group_info_list = []
        for idx, group in enumerate(self.topology_groups):
            score_layers = [find_layer(ln) for ln in group if isinstance(find_layer(ln), (nn.Conv2d, nn.Linear))]
            if not score_layers: continue
            group_info_list.append({'id': idx+1, 'layers': score_layers, 'names': group})

        # 2. Hessian 계산
        target_params = [m.weight for g in group_info_list for m in g['layers']]
        hv_list = self.engine.get_k_step_hessian_selective(loss, target_params, self.k_horizon)

        # 3. 레이어별(그룹별) Hessian Trace 평균 계산
        group_traces = []
        hv_idx = 0
        for g in group_info_list:
            traces = []
            for m in g['layers']:
                hv = hv_list[hv_idx]
                trace = hv.pow(2).mean().item() # Trace 근사
                traces.append(trace)
                hv_idx += 1
            group_traces.append(np.mean(traces))

        # 4. [HAP 정석 로직] Hessian에 반비례하게 Sparsity 할당 (총합 유지)
        # Sensitivity S_i = 1 / Trace_i
        sensitivities = [1.0 / (t + 1e-8) for t in group_traces]
        total_sens = sum(sensitivities)
        
        # 각 그룹이 가져갈 Sparsity 가중치 계산
        # 전체 목표 Sparsity가 0.3이라면, 민감도 비율에 따라 배분
        pruned_count_total = 0
        
        print(f"\n{'='*30} HAP Corrected Pruning: Epoch {current_epoch} {'='*30}")
        
        for i, g in enumerate(group_info_list):
            # 그룹별 할당 Sparsity = (전체 목표) * (내 민감도 / 평균 민감도)
            # 단, 너무 몰살되지 않게 하되 HAP 논리 유지
            group_sparsity = total_target_sparsity * (sensitivities[i] / (total_sens / len(group_info_list)))
            group_sparsity = min(group_sparsity, 1.0 - self.min_survival_ratio) # 최소 생존 보장
            
            # 마스크 적용
            mask = g['layers'][0].mask
            num_channels = mask.numel()
            num_prune = int(num_channels * group_sparsity)
            
            # 해당 그룹 내 모든 레이어 가중치의 L1-norm 기준으로 num_prune만큼 자름 (HAP 표준)
            # (Hessian으로 Sparsity '양'을 정하고, L1으로 '대상'을 정하는 것이 HAP의 일반적 방식)
            weight_mags = torch.mean(torch.stack([m.weight.data.abs().reshape(m.weight.shape[0], -1).mean(1) for m in g['layers']]), dim=0)
            _, prune_indices = torch.topk(weight_mags, k=num_prune, largest=False)
            
            with torch.no_grad():
                for ln in g['names']:
                    layer_obj = find_layer(ln)
                    if layer_obj is not None and hasattr(layer_obj, 'mask'):
                        layer_obj.mask[prune_indices] = 0.0
            
            pruned_count_total += num_prune
            print(f" Group {g['id']:2d} | Assigned Sparsity: {group_sparsity*100:4.1f}% | Alive: {num_channels-num_prune:4d}/{num_channels:4d}")

        print(f" [*] Total Pruned in this step: {pruned_count_total}")
        print(f"{'='*89}\n")
        # --- 학회용 리소스 분석 출력 ---
        eff = self.get_model_efficiency()
        print(f"\n[Scientific Metrics - Epoch {current_epoch}]")
        print(f" 🟢 Model Size: {eff['orig_mb']:.2f} MB -> {eff['curr_mb']:.2f} MB (Reduction: {eff['orig_mb'] - eff['curr_mb']:.2f} MB)")
        print(f" 🔵 Sparsity (Params/FLOPs): {eff['sparsity']:.2f} %")
        print(f" 🟡 Theoretical Speedup: {eff['speedup']:.2f}x")
        # 실제 측정치 (추론 모드 기준 아님, 현재 학습 세션 기준)
        print(f" 🟠 Current GPU Mem (Allocated): {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"{'='*89}\n")
        torch.cuda.empty_cache()
# ==============================================================================
# SNOWS (Hessian Trace) 비교 실험용 Pruner
# ==============================================================================
class SNOWSPruner(PDTPruner):
    """
    SNOWS 논문의 정석 로직: 중요도 = 순수 Hessian_Trace
    (Grad EMA 보정 없이 순수한 현재 배치의 Hessian 에너지만 사용)
    """
    def step_pruning(self, loss, current_epoch, total_epochs):
        all_modules = dict(self.model.named_modules())
        def find_layer(name):
            if name in all_modules: return all_modules[name]
            return all_modules.get(name.replace('_', '.'))

        progress = current_epoch / total_epochs
        total_target_keep_ratio = 1.0 - (progress * (1.0 - self.final_keep_ratio))
        
        group_info_list = []
        if self.topology_groups:
            for idx, group in enumerate(self.topology_groups):
                score_layers = [find_layer(ln) for ln in group if isinstance(find_layer(ln), (nn.Conv2d, nn.Linear))]
                if not score_layers: continue
                group_info_list.append({'id': idx+1, 'layers': score_layers, 'names': group})

        if not group_info_list: return

        # [SNOWS 핵심] 모든 그룹을 대상으로 Hessian Trace 계산
        target_params = [m.weight for g in group_info_list for m in g['layers']]
        hv_list = self.engine.get_k_step_hessian_selective(loss, target_params, self.k_horizon)

        target_unit_scores = []
        target_unit_costs = []
        target_unit_metadata = []

        hv_idx = 0
        for g in group_info_list:
            mask = g['layers'][0].mask
            alive_indices = torch.where(mask > 0.5)[0].cpu().numpy()
            
            for m in g['layers']:
                hv = hv_list[hv_idx]
                # SNOWS의 핵심 지표: Hessian Trace 근사치 (H-Vector Product의 2-norm)
                h_energy = hv.pow(2).reshape(hv.shape[0], -1).mean(1)
                m.hessian_score.copy_(h_energy)
                hv_idx += 1

            if len(alive_indices) > 0:
                for i in alive_indices:
                    # SNOWS 방식 레이어별 스코어 (Grad EMA 곱하지 않고 순수 Hessian만 합산)
                    s_gc_list = [m.hessian_score[i].item() for m in g['layers'] if i < m.hessian_score.size(0)]
                    s_gc = sum(s_gc_list)/len(s_gc_list) if s_gc_list else 0.0
                    
                    target_unit_scores.append(s_gc) 
                    target_unit_costs.append(sum(m.weight.nelement()/m.weight.shape[0] for m in g['layers']))
                    target_unit_metadata.append((g, i))

        current_sparsity = self.get_current_sparsity() / 100.0
        current_alive_ratio = 1.0 - current_sparsity
        pruned_count = 0

        if current_alive_ratio > total_target_keep_ratio and target_unit_scores:
            incremental_keep_ratio = total_target_keep_ratio / current_alive_ratio
            total_budget = np.sum(target_unit_costs) * incremental_keep_ratio
            # 공정한 비교를 위해 동일한 최적화 엔진(Lagrangian) 사용
            optimal_mask_flags = lagrangian_optimization(np.array(target_unit_scores), np.array(target_unit_costs), total_budget)

            with torch.no_grad():
                for idx, is_alive in enumerate(optimal_mask_flags):
                    if not is_alive:
                        group_obj, channel_idx = target_unit_metadata[idx]
                        for ln in group_obj['names']:
                            layer_obj = find_layer(ln)
                            if layer_obj is not None and hasattr(layer_obj, 'mask'):
                                if channel_idx < layer_obj.mask.size(0):
                                    layer_obj.mask[channel_idx] = 0.0
                        pruned_count += 1

        print(f"\n{'='*30} SNOWS Comparison Pruning: Epoch {current_epoch} {'='*30}")
        print(f" [*] Method: SNOWS (Pure Hessian Trace) | Pruned: {pruned_count}")
        
        group_info_list.sort(key=lambda x: x['id'])
        for g in group_info_list:
            m = g['layers'][0]
            total, alive = m.mask.numel(), int(m.mask.sum().item())
            sparsity = (1 - alive/total) * 100
            print(f" Group {g['id']:2d}      | {alive:4d}/{total:4d}      | {sparsity:>7.1f}%")
        print(f"{'='*89}\n")
        # --- 학회용 리소스 분석 출력 ---
        eff = self.get_model_efficiency()
        print(f"\n[Scientific Metrics - Epoch {current_epoch}]")
        print(f" 🟢 Model Size: {eff['orig_mb']:.2f} MB -> {eff['curr_mb']:.2f} MB (Reduction: {eff['orig_mb'] - eff['curr_mb']:.2f} MB)")
        print(f" 🔵 Sparsity (Params/FLOPs): {eff['sparsity']:.2f} %")
        print(f" 🟡 Theoretical Speedup: {eff['speedup']:.2f}x")
        # 실제 측정치 (추론 모드 기준 아님, 현재 학습 세션 기준)
        print(f" 🟠 Current GPU Mem (Allocated): {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"{'='*89}\n")
        torch.cuda.empty_cache()




    # ==============================================================================
# ATO (AutoTrainOnce - Magnitude based) 비교 실험용 Pruner
# ==============================================================================
class ATOPruner(PDTPruner):
    """
    ATO 논문의 핵심 개념: Magnitude(L1-norm) 기반 점진적 프루닝
    Hessian을 계산하지 않고, 가중치의 크기(L1-norm)를 중요도로 사용합니다.
    """
    def step_pruning(self, loss, current_epoch, total_epochs):
        all_modules = dict(self.model.named_modules())
        def find_layer(name):
            if name in all_modules: return all_modules[name]
            return all_modules.get(name.replace('_', '.'))

        progress = current_epoch / total_epochs
        total_target_keep_ratio = 1.0 - (progress * (1.0 - self.final_keep_ratio))
        
        group_info_list = []
        if self.topology_groups:
            for idx, group in enumerate(self.topology_groups):
                score_layers = [find_layer(ln) for ln in group if isinstance(find_layer(ln), (nn.Conv2d, nn.Linear))]
                if not score_layers: continue
                group_info_list.append({'id': idx+1, 'layers': score_layers, 'names': group})

        if not group_info_list: return

        target_unit_scores = []
        target_unit_costs = []
        target_unit_metadata = []

        # [ATO 핵심] Hessian 대신 L1-norm(Magnitude) 계산
        for g in group_info_list:
            mask = g['layers'][0].mask
            alive_indices = torch.where(mask > 0.5)[0].cpu().numpy()
            
            # 그룹 내 모든 레이어의 Magnitude를 합산하여 중요도 판단
            # 가중치의 절대값 평균이 낮을수록 덜 중요하다는 ATO의 논리 반영
            magnitude_scores = []
            for m in g['layers']:
                # (out_channels, in_channels, k, k) -> (out_channels,)
                m_score = m.weight.data.abs().reshape(m.weight.shape[0], -1).mean(1)
                magnitude_scores.append(m_score)
            
            group_magnitude = torch.mean(torch.stack(magnitude_scores), dim=0)

            if len(alive_indices) > 0:
                for i in alive_indices:
                    # 해당 채널의 Magnitude 점수
                    s_gc = group_magnitude[i].item()
                    
                    target_unit_scores.append(s_gc) 
                    target_unit_costs.append(sum(m.weight.nelement()/m.weight.shape[0] for m in g['layers']))
                    target_unit_metadata.append((g, i))

        # 라그랑주 최적화는 공정성을 위해 동일하게 적용
        current_sparsity = self.get_current_sparsity() / 100.0
        current_alive_ratio = 1.0 - current_sparsity
        pruned_count = 0

        if current_alive_ratio > total_target_keep_ratio and target_unit_scores:
            incremental_keep_ratio = total_target_keep_ratio / current_alive_ratio
            total_budget = np.sum(target_unit_costs) * incremental_keep_ratio
            optimal_mask_flags = lagrangian_optimization(np.array(target_unit_scores), np.array(target_unit_costs), total_budget)

            with torch.no_grad():
                for idx, is_alive in enumerate(optimal_mask_flags):
                    if not is_alive:
                        group_obj, channel_idx = target_unit_metadata[idx]
                        for ln in group_obj['names']:
                            layer_obj = find_layer(ln)
                            if layer_obj is not None and hasattr(layer_obj, 'mask'):
                                if channel_idx < layer_obj.mask.size(0):
                                    layer_obj.mask[channel_idx] = 0.0
                        pruned_count += 1

        print(f"\n{'='*30} ATO (Magnitude) Comparison: Epoch {current_epoch} {'='*30}")
        print(f" [*] Method: ATO (L1-norm) | Pruned: {pruned_count}")
        # --- 학회용 리소스 분석 출력 ---
        eff = self.get_model_efficiency()
        print(f"\n[Scientific Metrics - Epoch {current_epoch}]")
        print(f" 🟢 Model Size: {eff['orig_mb']:.2f} MB -> {eff['curr_mb']:.2f} MB (Reduction: {eff['orig_mb'] - eff['curr_mb']:.2f} MB)")
        print(f" 🔵 Sparsity (Params/FLOPs): {eff['sparsity']:.2f} %")
        print(f" 🟡 Theoretical Speedup: {eff['speedup']:.2f}x")
        # 실제 측정치 (추론 모드 기준 아님, 현재 학습 세션 기준)
        print(f" 🟠 Current GPU Mem (Allocated): {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"{'='*89}\n")
        # ... (결과 출력 생략 - PDT와 동일)

    
# ==============================================================================
# SuperTickets (ST - Gradient-Weight Product based) 비교 실험용 Pruner
# ==============================================================================
class STPruner(PDTPruner):
    """
    SuperTickets 개념: Gradient와 Weight의 곱을 중요도로 사용
    단순 Magnitude보다 학습의 기여도를 더 정확히 포착한다고 가정합니다.
    """
    def step_pruning(self, loss, current_epoch, total_epochs):
        all_modules = dict(self.model.named_modules())
        def find_layer(name):
            if name in all_modules: return all_modules[name]
            return all_modules.get(name.replace('_', '.'))

        progress = current_epoch / total_epochs
        total_target_keep_ratio = 1.0 - (progress * (1.0 - self.final_keep_ratio))
        
        group_info_list = []
        if self.topology_groups:
            for idx, group in enumerate(self.topology_groups):
                score_layers = [find_layer(ln) for ln in group if isinstance(find_layer(ln), (nn.Conv2d, nn.Linear))]
                if not score_layers: continue
                group_info_list.append({'id': idx+1, 'layers': score_layers, 'names': group})

        if not group_info_list: return

        target_unit_scores = []
        target_unit_costs = []
        target_unit_metadata = []

        # [ST 핵심] Weight * Gradient_EMA 를 중요도로 사용
        for g in group_info_list:
            mask = g['layers'][0].mask
            alive_indices = torch.where(mask > 0.5)[0].cpu().numpy()
            
            st_scores = []
            for m in g['layers']:
                # SuperTickets의 변형 로직: |W * dL/dW|
                # 이미 m.grad_ema에 기울기의 제곱 평균 등이 담겨있으므로 이를 활용
                score = (m.weight.data.abs() * m.grad_ema.reshape(m.weight.shape[0], 1, 1, 1).sqrt()).reshape(m.weight.shape[0], -1).mean(1)
                st_scores.append(score)
            
            group_st_score = torch.mean(torch.stack(st_scores), dim=0)

            if len(alive_indices) > 0:
                for i in alive_indices:
                    s_gc = group_st_score[i].item()
                    target_unit_scores.append(s_gc) 
                    target_unit_costs.append(sum(m.weight.nelement()/m.weight.shape[0] for m in g['layers']))
                    target_unit_metadata.append((g, i))

        # 라그랑주 최적화 적용
        current_sparsity = self.get_current_sparsity() / 100.0
        current_alive_ratio = 1.0 - current_sparsity
        pruned_count = 0

        if current_alive_ratio > total_target_keep_ratio and target_unit_scores:
            incremental_keep_ratio = total_target_keep_ratio / current_alive_ratio
            total_budget = np.sum(target_unit_costs) * incremental_keep_ratio
            optimal_mask_flags = lagrangian_optimization(np.array(target_unit_scores), np.array(target_unit_costs), total_budget)

            with torch.no_grad():
                for idx, is_alive in enumerate(optimal_mask_flags):
                    if not is_alive:
                        group_obj, channel_idx = target_unit_metadata[idx]
                        for ln in group_obj['names']:
                            layer_obj = find_layer(ln)
                            if layer_obj is not None and hasattr(layer_obj, 'mask'):
                                if channel_idx < layer_obj.mask.size(0):
                                    layer_obj.mask[channel_idx] = 0.0
                        pruned_count += 1

        print(f"\n{'='*30} SuperTickets (ST) Comparison: Epoch {current_epoch} {'='*30}")
        print(f" [*] Method: ST (W * Grad) | Pruned: {pruned_count}")
        # --- 학회용 리소스 분석 출력 ---
        eff = self.get_model_efficiency()
        print(f"\n[Scientific Metrics - Epoch {current_epoch}]")
        print(f" 🟢 Model Size: {eff['orig_mb']:.2f} MB -> {eff['curr_mb']:.2f} MB (Reduction: {eff['orig_mb'] - eff['curr_mb']:.2f} MB)")
        print(f" 🔵 Sparsity (Params/FLOPs): {eff['sparsity']:.2f} %")
        print(f" 🟡 Theoretical Speedup: {eff['speedup']:.2f}x")
        # 실제 측정치 (추론 모드 기준 아님, 현재 학습 세션 기준)
        print(f" 🟠 Current GPU Mem (Allocated): {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"{'='*89}\n")
        torch.cuda.empty_cache()

# ==============================================================================
# DFPC (Data-Free Parameter Compensation - Similarity based) 비교 실험용 Pruner
# ==============================================================================
class DFPCPruner(PDTPruner):
    """
    DFPC 개념: 데이터 없이 필터 자체의 기하학적 분포를 분석
    필터 간의 거리가 멀수록(고유할수록) 중요하다고 판단합니다.
    """
    def step_pruning(self, loss, current_epoch, total_epochs):
        all_modules = dict(self.model.named_modules())
        def find_layer(name):
            if name in all_modules: return all_modules[name]
            return all_modules.get(name.replace('_', '.'))

        progress = current_epoch / total_epochs
        total_target_keep_ratio = 1.0 - (progress * (1.0 - self.final_keep_ratio))
        
        group_info_list = []
        if self.topology_groups:
            for idx, group in enumerate(self.topology_groups):
                score_layers = [find_layer(ln) for ln in group if isinstance(find_layer(ln), (nn.Conv2d, nn.Linear))]
                if not score_layers: continue
                group_info_list.append({'id': idx+1, 'layers': score_layers, 'names': group})

        if not group_info_list: return

        target_unit_scores = []
        target_unit_costs = []
        target_unit_metadata = []

        # [DFPC 핵심] 필터 간의 L2 Distance (Uniqueness)를 중요도로 사용
        for g in group_info_list:
            mask = g['layers'][0].mask
            alive_indices = torch.where(mask > 0.5)[0].cpu().numpy()
            
            group_dfpc_scores = []
            for m in g['layers']:
                # weight shape: [out_channels, in_channels, k, k]
                w = m.weight.data.reshape(m.weight.shape[0], -1)
                
                # 각 필터가 다른 필터들과 얼마나 다른지(L2 distance의 합) 계산
                # 다른 필터들과 거리가 멀수록 고유한 정보를 가졌다고 판단
                dist_matrix = torch.cdist(w, w, p=2)
                importance = dist_matrix.sum(dim=1) 
                group_dfpc_scores.append(importance)
            
            group_score = torch.mean(torch.stack(group_dfpc_scores), dim=0)

            if len(alive_indices) > 0:
                for i in alive_indices:
                    s_gc = group_score[i].item()
                    target_unit_scores.append(s_gc) 
                    target_unit_costs.append(sum(m.weight.nelement()/m.weight.shape[0] for m in g['layers']))
                    target_unit_metadata.append((g, i))

        # 라그랑주 최적화 적용
        current_sparsity = self.get_current_sparsity() / 100.0
        current_alive_ratio = 1.0 - current_sparsity
        pruned_count = 0

        if current_alive_ratio > total_target_keep_ratio and target_unit_scores:
            incremental_keep_ratio = total_target_keep_ratio / current_alive_ratio
            total_budget = np.sum(target_unit_costs) * incremental_keep_ratio
            optimal_mask_flags = lagrangian_optimization(np.array(target_unit_scores), np.array(target_unit_costs), total_budget)

            with torch.no_grad():
                for idx, is_alive in enumerate(optimal_mask_flags):
                    if not is_alive:
                        group_obj, channel_idx = target_unit_metadata[idx]
                        for ln in group_obj['names']:
                            layer_obj = find_layer(ln)
                            if layer_obj is not None and hasattr(layer_obj, 'mask'):
                                if channel_idx < layer_obj.mask.size(0):
                                    layer_obj.mask[channel_idx] = 0.0
                        pruned_count += 1

        print(f"\n{'='*30} DFPC (Similarity) Comparison: Epoch {current_epoch} {'='*30}")
        print(f" [*] Method: DFPC (L2-Distance) | Pruned: {pruned_count}")
        # --- 학회용 리소스 분석 출력 ---
        eff = self.get_model_efficiency()
        print(f"\n[Scientific Metrics - Epoch {current_epoch}]")
        print(f" 🟢 Model Size: {eff['orig_mb']:.2f} MB -> {eff['curr_mb']:.2f} MB (Reduction: {eff['orig_mb'] - eff['curr_mb']:.2f} MB)")
        print(f" 🔵 Sparsity (Params/FLOPs): {eff['sparsity']:.2f} %")
        print(f" 🟡 Theoretical Speedup: {eff['speedup']:.2f}x")
        # 실제 측정치 (추론 모드 기준 아님, 현재 학습 세션 기준)
        print(f" 🟠 Current GPU Mem (Allocated): {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"{'='*89}\n")
        torch.cuda.empty_cache()

# ==============================================================================
# TPP (Towards Personalized Pruning - Weight-Activation Interaction) 비교 실험용 Pruner
# ==============================================================================
class TPPPruner(PDTPruner):
    """
    TPP 개념: 가중치의 절대값과 활성화 기여도(Gradient 활용)의 곱을 통해
    채널별 '개인화된' 중요도를 산출합니다.
    """
    def step_pruning(self, loss, current_epoch, total_epochs):
        all_modules = dict(self.model.named_modules())
        def find_layer(name):
            if name in all_modules: return all_modules[name]
            return all_modules.get(name.replace('_', '.'))

        progress = current_epoch / total_epochs
        total_target_keep_ratio = 1.0 - (progress * (1.0 - self.final_keep_ratio))
        
        group_info_list = []
        if self.topology_groups:
            for idx, group in enumerate(self.topology_groups):
                score_layers = [find_layer(ln) for ln in group if isinstance(find_layer(ln), (nn.Conv2d, nn.Linear))]
                if not score_layers: continue
                group_info_list.append({'id': idx+1, 'layers': score_layers, 'names': group})

        if not group_info_list: return

        target_unit_scores = []
        target_unit_costs = []
        target_unit_metadata = []

        # [TPP 핵심] Weight Magnitude * Gradient Persistence (Grad-EMA)
        for g in group_info_list:
            mask = g['layers'][0].mask
            alive_indices = torch.where(mask > 0.5)[0].cpu().numpy()
            
            tpp_scores = []
            for m in g['layers']:
                # 가중치 크기(W)와 학습 지속성(Grad-EMA)의 기하평균적 결합
                # TPP 논문의 핵심인 'Weight-Activation Interaction'을 모사
                w_abs = m.weight.data.abs().reshape(m.weight.shape[0], -1).mean(1)
                g_ema = m.grad_ema.reshape(m.weight.shape[0], -1).mean(1)
                
                # 가중치와 그래디언트 영향력을 결합하여 '잠재력' 평가
                score = w_abs * torch.sqrt(g_ema + 1e-8)
                tpp_scores.append(score)
            
            group_score = torch.mean(torch.stack(tpp_scores), dim=0)

            if len(alive_indices) > 0:
                for i in alive_indices:
                    s_gc = group_score[i].item()
                    target_unit_scores.append(s_gc) 
                    target_unit_costs.append(sum(m.weight.nelement()/m.weight.shape[0] for m in g['layers']))
                    target_unit_metadata.append((g, i))

        # 라그랑주 최적화 적용
        current_sparsity = self.get_current_sparsity() / 100.0
        current_alive_ratio = 1.0 - current_sparsity
        pruned_count = 0

        if current_alive_ratio > total_target_keep_ratio and target_unit_scores:
            incremental_keep_ratio = total_target_keep_ratio / current_alive_ratio
            total_budget = np.sum(target_unit_costs) * incremental_keep_ratio
            optimal_mask_flags = lagrangian_optimization(np.array(target_unit_scores), np.array(target_unit_costs), total_budget)

            with torch.no_grad():
                for idx, is_alive in enumerate(optimal_mask_flags):
                    if not is_alive:
                        group_obj, channel_idx = target_unit_metadata[idx]
                        for ln in group_obj['names']:
                            layer_obj = find_layer(ln)
                            if layer_obj is not None and hasattr(layer_obj, 'mask'):
                                if channel_idx < layer_obj.mask.size(0):
                                    layer_obj.mask[channel_idx] = 0.0
                        pruned_count += 1

        print(f"\n{'='*30} TPP (Personalized) Comparison: Epoch {current_epoch} {'='*30}")
        print(f" [*] Method: TPP (W * sqrt(G_ema)) | Pruned: {pruned_count}")
        # --- 학회용 리소스 분석 출력 ---
        eff = self.get_model_efficiency()
        print(f"\n[Scientific Metrics - Epoch {current_epoch}]")
        print(f" 🟢 Model Size: {eff['orig_mb']:.2f} MB -> {eff['curr_mb']:.2f} MB (Reduction: {eff['orig_mb'] - eff['curr_mb']:.2f} MB)")
        print(f" 🔵 Sparsity (Params/FLOPs): {eff['sparsity']:.2f} %")
        print(f" 🟡 Theoretical Speedup: {eff['speedup']:.2f}x")
        # 실제 측정치 (추론 모드 기준 아님, 현재 학습 세션 기준)
        print(f" 🟠 Current GPU Mem (Allocated): {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"{'='*89}\n")
        torch.cuda.empty_cache()