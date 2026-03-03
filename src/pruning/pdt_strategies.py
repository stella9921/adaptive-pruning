import torch
import torch.nn as nn
from .base import BasePruner
from .engine.hessian_free import SNOWSEngine
import numpy as np
from .optimizer import lagrangian_optimization 
import sys

class PDTPruner(BasePruner):

    # def __init__(self, model, config, args=None, topology_groups=None):
    #     super().__init__(model, config)
        
    #     strat_cfg = config.get('strategy', {})
        
    #     # --- 인자 설정 우선순위 로직 ---
    #     self.group_selection_ratio = strat_cfg.get('group_selection_ratio', 1.0)
    #     self.final_keep_ratio = strat_cfg.get('channel_keep_ratio', 0.2)
    #     self.min_survival_ratio = strat_cfg.get('min_survival_ratio', 0.1)

    #     if args:
    #         if hasattr(args, 'group_selection_ratio') and args.group_selection_ratio is not None:
    #             self.group_selection_ratio = args.group_selection_ratio
    #         if hasattr(args, 'channel_keep_ratio') and args.channel_keep_ratio is not None:
    #             self.final_keep_ratio = args.channel_keep_ratio
    #         if hasattr(args, 'min_survival_ratio') and args.min_survival_ratio is not None:
    #             self.min_survival_ratio = args.min_survival_ratio

    #     self.ema_decay = strat_cfg.get('ema_decay', 0.95)
    #     self.lambda_h = strat_cfg.get('lambda_h', 0.005)
    #     self.k_horizon = strat_cfg.get('k_horizon', 25)
    #     self.engine = SNOWSEngine(n_iter=strat_cfg.get('hessian_iter', 10))

    #     self.topology_groups = topology_groups
    #     self.layers_dict = nn.ModuleDict()
        
        
    #     # 🔥 [추가] 만약 전달받은 그룹이 없다면, 직접 레이어 단위 그룹 생성
    #     if self.topology_groups is None or len(self.topology_groups) == 0:
    #         print("[WARNING] topology_groups is empty! Creating fallback groups...")
    #         self.topology_groups = [[name] for name, m in model.named_modules() 
    #                                 if isinstance(m, (nn.Conv2d, nn.Linear))]
        
    #     for name, m in model.named_modules():
    #         if isinstance(m, (nn.Conv2d, nn.Linear)):
    #             self.layers_dict[name.replace('.', '_')] = m
        
    #     self.layers = list(self.layers_dict.values())
    #     self._check_buffers()
        
    #     print(f"\n[Pruner Init] Strict Conv-Topology Mode Activated")
    #     print(f"[*] Applied Min Survival Guarantee: {self.min_survival_ratio*100:.1f}%")
    #     print(f"[*] Only defined groups (1-{len(self.topology_groups) if self.topology_groups else 0}) will be pruned.")
    def __init__(self, model, config, args=None, topology_groups=None):
        super().__init__(model, config)
        
        strat_cfg = config.get('strategy', {})
        self.group_selection_ratio = strat_cfg.get('group_selection_ratio', 1.0)
        self.final_keep_ratio = strat_cfg.get('channel_keep_ratio', 0.2)
        self.min_survival_ratio = strat_cfg.get('min_survival_ratio', 0.1)

        # args 우선순위 적용
        if args:
            if hasattr(args, 'channel_keep_ratio') and args.channel_keep_ratio is not None:
                self.final_keep_ratio = args.channel_keep_ratio

        self.ema_decay = strat_cfg.get('ema_decay', 0.95)
        self.lambda_h = strat_cfg.get('lambda_h', 0.005)
        self.k_horizon = strat_cfg.get('k_horizon', 25)
        self.engine = SNOWSEngine(n_iter=strat_cfg.get('hessian_iter', 10))

        # --- 🔥 [긴급 수정 섹션: 여기가 핵심입니다] ---
        self.topology_groups = topology_groups
        
        # 만약 밖에서 topology_groups를 제대로 안 줬다면? 여기서 직접 생성 (Safety Net)
        if self.topology_groups is None or len(self.topology_groups) == 0:
            print("\n[WARNING] topology_groups is empty! Creating fallback groups from all Conv/Linear...")
            fallback_groups = []
            for name, m in model.named_modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    # 레이어 하나하나를 개별 그룹으로 묶어줌
                    fallback_groups.append([name])
            self.topology_groups = fallback_groups
        # ----------------------------------------------

        self.layers_dict = nn.ModuleDict()
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                self.layers_dict[name.replace('.', '_')] = m
        
        self.layers = list(self.layers_dict.values())
        self._check_buffers()
        
        print(f"\n[Pruner Init] Identified {len(self.topology_groups)} groups for pruning.")
        print(f"[*] Applied Min Survival Guarantee: {self.min_survival_ratio*100:.1f}%")

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
    # def step_pruning(self, loss, current_epoch, total_epochs):
    #     all_modules = dict(self.model.named_modules())
    #     def find_layer(name):
    #         if name in all_modules: return all_modules[name]
    #         return all_modules.get(name.replace('_', '.'))

    #     # [1] 에폭 수 연동 및 목표 생존율 계산
    #     # main.py에서 pdt_engine.total_epochs = total_epochs로 넘겨준 값을 우선 사용
    #     actual_total = getattr(self, 'total_epochs', total_epochs)
    #     progress = current_epoch / actual_total
    #     total_target_keep_ratio = 1.0 - (progress * (1.0 - self.final_keep_ratio))
        
    #     print(f"\n[DEBUG] Pruning Scan | Epoch {current_epoch}/{actual_total} | Target Keep Ratio: {total_target_keep_ratio:.4f}")

    #     # [2] 토폴로지 그룹 수집
    #     group_info_list = []
    #     if self.topology_groups:
    #         for idx, group in enumerate(self.topology_groups):
    #             score_layers = [find_layer(ln) for ln in group 
    #                             if isinstance(find_layer(ln), (nn.Conv2d, nn.Linear))]
    #             if not score_layers: continue
    #             # Grad-EMA 기반 그룹 중요도 산출
    #             w_g = torch.mean(torch.stack([m.grad_ema.mean() for m in score_layers])).item()
    #             group_info_list.append({
    #                 'id': idx+1, 
    #                 'layers': score_layers, 
    #                 'w_g': w_g, 
    #                 'names': group
    #             })

    #     if not group_info_list: return

    #     # [3] 중요도 정렬 및 타겟 선정
    #     sorted_groups = sorted(group_info_list, key=lambda x: x['w_g'])
    #     num_targets = int(len(sorted_groups) * self.group_selection_ratio)
    #     target_group_ids = [g['id'] for g in sorted_groups[:num_targets]]

    #     # [4] Hessian 계산 및 채널별 점수화
    #     target_params = [m.weight for g in sorted_groups[:num_targets] for m in g['layers']]
    #     hv_list = self.engine.get_k_step_hessian_selective(loss, target_params, self.k_horizon)

    #     target_unit_scores = []
    #     target_unit_costs = []
    #     target_unit_metadata = []

    #     hv_idx = 0
    #     for g in sorted_groups[:num_targets]:
    #         mask = g['layers'][0].mask
    #         alive_indices = torch.where(mask > 0.5)[0].cpu().numpy()
    #         total_n = mask.numel()

    #         # 레이어별 Hessian 점수 저장
    #         for m in g['layers']:
    #             hv = hv_list[hv_idx]
    #             h_energy = hv.pow(2).reshape(hv.shape[0], -1).mean(1)
    #             # Hessian 정규화 (Min-Max)
    #             if h_energy.max() > h_energy.min():
    #                 h_energy = (h_energy - h_energy.min()) / (h_energy.max() - h_energy.min() + 1e-8)
    #             m.hessian_score.copy_(h_energy)
    #             hv_idx += 1

    #         # 최소 생존 보장선 확인
    #         if (len(alive_indices) / total_n) <= self.min_survival_ratio:
    #             continue

    #         # 살아있는 채널들에 대해 가중치 점수 산출
    #         if len(alive_indices) > 0:
    #             for i in alive_indices:
    #                 s_gc_list = [m.hessian_score[i].item() for m in g['layers'] if i < m.hessian_score.size(0)]
    #                 s_gc = sum(s_gc_list)/len(s_gc_list) if s_gc_list else 0.0
                    
    #                 # 최종 Saliency Score = Grad_EMA * (Hessian * Lambda)
    #                 raw_score = g['w_g'] * (s_gc * self.lambda_h)
                    
    #                 target_unit_scores.append(raw_score) 
    #                 target_unit_costs.append(sum(m.weight.nelement()/m.weight.shape[0] for m in g['layers']))
    #                 target_unit_metadata.append((g, i))

    #     # [5] 프루닝 최적화 및 마스크 적용 (VRAM 측정 및 강제 집행)
    #     mem_before = torch.cuda.memory_allocated() / (1024**2)
        
    #     # 현재 실제 생존율 계산 (0~1 범위로 보정)
    #     curr_sp_val = self.get_current_sparsity()
    #     current_alive_ratio = 1.0 - (curr_sp_val / 100.0)
        
    #     pruned_count = 0
    #     cutoff_threshold = 0.0

    #     # 조건 진입 (-0.01 버퍼)
    #     if target_unit_scores and (current_alive_ratio > total_target_keep_ratio - 0.01):
    #         incremental_keep_ratio = total_target_keep_ratio / (current_alive_ratio + 1e-7)
    #         total_budget = np.sum(target_unit_costs) * incremental_keep_ratio
            
    #         # 라그랑주 최적화 호출
    #         optimal_mask_flags = lagrangian_optimization(
    #             np.array(target_unit_scores), 
    #             np.array(target_unit_costs), 
    #             total_budget
    #         )
            
    #         # 🔥 [Emergency FORCE] 라그랑주가 너무 보수적일 때 강제로 하위 유닛 쳐냄
    #         if np.all(optimal_mask_flags == 1) and incremental_keep_ratio < 0.999:
    #             num_force = int(len(optimal_mask_flags) * (1 - incremental_keep_ratio))
    #             if num_force > 0:
    #                 force_idx = np.argsort(target_unit_scores)[:num_force]
    #                 optimal_mask_flags[force_idx] = 0
    #                 print(f" [FORCE] Lagrangian bypassed. Forcing {num_force} units to 0.")

    #         # 마스크 임계값 기록용
    #         dead_scores = np.array(target_unit_scores)[optimal_mask_flags == 0]
    #         if len(dead_scores) > 0: cutoff_threshold = np.max(dead_scores)

    #         # 실제 마스크 버퍼 업데이트
    #         with torch.no_grad():
    #             for idx, is_alive in enumerate(optimal_mask_flags):
    #                 if not is_alive:
    #                     group_obj, channel_idx = target_unit_metadata[idx]
    #                     for ln in group_obj['names']:
    #                         layer_obj = find_layer(ln)
    #                         if layer_obj is not None and hasattr(layer_obj, 'mask'):
    #                             if channel_idx < layer_obj.mask.size(0):
    #                                 layer_obj.mask.data[channel_idx] = 0.0 # .data 물리적 수정
    #                                 pruned_count += 1
            
    #         self.apply_mask_to_weights()
    #         torch.cuda.empty_cache()

    #     mem_after = torch.cuda.memory_allocated() / (1024**2)
    #     print(f" [VRAM Status] Pruning step memory: {mem_before:.1f}MB -> {mem_after:.1f}MB (Saved: {max(0, mem_before-mem_after):.1f}MB)")

    #     # [6] 상세 토폴로지 상태 출력 (무조건 출력)
    #     print(f"\n{'='*30} PDT Pruning Status: Epoch {current_epoch} {'='*30}")
    #     print(f" [*] Cut-off Threshold: {cutoff_threshold:.6f} | Pruned Units this step: {pruned_count}")
    #     print(f" {'Group ID':<10} | {'Alive/Total':>12} | {'Sparsity':>8} | {'Hessian(avg)':>12} | {'Status'}")
    #     print(f" {'-'*87}")
        
    #     group_info_list.sort(key=lambda x: x['id'])
    #     for g in group_info_list:
    #         m = g['layers'][0]
    #         total = m.mask.numel()
    #         alive = int(m.mask.data.sum().item())
    #         sparsity = (1 - alive/total) * 100
    #         h_avg = m.hessian_score.mean().item()
            
    #         status = "TARGET" if g['id'] in target_group_ids else "FIXED"
    #         if (alive/total) <= self.min_survival_ratio: status = "MIN-SURV"
    #         print(f" Group {g['id']:2d}      | {alive:4d}/{total:4d}      | {sparsity:>7.1f}% | {h_avg:>12.6f} | [{status}]")
        
    #     # [7] 학회용 리소스 요약 분석 출력
    #     eff = self.get_model_efficiency()
    #     print(f"\n[Scientific Metrics - Epoch {current_epoch}]")
    #     print(f" 🟢 Model Size: {eff['orig_mb']:.2f} MB -> {eff['curr_mb']:.2f} MB (Reduction: {eff['orig_mb'] - eff['curr_mb']:.2f} MB)")
    #     print(f" 🔵 Sparsity (Params/FLOPs): {eff['sparsity']:.2f} %")
    #     print(f" 🟡 Theoretical Speedup: {eff['speedup']:.2f}x")
    #     print(f" 🟠 Current GPU Mem (Allocated): {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    #     print(f"{'='*89}\n")
    #     torch.cuda.empty_cache()
    def step_pruning(self, loss, current_epoch, total_epochs):
        all_modules = dict(self.model.named_modules())
        def find_layer(name):
            if name in all_modules: return all_modules[name]
            return all_modules.get(name.replace('_', '.'))

        

        # --- 🔥 [진짜 최종 수정] ---
        # 1. 먼저 self.topology_groups가 있는지 확인
        # 2. 있다면 active_groups에 할당
        # 3. 없다면 그제서야 Fallback 실행
        if hasattr(self, 'topology_groups') and self.topology_groups is not None and len(self.topology_groups) > 0:
            active_groups = self.topology_groups
            # print(f"[DEBUG] Success! Using {len(active_groups)} groups.")
        else:
            # 이 코드는 ResNet에서 topology_groups 배달사고 났을 때만 실행되어야 함
            print("[WARNING] Fallback scan initiated...")
            active_groups = [[name] for name, m in self.model.named_modules() 
                             if isinstance(m, (nn.Conv2d, nn.Linear))]
        # ---------------------------



        # [긴급] 그룹 정보 강제 복구
        # if not hasattr(self, 'topology_groups') or self.topology_groups is None:
        #     self.topology_groups = [[name] for name, m in self.model.named_modules() 
        #                             if isinstance(m, (nn.Conv2d, nn.Linear))]

        actual_total = getattr(self, 'total_epochs', total_epochs)
        progress = current_epoch / actual_total
        total_target_keep_ratio = 1.0 - (progress * (1.0 - self.final_keep_ratio))
        
        print(f"\n[DEBUG] !!! PRUNING TRIGGERED !!! Epoch {current_epoch}/{actual_total}")

        # [핵심 수정] 모든 타입의 레이어를 일단 수집
        group_info_list = []
        for idx, group in enumerate(active_groups):
            # Conv/Linear 뿐만 아니라 마스크가 있는 모든 레이어를 찾음
            score_layers = []
            for ln in group:
                layer = find_layer(ln)
                if layer is not None and (isinstance(layer, (nn.Conv2d, nn.Linear)) or hasattr(layer, 'mask')):
                    score_layers.append(layer)
            
            if not score_layers: continue
            
            # Grad-EMA 평균 (안전하게 처리)
            valid_emas = [m.grad_ema.mean() for m in score_layers if hasattr(m, 'grad_ema')]
            w_g = torch.mean(torch.stack(valid_emas)).item() if valid_emas else 0.0
            
            group_info_list.append({
                'id': idx+1, 'layers': score_layers, 'w_g': w_g, 'names': group
            })

        if not group_info_list:
            print("[ERROR] Still no groups found! Forcing all Conv layers into groups...")
            # 강제로 모든 Conv를 그룹화
            for name, m in self.model.named_modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    group_info_list.append({'id': 999, 'layers': [m], 'w_g': 0.0, 'names': [name]})

        # --- [이후 Hessian 및 Pruning 로직] ---
        target_params = [m.weight for g in group_info_list for m in g['layers'] if hasattr(m, 'weight')]
        hv_list = self.engine.get_k_step_hessian_selective(loss, target_params, self.k_horizon)

        target_unit_scores, target_unit_costs, target_unit_metadata = [], [], []
        hv_idx = 0
        for g in group_info_list:
            # 그룹 내 첫 번째 레이어를 기준으로 살아있는 인덱스 확인
            base_layer = g['layers'][0]
            if not hasattr(base_layer, 'mask'): continue
            
            alive_indices = torch.where(base_layer.mask > 0.5)[0].cpu().numpy()
            
            for m in g['layers']:
                if not hasattr(m, 'weight'): continue
                hv = hv_list[hv_idx]; hv_idx += 1
                h_energy = hv.pow(2).reshape(hv.shape[0], -1).mean(1)
                if h_energy.max() > h_energy.min():
                    h_energy = (h_energy - h_energy.min()) / (h_energy.max() - h_energy.min() + 1e-8)
                m.hessian_score.copy_(h_energy)

            for i in alive_indices:
                scores = [m.hessian_score[i].item() for m in g['layers'] if hasattr(m, 'hessian_score') and i < m.hessian_score.size(0)]
                s_gc = sum(scores)/len(scores) if scores else 0.0
                target_unit_scores.append(g['w_g'] * (s_gc * self.lambda_h))
                target_unit_costs.append(sum(m.weight.nelement()/m.weight.shape[0] for m in g['layers'] if hasattr(m, 'weight')))
                target_unit_metadata.append((g, i))

        # [최적화 및 집행]
        mem_before = torch.cuda.memory_allocated() / (1024**2)
        curr_sp = self.get_current_sparsity()
        current_alive_ratio = 1.0 - (curr_sp / 100.0)
        pruned_count = 0

        if target_unit_scores and (current_alive_ratio > total_target_keep_ratio - 0.05):
            target_ratio = total_target_keep_ratio / (current_alive_ratio + 1e-7)
            optimal_mask_flags = lagrangian_optimization(np.array(target_unit_scores), np.array(target_unit_costs), np.sum(target_unit_costs) * target_ratio)
            
            # FORCE
            if np.all(optimal_mask_flags == 1) and target_ratio < 0.999:
                num_force = int(len(optimal_mask_flags) * (1 - target_ratio))
                optimal_mask_flags[np.argsort(target_unit_scores)[:num_force]] = 0

            with torch.no_grad():
                for idx, is_alive in enumerate(optimal_mask_flags):
                    if not is_alive:
                        g_obj, ch_idx = target_unit_metadata[idx]
                        for ln in g_obj['names']:
                            l_obj = find_layer(ln)
                            if l_obj is not None and hasattr(l_obj, 'mask'):
                                if ch_idx < l_obj.mask.size(0):
                                    l_obj.mask.data[ch_idx] = 0.0
                                    pruned_count += 1
            self.apply_mask_to_weights()
            torch.cuda.empty_cache()

        # [결과 출력 - 무조건 실행]
        mem_after = torch.cuda.memory_allocated() / (1024**2)
        print(f"\n{'='*30} PDT Pruning Report: Epoch {current_epoch} {'='*30}")
        print(f" [*] Pruned this step: {pruned_count} units | VRAM: {mem_before:.1f}MB -> {mem_after:.1f}MB")
        
        for g in sorted(group_info_list, key=lambda x: x['id']):
            m = g['layers'][0]
            if not hasattr(m, 'mask'): continue
            t, a = m.mask.numel(), int(m.mask.data.sum().item())
            print(f" Group {g['id']:2d} | {a:4d}/{t:4d} | {(1-a/t)*100:>7.1f}% | Hessian: {m.hessian_score.mean().item():.6f}")

        eff = self.get_model_efficiency()
        print(f"\n[Scientific Metrics] 🟢 Size: {eff['curr_mb']:.2f}MB | 🔵 Sparsity: {eff['sparsity']:.2f}% | 🟡 Speedup: {eff['speedup']:.2f}x")
        print(f"{'='*89}\n")
    # def step_pruning(self, loss, current_epoch, total_epochs):
    #     all_modules = dict(self.model.named_modules())
    #     def find_layer(name):
    #         if name in all_modules: return all_modules[name]
    #         return all_modules.get(name.replace('_', '.'))

    #     # --- 🔥 [긴급 처방] 그룹이 없으면 여기서 즉시 직접 만듭니다 ---
    #     if not hasattr(self, 'topology_groups') or self.topology_groups is None or len(self.topology_groups) == 0:
    #         print("[WARNING] topology_groups lost! Re-scanning layers directly...")
    #         # ResNet-18의 모든 Conv/Linear 레이어를 개별 그룹으로 강제 등록
    #         self.topology_groups = [[name] for name, m in self.model.named_modules() 
    #                                 if isinstance(m, (nn.Conv2d, nn.Linear))]
    #     # --------------------------------------------------------

    #     # [1] 에폭 수 연동 및 목표 생존율 계산
    #     actual_total = getattr(self, 'total_epochs', total_epochs)
    #     progress = current_epoch / actual_total
    #     total_target_keep_ratio = 1.0 - (progress * (1.0 - self.final_keep_ratio))
        
    #     print(f"\n[DEBUG] !!! PRUNING TRIGGERED !!! Epoch {current_epoch}/{actual_total}")
    #     print(f"[DEBUG] Target Keep Ratio: {total_target_keep_ratio:.4f} | Groups: {len(self.topology_groups)}")

    #     # [2] 토폴로지 그룹 수집
    #     group_info_list = []
    #     for idx, group in enumerate(self.topology_groups or []):
    #         score_layers = [find_layer(ln) for ln in group if isinstance(find_layer(ln), (nn.Conv2d, nn.Linear))]
    #         if not score_layers: continue
    #         w_g = torch.mean(torch.stack([m.grad_ema.mean() for m in score_layers])).item()
    #         group_info_list.append({'id': idx+1, 'layers': score_layers, 'w_g': w_g, 'names': group})

    #     if not group_info_list:
    #         print("[ERROR] No topology groups found. Check Stage 1.")
    #         return

    #     # [3] Hessian 계산
    #     sorted_groups = sorted(group_info_list, key=lambda x: x['w_g'])
    #     num_targets = int(len(sorted_groups) * self.group_selection_ratio)
    #     target_group_ids = [g['id'] for g in sorted_groups[:num_targets]]

    #     target_params = [m.weight for g in sorted_groups[:num_targets] for m in g['layers']]
    #     print(f"[DEBUG] Computing Hessian for {len(target_params)} parameters...")
    #     hv_list = self.engine.get_k_step_hessian_selective(loss, target_params, self.k_horizon)

    #     target_unit_scores, target_unit_costs, target_unit_metadata = [], [], []
    #     hv_idx = 0
    #     for g in sorted_groups[:num_targets]:
    #         mask = g['layers'][0].mask
    #         alive_indices = torch.where(mask > 0.5)[0].cpu().numpy()
    #         for m in g['layers']:
    #             hv = hv_list[hv_idx]; hv_idx += 1
    #             h_energy = hv.pow(2).reshape(hv.shape[0], -1).mean(1)
    #             if h_energy.max() > h_energy.min():
    #                 h_energy = (h_energy - h_energy.min()) / (h_energy.max() - h_energy.min() + 1e-8)
    #             m.hessian_score.copy_(h_energy)

    #         # 살아있는 채널 점수화
    #         for i in alive_indices:
    #             s_gc_list = [m.hessian_score[i].item() for m in g['layers'] if i < m.hessian_score.size(0)]
    #             s_gc = sum(s_gc_list)/len(s_gc_list) if s_gc_list else 0.0
    #             target_unit_scores.append(g['w_g'] * (s_gc * self.lambda_h))
    #             target_unit_costs.append(sum(m.weight.nelement()/m.weight.shape[0] for m in g['layers']))
    #             target_unit_metadata.append((g, i))

    #     # [4] 프루닝 최적화 및 강제 집행
    #     mem_before = torch.cuda.memory_allocated() / (1024**2)
    #     curr_sp_val = self.get_current_sparsity()
    #     current_alive_ratio = 1.0 - (curr_sp_val / 100.0)
    #     pruned_count = 0

    #     # 무조건 진입 시도 (Target보다 높으면 실행)
    #     if target_unit_scores and (current_alive_ratio > total_target_keep_ratio - 0.05):
    #         target_ratio_within_alive = total_target_keep_ratio / (current_alive_ratio + 1e-7)
    #         total_budget = np.sum(target_unit_costs) * target_ratio_within_alive
            
    #         optimal_mask_flags = lagrangian_optimization(np.array(target_unit_scores), np.array(target_unit_costs), total_budget)
            
    #         # 🔥 [강제 트리거] 라그랑주가 안 자르면 점수 낮은 순으로 강제 집행
    #         if np.all(optimal_mask_flags == 1) and target_ratio_within_alive < 0.999:
    #             num_force = int(len(optimal_mask_flags) * (1 - target_ratio_within_alive))
    #             if num_force > 0:
    #                 force_idx = np.argsort(target_unit_scores)[:num_force]
    #                 optimal_mask_flags[force_idx] = 0
    #                 print(f" [FORCE] Forced pruning {num_force} units.")

    #         with torch.no_grad():
    #             for idx, is_alive in enumerate(optimal_mask_flags):
    #                 if not is_alive:
    #                     group_obj, channel_idx = target_unit_metadata[idx]
    #                     for ln in group_obj['names']:
    #                         layer_obj = find_layer(ln)
    #                         if layer_obj is not None and hasattr(layer_obj, 'mask'):
    #                             if channel_idx < layer_obj.mask.size(0):
    #                                 layer_obj.mask.data[channel_idx] = 0.0
    #                                 pruned_count += 1
    #         self.apply_mask_to_weights()
    #         torch.cuda.empty_cache()

    #     mem_after = torch.cuda.memory_allocated() / (1024**2)
    #     print(f" [VRAM Status] Memory Change: {mem_before:.1f}MB -> {mem_after:.1f}MB")

    #     # [5] 결과 출력 (이 위치가 함수 끝에서 무조건 호출되도록 고정)
    #     print(f"\n{'='*30} PDT Pruning Report: Epoch {current_epoch} {'='*30}")
    #     print(f" [*] Pruned this step: {pruned_count} units")
    #     print(f" {'Group ID':<10} | {'Alive/Total':>12} | {'Sparsity':>8} | {'Hessian(avg)':>12}")
    #     print(f" {'-'*80}")
        
    #     for g in sorted(group_info_list, key=lambda x: x['id']):
    #         m = g['layers'][0]
    #         t, a = m.mask.numel(), int(m.mask.data.sum().item())
    #         h_avg = m.hessian_score.mean().item()
    #         print(f" Group {g['id']:2d}      | {a:4d}/{t:4d}      | {(1-a/t)*100:>7.1f}% | {h_avg:>12.6f}")

    #     eff = self.get_model_efficiency()
    #     print(f"\n[Scientific Metrics]")
    #     print(f" 🟢 Model Size: {eff['curr_mb']:.2f} MB | 🔵 Sparsity: {eff['sparsity']:.2f} % | 🟡 Speedup: {eff['speedup']:.2f}x")
    #     print(f"{'='*89}\n")
    
    # def step_pruning(self, loss, current_epoch, total_epochs):
    #     all_modules = dict(self.model.named_modules())
    #     def find_layer(name):
    #         if name in all_modules: return all_modules[name]
    #         return all_modules.get(name.replace('_', '.'))

    #     # [수정] main.py에서 전달받은 실제 total_epochs가 있다면 우선 사용
    #     actual_total = getattr(self, 'total_epochs', total_epochs)
    #     progress = current_epoch / actual_total
    #     total_target_keep_ratio = 1.0 - (progress * (1.0 - self.final_keep_ratio))
        
    #     print(f"\n[DEBUG] Pruning Scan | Epoch {current_epoch}/{actual_total} | Target Keep Ratio: {total_target_keep_ratio:.4f}")

    #     # 2. 토폴로지 그룹 수집
    #     group_info_list = []
    #     if self.topology_groups:
    #         for idx, group in enumerate(self.topology_groups):
    #             score_layers = [find_layer(ln) for ln in group 
    #                             if isinstance(find_layer(ln), (nn.Conv2d, nn.Linear))]
    #             if not score_layers: continue
    #             w_g = torch.mean(torch.stack([m.grad_ema.mean() for m in score_layers])).item()
    #             group_info_list.append({
    #                 'id': idx+1, 
    #                 'layers': score_layers, 
    #                 'w_g': w_g, 
    #                 'names': group
    #             })

    #     if not group_info_list: return

    #     # 3. 중요도 정렬 및 타겟 선정
    #     sorted_groups = sorted(group_info_list, key=lambda x: x['w_g'])
    #     num_targets = int(len(sorted_groups) * self.group_selection_ratio)
    #     target_group_ids = [g['id'] for g in sorted_groups[:num_targets]]

    #     # --- [4] Hessian 계산 및 2단계 정규화 ---
    #     target_params = [m.weight for g in sorted_groups[:num_targets] for m in g['layers']]
    #     hv_list = self.engine.get_k_step_hessian_selective(loss, target_params, self.k_horizon)

    #     target_unit_scores = []
    #     target_unit_costs = []
    #     target_unit_metadata = []

    #     hv_idx = 0
    #     for g in sorted_groups[:num_targets]:
    #         mask = g['layers'][0].mask
    #         alive_indices = torch.where(mask > 0.5)[0].cpu().numpy()
    #         total_n = mask.numel()

    #         for m in g['layers']:
    #             hv = hv_list[hv_idx]
    #             h_energy = hv.pow(2).reshape(hv.shape[0], -1).mean(1)
    #             if h_energy.max() > h_energy.min():
    #                 h_energy = (h_energy - h_energy.min()) / (h_energy.max() - h_energy.min() + 1e-8)
    #             m.hessian_score.copy_(h_energy)
    #             hv_idx += 1

    #         if (len(alive_indices) / total_n) <= self.min_survival_ratio:
    #             if g['id'] in target_group_ids: target_group_ids.remove(g['id'])
    #             continue

    #         if len(alive_indices) > 0:
    #             for i in alive_indices:
    #                 s_gc_list = [m.hessian_score[i].item() for m in g['layers'] if i < m.hessian_score.size(0)]
    #                 s_gc = sum(s_gc_list)/len(s_gc_list) if s_gc_list else 0.0
    #                 raw_score = g['w_g'] * (s_gc * self.lambda_h)
                    
    #                 target_unit_scores.append(raw_score) 
    #                 target_unit_costs.append(sum(m.weight.nelement()/m.weight.shape[0] for m in g['layers']))
    #                 target_unit_metadata.append((g, i))

    #     # --- [5] 프루닝 최적화 및 마스크 적용 (수정 완료 버전) ---
    #     mem_before = torch.cuda.memory_allocated() / (1024**2)
        
    #     # [정밀 수정] % 단위를 비율(0~1) 단위로 보정
    #     curr_sp_val = self.get_current_sparsity() 
    #     current_alive_ratio = 1.0 - (curr_sp_val / 100.0)
        
    #     pruned_count = 0
    #     cutoff_threshold = 0.0

    #     # [조건 완화] 소수점 오차 방지를 위해 -0.01 버퍼 부여
    #     if target_unit_scores and (current_alive_ratio > total_target_keep_ratio - 0.01):
    #         incremental_keep_ratio = total_target_keep_ratio / (current_alive_ratio + 1e-7)
    #         total_budget = np.sum(target_unit_costs) * incremental_keep_ratio
            
    #         optimal_mask_flags = lagrangian_optimization(
    #             np.array(target_unit_scores), 
    #             np.array(target_unit_costs), 
    #             total_budget
    #         )
            
    #         # 🔥 [FORCE] 라그랑주가 너무 보수적일 때 강제 집행 (ResNet 필수 로직)
    #         if np.all(optimal_mask_flags == 1) and incremental_keep_ratio < 0.999:
    #             num_force = int(len(optimal_mask_flags) * (1 - incremental_keep_ratio))
    #             if num_force > 0:
    #                 force_idx = np.argsort(target_unit_scores)[:num_force]
    #                 optimal_mask_flags[force_idx] = 0
    #                 print(f" [FORCE] Lagrangian bypassed. Forcing {num_force} units to 0.")

    #         dead_scores = np.array(target_unit_scores)[optimal_mask_flags == 0]
    #         if len(dead_scores) > 0: cutoff_threshold = np.max(dead_scores)

    #         with torch.no_grad():
    #             for idx, is_alive in enumerate(optimal_mask_flags):
    #                 if not is_alive:
    #                     group_obj, channel_idx = target_unit_metadata[idx]
    #                     for ln in group_obj['names']:
    #                         layer_obj = find_layer(ln)
    #                         if layer_obj is not None and hasattr(layer_obj, 'mask'):
    #                             if channel_idx < layer_obj.mask.size(0):
    #                                 # .data.fill_(0.0)로 물리적 버퍼 업데이트 보장
    #                                 layer_obj.mask.data[channel_idx] = 0.0
    #                                 pruned_count += 1
            
    #         self.apply_mask_to_weights()
    #         torch.cuda.empty_cache()

    #     mem_after = torch.cuda.memory_allocated() / (1024**2)
    #     print(f" [VRAM Status] Step Memory: {mem_before:.1f}MB -> {mem_after:.1f}MB (Saved: {max(0, mem_before-mem_after):.1f}MB)")

    #     # 🔥 [출력 위치 변경] IF문 밖으로 빼서 무조건 출력되도록 함
    #     print(f"\n{'='*30} PDT Pruning Status: Epoch {current_epoch} {'='*30}")
    #     print(f" [*] Cut-off Threshold: {cutoff_threshold:.6f} | Pruned Units this step: {pruned_count}")
    #     print(f" {'Group ID':<10} | {'Alive/Total':>12} | {'Sparsity':>8} | {'Hessian(avg)':>12} | {'Status'}")
    #     print(f" {'-'*87}")
        
    #     group_info_list.sort(key=lambda x: x['id'])
    #     for g in group_info_list:
    #         m = g['layers'][0]
    #         total, alive = m.mask.numel(), int(m.mask.data.sum().item())
    #         sparsity = (1 - alive/total) * 100
    #         h_avg = g['layers'][0].hessian_score.mean().item() if g['id'] in target_group_ids else 0.0
            
    #         status = "TARGET" if g['id'] in target_group_ids else "FIXED"
    #         if (alive/total) <= self.min_survival_ratio: status = "MIN-SURV"
    #         print(f" Group {g['id']:2d}      | {alive:4d}/{total:4d}      | {sparsity:>7.1f}% | {h_avg:>12.6f} | [{status}]")
        
    #     # 학회용 리소스 분석 출력
    #     eff = self.get_model_efficiency()
    #     print(f"\n[Scientific Metrics - Epoch {current_epoch}]")
    #     print(f" 🟢 Model Size: {eff['orig_mb']:.2f} MB -> {eff['curr_mb']:.2f} MB")
    #     print(f" 🔵 Sparsity (Params/FLOPs): {eff['sparsity']:.2f} %")
    #     print(f" 🟡 Theoretical Speedup: {eff['speedup']:.2f}x")
    #     print(f" 🟠 Current GPU Mem (Allocated): {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    #     print(f"{'='*89}\n")
    #     torch.cuda.empty_cache()
    
    
    
    def get_current_sparsity(self):
        total_p = sum(m.mask.numel() for m in self.layers)
        active_p = sum(m.mask.sum().item() for m in self.layers)
        return (1.0 - (active_p / total_p)) * 100.0 if total_p > 0 else 0.0

    def get_model_efficiency(self, example_inputs=None):
        """FLOPs, Latency(Proxy), Memory, Size를 이론적으로 계산"""
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

    def _global_rank_prune(self, scores, metadata, total_epochs,epoch, method_name):
        """라그랑주 대신 모든 경쟁 기법이 공통으로 사용할 Global Ranking 실행부"""
        # 1. 현재 목표 Sparsity에 따라 자를 개수 계산
        progress = epoch /  total_epochs  
        total_target_sparsity = progress * (1.0 - self.final_keep_ratio)
        
        num_total = len(scores)
        num_to_prune = int(num_total * total_target_sparsity)

        if num_to_prune > 0:
            # 2. 점수가 낮은 순서대로 정렬 (Global Ranking)
            indices = np.argsort(scores)[:num_to_prune]
            for idx in indices:
                group_obj, channel_idx = metadata[idx]
                for ln in group_obj['names']:
                    layer = self.model.get_submodule(ln.replace('_', '.'))
                    if hasattr(layer, 'mask'):
                        # [수정 포인트] 자르려는 채널 번호(channel_idx)가 
                        # 실제 레이어의 마스크 크기보다 작을 때만 실행
                        if channel_idx < layer.mask.size(0):
                            layer.mask[channel_idx] = 0.0
                        
        
        print(f"\n{'='*30} {method_name} Global Pruning: Epoch {epoch} {'='*30}")
        print(f" [*] Method: {method_name} | Pruned: {num_to_prune}")
        
        # 3.학회용 측정치 출력 (통합 호출)
        eff = self.get_model_efficiency()
        print(f"\n[Scientific Metrics - Epoch {epoch}]")
        print(f" 🟢 Model Size: {eff['orig_mb']:.2f} MB -> {eff['curr_mb']:.2f} MB")
        print(f" 🔵 Sparsity (Params/FLOPs): {eff['sparsity']:.2f} %")
        print(f" 🟡 Theoretical Speedup: {eff['speedup']:.2f}x")
        print(f" 🟠 Current GPU Mem (Allocated): {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"{'='*89}\n")
        torch.cuda.empty_cache()


# ==============================================================================
# HAP (Hessian-Aware Pruning) 비교 실험용 Pruner
# ==============================================================================
# ==============================================================================
# HAP (Hessian-Aware Pruning) 정석 구현 버전
# ==============================================================================
class HAPPruner(PDTPruner):
    """
    HAP 논문 로직 (Emergency Mode):
    1. 모델 전체 Conv/Linear의 Hessian Trace를 직접 계산.
    2. Trace의 역수(Sensitivity)에 비례하여 레이어별 Sparsity를 동적 할당.
    3. 각 레이어 내에서도 Hessian 에너지가 낮은 채널부터 제거.
    """
    def step_pruning(self, loss, current_epoch, total_epochs):
        print(f"\n[DEBUG] === HAP Emergency Pruning: Epoch {current_epoch} ===")
        
        # 1. 대상 레이어 직접 수집 (Naming Bridge 무시)
        target_layers = []
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if not hasattr(m, 'mask'):
                    n_f = m.weight.shape[0]
                    m.register_buffer("mask", torch.ones(n_f, device=m.weight.device))
                target_layers.append((name, m))

        if not target_layers:
            print("[DEBUG] 🚨 FATAL: No layers found for HAP.")
            return

        # 2. 전체 목표 Sparsity 계산 (스케줄링)
        progress = current_epoch / total_epochs
        # total_target_sparsity = progress * (1.0 - self.final_keep_ratio)
        total_target_sparsity = min(progress * 2.0, 1.0) * (1.0 - self.final_keep_ratio)

        # 3. Hessian Trace 계산 및 레이어별 민감도 산출
        target_params = [m.weight for name, m in target_layers]
        hv_list = self.engine.get_k_step_hessian_selective(loss, target_params, self.k_horizon)

        layer_traces = []
        layer_hessian_energies = []
        
        for idx, (name, m) in enumerate(target_layers):
            hv = hv_list[idx]
            with torch.no_grad():
                # 필터별 에너지 [Out_channels]
                h_energy = hv.pow(2).reshape(hv.shape[0], -1).mean(1)
                m.hessian_score.copy_(h_energy)
                
                # 레이어 전체의 평균 Trace
                trace = h_energy.mean().item()
                layer_traces.append(trace)
                layer_hessian_energies.append(h_energy)

        # # 4. [HAP 핵심] 민감도(1/Trace) 기반 Sparsity 배분
        # # sensitivities = [1.0 / (t + 1e-8) for t in layer_traces]/
        # sensitivities = [(1.0 / (t + 1e-8))**0.5 for t in layer_traces]
        # avg_sens = sum(sensitivities) / len(sensitivities)

        # actual_pruned_total = 0
        # [수정] 4. 민감도 기반 Sparsity 배분 (더 공격적으로)
        # 루트(0.5) 대신 1.0승 혹은 그 이상을 사용하여 레이어 간 차이를 극대화
        sensitivities = [1.0 / (t + 1e-10) for t in layer_traces]
        avg_sens = sum(sensitivities) / len(sensitivities)

        actual_pruned_total = 0
        
        # [수정] 5. 레이어별 개별 프루닝 실행
        for i, (name, m) in enumerate(target_layers):
            # 레이어별 민감도 비율에 가중치를 부여 (예: 1.2배)
            # 이를 통해 가성비 좋은 레이어는 더 과감하게 깎음
            relative_sens = sensitivities[i] / avg_sens
            layer_target_sparsity = total_target_sparsity * relative_sens
            
            # [핵심 수정] 최소 생존 보장 비율을 대폭 낮춤 (예: 10%만 남겨도 생존)
            # self.min_survival_ratio가 너무 높다면 여기서 강제로 0.1 등으로 낮춰보세요.
            max_prunable = 1.0 - 0.1  # 최소 10%는 남김
            layer_target_sparsity = min(layer_target_sparsity, max_prunable)
            
            num_channels = m.mask.numel()
            # 이미 잘린 채널을 제외하고 추가로 자르는 것이 아니라, 
            # '전체 채널 중 이만큼이 0이어야 한다'는 목표치로 설정
            num_to_be_zero = int(num_channels * layer_target_sparsity)
            
            # 현재 이미 0인 개수 파악
            current_zero = int(num_channels - m.mask.sum().item())
            
            # 추가로 더 잘라야 할 개수 계산
            num_prune = num_to_be_zero - current_zero

            if num_prune > 0:
                # 살아있는(mask==1) 채널 중에서 Hessian 점수가 낮은 것 선택
                alive_indices = torch.where(m.mask > 0.5)[0]
                if len(alive_indices) > 0:
                    # 살아있는 채널의 Hessian Score만 추출
                    alive_scores = m.hessian_score[alive_indices]
                    
                    # 그 중 하위 k개 선택
                    k = min(num_prune, len(alive_indices))
                    _, sub_indices = torch.topk(alive_scores, k=k, largest=False)
                    prune_indices = alive_indices[sub_indices]
                    
                    with torch.no_grad():
                        m.mask[prune_indices] = 0.0
                        actual_pruned_total += len(prune_indices)
        # # 5. 레이어별 개별 프루닝 실행
        # for i, (name, m) in enumerate(target_layers):
        #     # 레이어별 할당 Sparsity = (전체 목표) * (상대적 민감도 비율)
        #     group_sparsity = total_target_sparsity * (sensitivities[i] / avg_sens)
        #     # 최소 생존 보장 (90% 이상은 안 자름)
        #     group_sparsity = min(group_sparsity, 1.0 - self.min_survival_ratio)
            
        #     num_channels = m.mask.numel()
        #     num_prune = int(num_channels * group_sparsity)

        #     if num_prune > 0:
        #         # Hessian 점수가 낮은 순으로 채널 선정
        #         _, prune_indices = torch.topk(m.hessian_score, k=num_prune, largest=False)
                
        #         with torch.no_grad():
        #             m.mask[prune_indices] = 0.0
        #             actual_pruned_total += num_prune

        # 6. 물리적 가중치 적용 및 결과 출력
        self.apply_mask_to_weights()
        
        print(f"\n{'='*30} HAP Emergency Pruning: Epoch {current_epoch} {'='*30}")
        print(f" [*] Method: HAP | Total Pruned Units: {actual_pruned_total}")
        
        # 리소스 측정 및 출력
        eff = self.get_model_efficiency()
        print(f"\n[Scientific Metrics - Epoch {current_epoch}]")
        print(f" 🟢 Model Size: {eff['orig_mb']:.2f} MB -> {eff['curr_mb']:.2f} MB")
        print(f" 🔵 Sparsity: {eff['sparsity']:.2f} %")
        print(f" 🟡 Theoretical Speedup: {eff['speedup']:.2f}x")
        print(f"{'='*89}\n")
        torch.cuda.empty_cache()
class SNOWSPruner(PDTPruner):
    """
    SNOWS 논문 로직 (Emergency Mode): 
    중요도 = 순수 Hessian_Trace (H-Vector Product의 2-norm 제곱 평균)
    토폴로지 그룹 무시하고 모델 전체의 모든 Conv/Linear를 직접 수집하여 프루닝합니다.
    """
    def step_pruning(self, loss, current_epoch, total_epochs):
        print(f"\n[DEBUG] === SNOWS Emergency Pruning: Epoch {current_epoch} ===")
        
        # 1. 대상 레이어 수집 (ATO와 동일한 저인망식 방식)
        target_layers = []
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if not hasattr(m, 'mask'):
                    n_f = m.weight.shape[0]
                    m.register_buffer("mask", torch.ones(n_f, device=m.weight.device))
                target_layers.append((name, m))

        if not target_layers:
            print("[DEBUG] 🚨 FATAL: No Conv2d/Linear layers found for SNOWS.")
            return

        # 2. Hessian Trace 계산 (SNOWS 엔진 활용)
        target_params = [m.weight for name, m in target_layers]
        # SNOWS는 배치 데이터를 통해 현재 시점의 Hessian 에너지를 추출합니다.
        hv_list = self.engine.get_k_step_hessian_selective(loss, target_params, self.k_horizon)

        target_unit_scores = []
        target_unit_metadata = []

        # 3. 채널별 Hessian Energy 점수 매기기
        for idx, (name, m) in enumerate(target_layers):
            hv = hv_list[idx]
            # Hessian Trace (H-Vector Product의 에너지를 채널별로 요약)
            with torch.no_grad():
                h_energy = hv.pow(2).reshape(hv.shape[0], -1).mean(1)
            
            alive_indices = torch.where(m.mask > 0.5)[0].cpu().numpy()
            
            if len(alive_indices) > 0:
                for i in alive_indices:
                    # SNOWS 점수: i번째 채널의 Hessian Trace 값
                    target_unit_scores.append(h_energy[i].item())
                    
                    # Global Ranking 함수가 기대하는 metadata 형식 (fake group)
                    fake_group = {'names': [name]}
                    target_unit_metadata.append((fake_group, i))

        print(f"[DEBUG] SNOWS: Found {len(target_unit_scores)} candidates to prune.")

        # 4. 부모의 Global Ranking 호출 및 물리적 적용
        if len(target_unit_scores) > 0:
            self._global_rank_prune(
                scores=target_unit_scores, 
                metadata=target_unit_metadata, 
                epoch=current_epoch, 
                total_epochs=total_epochs, 
                method_name="SNOWS"
            )
            self.apply_mask_to_weights()

class ATOPruner(PDTPruner):
    def step_pruning(self, loss, current_epoch, total_epochs):
        print(f"\n[DEBUG] === ATO Emergency Pruning: Epoch {current_epoch} ===")
        
        target_unit_scores = []
        target_unit_metadata = []

        # 1. 모델의 모든 레이어를 그냥 직접 훑습니다 (토폴로지 그룹 무시)
        # ResNet-18 내부의 모든 Conv/Linear를 다 찾습니다.
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # 마스크가 없으면 강제로 만들어줍니다
                if not hasattr(m, 'mask'):
                    n_f = m.weight.shape[0]
                    m.register_buffer("mask", torch.ones(n_f, device=m.weight.device))
                
                # 살아있는 채널 인덱스 확인
                alive_indices = torch.where(m.mask > 0.5)[0].cpu().numpy()
                
                # 채널별 L1-Norm(Magnitude) 점수 계산
                # (Out_channels,) 형태로 점수화
                with torch.no_grad():
                    m_score = m.weight.data.abs().reshape(m.weight.shape[0], -1).mean(1)
                
                if len(alive_indices) > 0:
                    for i in alive_indices:
                        # 메타데이터에 (레이어 객체, 채널 인덱스)를 직접 넣습니다.
                        # 그룹 개념 없이 개별 레이어로 처리
                        target_unit_scores.append(m_score[i].item())
                        
                        # metadata 형식을 _global_rank_prune이 기대하는 {'names': [이름]} 구조로 가짜 그룹화
                        fake_group = {'names': [name]}
                        target_unit_metadata.append((fake_group, i))

        print(f"[DEBUG] Found {len(target_unit_scores)} candidates to prune.")

        # 2. 부모의 랭킹 함수 호출
        # 여기서 progress = 0.5가 박혀있다면 절반이 날아가야 정상입니다.
        if len(target_unit_scores) > 0:
            self._global_rank_prune(target_unit_scores, target_unit_metadata, total_epochs, current_epoch, "ATO")
            self.apply_mask_to_weights()
        else:
            print("[DEBUG] 🚨 FATAL: Still 0 candidates. Check if the model has Conv2d/Linear layers.")
# ==============================================================================
# SuperTickets (ST - Gradient-Weight Product based) 비교 실험용 Pruner
# ==============================================================================
class STPruner(PDTPruner):
    """
    SuperTickets 논문 로직 (Emergency Mode):
    중요도 = |Weight| * |Gradient|
    가중치의 크기와 그래디언트의 크기를 곱하여 학습 기여도가 낮은 채널을 제거합니다.
    """
    def step_pruning(self, loss, current_epoch, total_epochs):
        print(f"\n[DEBUG] === ST Emergency Pruning: Epoch {current_epoch} ===")
        
        target_unit_scores = []
        target_unit_metadata = []

        # 1. 모델의 모든 레이어를 직접 훑습니다 (토폴로지 그룹 무시)
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # 마스크 자동 생성 로직 (안전장치)
                if not hasattr(m, 'mask'):
                    n_f = m.weight.shape[0]
                    m.register_buffer("mask", torch.ones(n_f, device=m.weight.device))
                
                # 살아있는 채널 인덱스 확인
                alive_indices = torch.where(m.mask > 0.5)[0].cpu().numpy()
                
                # [ST 핵심 점수 계산] interaction = |W| * sqrt(G_ema)
                # G_ema에는 이미 그래디언트의 제곱 평균 정보가 담겨 있습니다.
                with torch.no_grad():
                    w_abs = m.weight.data.abs().reshape(m.weight.shape[0], -1).mean(1)
                    # sqrt(grad_ema)를 통해 그래디언트의 크기(L2-like) 추출
                    g_magnitude = torch.sqrt(m.grad_ema + 1e-8)
                    st_score = w_abs * g_magnitude
                
                if len(alive_indices) > 0:
                    for i in alive_indices:
                        # 채널별 ST 중요도 점수 추가
                        target_unit_scores.append(st_score[i].item())
                        
                        # metadata 형식을 _global_rank_prune이 기대하는 구조로 가짜 그룹화
                        fake_group = {'names': [name]}
                        target_unit_metadata.append((fake_group, i))

        print(f"[DEBUG] ST: Found {len(target_unit_scores)} candidates to prune.")

        # 2. 부모의 Global Ranking 함수 호출 (여기서 progress 기반 컷팅이 일어남)
        if len(target_unit_scores) > 0:
            self._global_rank_prune(
                scores=target_unit_scores, 
                metadata=target_unit_metadata, 
                epoch=current_epoch, 
                total_epochs=total_epochs, 
                method_name="ST"
            )
            # 물리적 가중치 제거 적용
            self.apply_mask_to_weights()
        else:
            print("[DEBUG] 🚨 FATAL: Still 0 candidates for ST.")

# ==============================================================================
# DFPC (Data-Free Parameter Compensation - Similarity based) 비교 실험용 Pruner
# ==============================================================================
class DFPCPruner(PDTPruner):
    """
    DFPC 논문 로직 (Emergency Mode):
    데이터 없이 필터 자체의 기하학적 분포를 분석.
    필터 간 L2 거리가 가까울수록(중복될수록) 중요도가 낮다고 판단하여 제거합니다.
    """
    def step_pruning(self, loss, current_epoch, total_epochs):
        print(f"\n[DEBUG] === DFPC Emergency Pruning: Epoch {current_epoch} ===")
        
        target_unit_scores = []
        target_unit_metadata = []

        # 1. 모델의 모든 레이어를 직접 훑습니다 (토폴로지 그룹 무시)
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # 마스크 자동 생성 로직 (안전장치)
                if not hasattr(m, 'mask'):
                    n_f = m.weight.shape[0]
                    m.register_buffer("mask", torch.ones(n_f, device=m.weight.device))
                
                # 살아있는 채널 인덱스 확인
                alive_indices = torch.where(m.mask > 0.5)[0].cpu().numpy()
                
                # [DFPC 핵심 점수 계산] 필터 간 고유성 (Geometric Uniqueness)
                with torch.no_grad():
                    # weight shape: [out_channels, in_channels * k * k]
                    w = m.weight.data.reshape(m.weight.shape[0], -1)
                    
                    # 각 필터 간의 L2 Distance Matrix 계산
                    # dist_matrix[i, j]는 i번째 필터와 j번째 필터 사이의 거리
                    dist_matrix = torch.cdist(w, w, p=2)
                    
                    # 중요도 점수 = 다른 필터들과의 거리 합 (클수록 고유함)
                    # [Out_channels] 크기의 벡터 탄생
                    geometric_importance = dist_matrix.sum(dim=1)
                
                if len(alive_indices) > 0:
                    for i in alive_indices:
                        # 채널별 DFPC 점수 추가
                        target_unit_scores.append(geometric_importance[i].item())
                        
                        # metadata 형식을 _global_rank_prune이 기대하는 구조로 가짜 그룹화
                        fake_group = {'names': [name]}
                        target_unit_metadata.append((fake_group, i))

        print(f"[DEBUG] DFPC: Found {len(target_unit_scores)} candidates to prune.")

        # 2. 부모의 Global Ranking 함수 호출 (여기서 progress = epoch / total_epochs 가 적용됨)
        if len(target_unit_scores) > 0:
            self._global_rank_prune(
                scores=target_unit_scores, 
                metadata=target_unit_metadata, 
                epoch=current_epoch, 
                total_epochs=total_epochs, 
                method_name="DFPC"
            )
            # 물리적 가중치 제거 적용
            self.apply_mask_to_weights()
        else:
            print("[DEBUG] 🚨 FATAL: Still 0 candidates for DFPC.")
# ==============================================================================
# TPP (Towards Personalized Pruning - Weight-Activation Interaction) 비교 실험용 Pruner
# ==============================================================================
class TPPPruner(PDTPruner):
    """
    TPP 논문 로직 (Emergency Mode):
    중요도 = Weight Magnitude * sqrt(Gradient Persistence)
    토폴로지 그룹 무시하고 모델 전체의 모든 Conv/Linear를 직접 수집하여 프루닝합니다.
    """
    def step_pruning(self, loss, current_epoch, total_epochs):
        print(f"\n[DEBUG] === TPP Emergency Pruning: Epoch {current_epoch} ===")
        
        target_unit_scores = []
        target_unit_metadata = []

        # 1. 모델의 모든 레이어를 직접 훑습니다 (Emergency Mode 공통 로직)
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if not hasattr(m, 'mask'):
                    n_f = m.weight.shape[0]
                    m.register_buffer("mask", torch.ones(n_f, device=m.weight.device))
                
                # 살아있는 채널 인덱스 확인
                alive_indices = torch.where(m.mask > 0.5)[0].cpu().numpy()
                
                # [TPP 핵심 점수 계산] interaction = |W| * sqrt(G_ema)
                with torch.no_grad():
                    w_abs = m.weight.data.abs().reshape(m.weight.shape[0], -1).mean(1)
                    # G_ema는 이미 PDTPruner에서 업데이트되고 있음
                    g_score = torch.sqrt(m.grad_ema + 1e-8)
                    interaction_score = w_abs * g_score
                
                if len(alive_indices) > 0:
                    for i in alive_indices:
                        # 채널별 TPP 점수 추가
                        target_unit_scores.append(interaction_score[i].item())
                        
                        # metadata 형식을 _global_rank_prune이 기대하는 구조로 가짜 그룹화
                        fake_group = {'names': [name]}
                        target_unit_metadata.append((fake_group, i))

        print(f"[DEBUG] TPP: Found {len(target_unit_scores)} candidates to prune.")

        # 2. 부모의 Global Ranking 함수 호출
        if len(target_unit_scores) > 0:
            self._global_rank_prune(
                scores=target_unit_scores, 
                metadata=target_unit_metadata, 
                epoch=current_epoch, 
                total_epochs=total_epochs, 
                method_name="TPP"
            )
            # 물리적 가중치 제거 적용
            self.apply_mask_to_weights()
        else:
            print("[DEBUG] 🚨 FATAL: Still 0 candidates for TPP.")