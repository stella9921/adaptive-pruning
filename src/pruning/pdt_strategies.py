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
        
        # --- 인자 설정 우선순위 로직 (YAML > CLI if not None > Default) ---
        # 1. YAML/Config에서 먼저 가져오기
        self.group_selection_ratio = strat_cfg.get('group_selection_ratio', 1.0)
        self.final_keep_ratio = strat_cfg.get('channel_keep_ratio', 0.2)
        self.min_survival_ratio = strat_cfg.get('min_survival_ratio', 0.1)

        # 2. CLI 인자가 명시적으로 존재하면 덮어쓰기
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
        
        # 모든 레이어를 등록하되, 프루닝은 step_pruning에서 토폴로지 그룹으로만 한정합니다.
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
        
        # 2. 오직 명시된 토폴로지 그룹만 수집
        group_info_list = []
        if self.topology_groups:
            for idx, group in enumerate(self.topology_groups):
                # [안전장치] Grad EMA가 있는 레이어(Conv, Linear)만 스코어 계산용으로 분류
                score_layers = [find_layer(ln) for ln in group 
                                if isinstance(find_layer(ln), (nn.Conv2d, nn.Linear))]
                
                if not score_layers: continue
                
                # Grad EMA 평균 계산 (AttributeError 방지)
                w_g = torch.mean(torch.stack([m.grad_ema.mean() for m in score_layers])).item()
                
                group_info_list.append({
                    'id': idx+1, 
                    'layers': score_layers, # Hessian 계산용
                    'w_g': w_g, 
                    'names': group          # 마스크 적용용 전체 이름 리스트
                })

        if not group_info_list:
            print("[Warning] No valid topology groups found. Skipping step.")
            return

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
            current_group_raw_scores = []
            # 그룹의 첫 번째 레이어를 기준으로 생존 채널 탐색
            mask = g['layers'][0].mask
            alive_indices = torch.where(mask > 0.5)[0].cpu().numpy()
            total_n = mask.numel()

            # 1단계: 레이어 내부 정규화
            for m in g['layers']:
                hv = hv_list[hv_idx]
                h_energy = hv.pow(2).reshape(hv.shape[0], -1).mean(1)
                if h_energy.max() > h_energy.min():
                    h_energy = (h_energy - h_energy.min()) / (h_energy.max() - h_energy.min() + 1e-8)
                m.hessian_score.copy_(h_energy)
                hv_idx += 1

            # 마지노선 체크
            if (len(alive_indices) / total_n) <= self.min_survival_ratio:
                if g['id'] in target_group_ids: target_group_ids.remove(g['id'])
                continue

            # 2단계: 그룹 단위 정규화 및 스코어 합산 (IndexError 방지 로직 포함)
            if len(alive_indices) > 0:
                for i in alive_indices:
                    # [핵심 수정] 현재 채널 인덱스 i가 레이어의 점수 배열 크기 안에 있을 때만 합산
                    s_gc_list = [m.hessian_score[i].item() for m in g['layers'] if i < m.hessian_score.size(0)]
                    s_gc = sum(s_gc_list)/len(s_gc_list) if s_gc_list else 0.0
                    
                    raw_score = g['w_g'] * (s_gc * self.lambda_h)
                    current_group_raw_scores.append(raw_score)
                
                scores_arr = np.array(current_group_raw_scores)
                if scores_arr.max() > scores_arr.min():
                    norm_scores = (scores_arr - scores_arr.min()) / (scores_arr.max() - scores_arr.min() + 1e-8)
                else:
                    norm_scores = scores_arr

                for idx, i in enumerate(alive_indices):
                    target_unit_scores.append(norm_scores[idx]) 
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
            if len(dead_scores) > 0:
                cutoff_threshold = np.max(dead_scores)

            with torch.no_grad():
                for idx, is_alive in enumerate(optimal_mask_flags):
                    if not is_alive:
                        group_obj, channel_idx = target_unit_metadata[idx]
                        # 그룹 내 모든 레이어(BN 포함)를 찾아 마스크 업데이트
                        for ln in group_obj['names']:
                            layer_obj = find_layer(ln)
                            if layer_obj is not None and hasattr(layer_obj, 'mask'):
                                # [핵심 수정] 마스크 적용 시에도 인덱스 범위 체크 (IndexError 방지)
                                if channel_idx < layer_obj.mask.size(0):
                                    layer_obj.mask[channel_idx] = 0.0
                        pruned_count += 1

        # --- [6] 결과 출력 ---
        print(f"\n{'='*30} Conv-Only Pruning: Epoch {current_epoch} {'='*30}")
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
        torch.cuda.empty_cache()

    def get_current_sparsity(self):
        total_p = sum(m.mask.numel() for m in self.layers)
        active_p = sum(m.mask.sum().item() for m in self.layers)
        return (1.0 - (active_p / total_p)) * 100.0 if total_p > 0 else 0.0