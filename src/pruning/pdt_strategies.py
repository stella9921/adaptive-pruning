import torch
import torch.nn as nn
from .base import BasePruner
from .engine.hessian_free import SNOWSEngine
import numpy as np
from .optimizer import lagrangian_optimization 
import time
  

class PDTPruner(BasePruner):
    def __init__(self, model, config, args=None, topology_groups=None):
        super().__init__(model, config)
        
        strat_cfg = config.get('strategy', {})
        
        # --- 인자 설정 (None일 경우를 대비한 하드코딩 기본값 추가) ---
        # 1. args에서 먼저 찾고, 없으면 config, 그것도 없으면 기본값(0.8, 0.2, 0.1) 사용
        self.group_selection_ratio = getattr(args, 'group_selection_ratio', None) or strat_cfg.get('group_selection_ratio', 0.9)
        self.final_keep_ratio = getattr(args, 'channel_keep_ratio', None) or strat_cfg.get('channel_keep_ratio', 0.3)
        self.min_survival_ratio = getattr(args, 'min_survival_ratio', None) or strat_cfg.get('min_survival_ratio', 0.1)

        # 나머지 설정
        self.ema_decay = strat_cfg.get('ema_decay', 0.95)
        self.lambda_h = strat_cfg.get('lambda_h', 0.1)
        self.k_horizon = strat_cfg.get('k_horizon', 25)
        self.engine = SNOWSEngine(n_iter=strat_cfg.get('hessian_iter', 10))
        self.topology_groups = topology_groups

        # layers_dict 선언 및 레이어 수집
        self.layers_dict = nn.ModuleDict() 
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                self.layers_dict[name.replace('.', '_')] = m
        
        self.layers = list(self.layers_dict.values())
        self._check_buffers()
        
        print(f"\n[Pruner Init] Strict Conv-Topology Mode Activated")
        # 이제 self.min_survival_ratio가 None이 아니므로 에러가 나지 않습니다.
        print(f"[*] Applied Min Survival Guarantee: {self.min_survival_ratio*100:.1f}%")
        print(f"[*] Total {len(self.topology_groups)} groups linked to {len(self.layers)} prunable layers.")

    def _check_buffers(self):
        for m in self.layers:
            n_f = m.weight.shape[0]
            if not hasattr(m, 'mask'):
                m.register_buffer("mask", torch.ones(n_f, device=self.device))
            if not hasattr(m, 'grad_ema'):
                m.register_buffer("grad_ema", torch.zeros(n_f, device=self.device))
            if not hasattr(m, 'hessian_score'):
                m.register_buffer("hessian_score", torch.zeros(n_f, device=self.device))

    # def apply_mask_to_weights(self, optimizer=None):
    #     with torch.no_grad():
    #         for m in self.layers:
    #             mask = m.mask
    #             m_view = mask.view(-1, 1, 1, 1) if m.weight.dim() == 4 else mask.view(-1, 1)
    #             m.weight.data.mul_(m_view)
    #             if hasattr(m, 'bias') and m.bias is not None:
    #                 m.bias.data.mul_(mask)
    # def apply_mask_to_weights(self, optimizer=None):
    #     with torch.no_grad():
    #         for m in self.layers:
    #             if not hasattr(m, 'mask'): continue
                
    #             mask = m.mask
    #             # 레이어 타입에 따른 마스크 모양(Shape) 맞추기
    #             if m.weight.dim() == 4: # Conv2d
    #                 m_view = mask.view(-1, 1, 1, 1)
    #             elif m.weight.dim() == 2: # Linear
    #                 m_view = mask.view(-1, 1)
    #             else: # BatchNorm (1D weight)
    #                 m_view = mask
                
    #             # 가중치에 마스크 적용
    #             m.weight.data.mul_(m_view)
                
    #             # 편향(Bias)이 있다면 적용
    #             if hasattr(m, 'bias') and m.bias is not None:
    #                 m.bias.data.mul_(mask)

    def apply_mask_to_weights(self, optimizer=None):
        all_modules = dict(self.model.named_modules())
        
        def find_layer(name):
            if name in all_modules: return all_modules[name]
            dot_name = name.replace('_', '.')
            return all_modules.get(dot_name)

        with torch.no_grad():
            # 1. 🚨 [핵심] 토폴로지 그룹 기반 연쇄 절삭 (Coupling)
            # BN 마스크 하나로 그 그룹의 모든 Conv/BN을 다 죽여야 Sparsity가 올라갑니다.
            if self.topology_groups:
                for group in self.topology_groups:
                    # 그룹 내에서 마스크를 가진 대표 레이어(주로 BN) 찾기
                    representative_mask = None
                    for ln in group:
                        layer = find_layer(ln)
                        if layer and hasattr(layer, 'mask'):
                            representative_mask = layer.mask
                            break
                    
                    if representative_mask is not None:
                        for ln in group:
                            m = find_layer(ln)
                            if not m or not hasattr(m, 'weight'): continue
                            
                            # 분류기(Linear) 레이어는 절대 건드리지 않음 (방어 로직)
                            if isinstance(m, nn.Linear): continue
                            
                            # 레이어 차원에 맞춰 마스크 적용
                            if m.weight.dim() == 4: # Conv2d [Out, In, K, K]
                                m_view = representative_mask.view(-1, 1, 1, 1)
                            else: # BatchNorm [Channels]
                                m_view = representative_mask
                            
                            m.weight.data.mul_(m_view)
                            if hasattr(m, 'bias') and m.bias is not None:
                                m.bias.data.mul_(representative_mask)

            # 2. 그룹 외 개별 레이어들 마스크 적용 (혹시 누락된 레이어 방어)
            for m in self.layers:
                if not hasattr(m, 'mask') or isinstance(m, nn.Linear):
                    continue
                
                # 이미 위에서 그룹으로 처리되었겠지만, 이중 확인
                mask = m.mask
                if m.weight.dim() == 4:
                    m_view = mask.view(-1, 1, 1, 1)
                else:
                    m_view = mask
                
                # 안전한 곱셈을 위해 shape 체크 추가
                if m.weight.data.shape[0] == m_view.shape[0]:
                    m.weight.data.mul_(m_view)
    # def update_ema_and_mask_grad(self):
    #     with torch.no_grad():
    #         for m in self.layers:
    #             if hasattr(m, 'weight') and m.weight.grad is not None:
    #                 m_view = m.mask.view(-1, 1, 1, 1) if m.weight.dim()==4 else m.mask.view(-1, 1)
    #                 m.weight.grad.mul_(m_view)
    #                 g = m.weight.grad.pow(2).reshape(m.weight.shape[0], -1).mean(1)
    #                 m.grad_ema.mul_(self.ema_decay).add_(g, alpha=1 - self.ema_decay)
    def update_ema_and_mask_grad(self):
        with torch.no_grad():
            for m in self.layers:
                # 1. 가중치나 그라디언트가 없는 경우 패스
                if not hasattr(m, 'weight') or m.weight.grad is None:
                    continue
                
                # 2. 마스크 레이어 타입에 따른 m_view 생성 (RuntimeError 방지)
                if m.weight.dim() == 4: # Conv2d
                    m_view = m.mask.view(-1, 1, 1, 1)
                elif m.weight.dim() == 2: # Linear
                    m_view = m.mask.view(-1, 1)
                else: # BatchNorm (1D)
                    m_view = m.mask
                
                # 3. 그라디언트에 마스크 적용 (Pruned된 채널의 그라디언트 차단)
                m.weight.grad.mul_(m_view)
                
                # 4. Grad EMA 업데이트
                # (채널별 그라디언트 에너지 계산)
                g = m.weight.grad.pow(2).reshape(m.weight.shape[0], -1).mean(1)
                m.grad_ema.mul_(self.ema_decay).add_(g, alpha=1 - self.ema_decay)
    # def step_pruning(self, loss, current_epoch, total_epochs):
    #     all_modules = dict(self.model.named_modules())
    #     def find_layer(name):
    #         if name in all_modules: return all_modules[name]
    #         return all_modules.get(name.replace('_', '.'))

    #     # 1. 타겟 생존율 계산 (기본 progress는 유지하되 내부에서 보정)
    #     progress = current_epoch / total_epochs
        
    #     # 2. 토폴로지 그룹 수집
    #     group_info_list = []
    #     if self.topology_groups:
    #         for idx, group in enumerate(self.topology_groups):
    #             score_layers = [find_layer(ln) for ln in group 
    #                             if isinstance(find_layer(ln), (nn.Conv2d, nn.Linear))]
    #             if not score_layers: continue
    #             # grad_ema가 0일 경우를 대비해 epsilon(1e-9) 추가
    #             w_g = torch.mean(torch.stack([m.grad_ema.mean() for m in score_layers])).item() + 1e-9
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
    #     print(m.mask)
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
    #         print(m.mask)
            

    #         # --- 현장 검거용 출력 ---
    #         mask_sum = g['layers'][0].mask.sum().item()
    #         mask_size = g['layers'][0].mask.numel()
    #         print(f"[CHECK] Group ID: {g['id']} | Mask Alive: {mask_sum}/{mask_size}")



    #         if len(alive_indices) > 0:
    #             for i in alive_indices:
    #                 s_gc_list = [m.hessian_score[i].item() for m in g['layers'] if i < m.hessian_score.size(0)]
    #                 s_gc = sum(s_gc_list)/len(s_gc_list) if s_gc_list else 0.0
                    
    #                 # PDT 스코어 계산 (0 방지를 위해 epsilon 추가)
    #                 raw_score = (g['w_g'] * (s_gc * self.lambda_h)) + 1e-9
                    
    #                 target_unit_scores.append(raw_score) 
    #                 target_unit_costs.append(sum(m.weight.nelement()/m.weight.shape[0] for m in g['layers']))
    #                 target_unit_metadata.append((g, i))

    #     # --- [5] 프루닝 최적화 및 마스크 적용 (CCTV 보강 버전) ---
    #     current_sparsity = self.get_current_sparsity() / 100.0
    #     current_alive_ratio = 1.0 - current_sparsity
    #     pruned_count = 0

    #     if len(target_unit_scores) > 0:
    #         start_ep = 120
    #         # 강제 진행률 보정
    #         eff_progress = (current_epoch - start_ep + 1) / (total_epochs - start_ep + 1e-8)
    #         eff_progress = max(0.15, min(1.0, eff_progress)) # 🚨 첫날부터 15% 진행된 걸로 강제 인식
            
    #         calculated_keep_ratio = 1.0 - (eff_progress * (1.0 - self.final_keep_ratio))
    #         # 🚨 현재 생존율보다 무조건 10% 더 낮게 목표 설정 (즉, 10% 강제 절삭)
    #         total_target_keep_ratio = min(calculated_keep_ratio, current_alive_ratio - 0.10)

    #         incremental_keep_ratio = total_target_keep_ratio / (current_alive_ratio + 1e-8)
    #         total_budget = np.sum(target_unit_costs) * incremental_keep_ratio
            
    #         # 최적화 엔진 가동
    #         optimal_mask_flags = lagrangian_optimization(np.array(target_unit_scores), np.array(target_unit_costs), total_budget)
            
    #         # [CCTV] 최적화 결과 확인
    #         to_be_pruned = np.sum(optimal_mask_flags == 0)
    #         print(f"[DEBUG] Lagrangian decided to prune {to_be_pruned} channels out of {len(optimal_mask_flags)}")

    #         with torch.no_grad():
    #             for idx, is_alive in enumerate(optimal_mask_flags):
    #                 if not is_alive:
    #                     group_obj, channel_idx = target_unit_metadata[idx]
    #                     # group_obj는 이름 리스트 ['layer1.0.conv1', ...]
    #                     for ln in group_obj:
    #                         layer_obj = find_layer(ln)
    #                         if layer_obj is not None and hasattr(layer_obj, 'mask'):
    #                             if channel_idx < layer_obj.mask.size(0):
    #                                 layer_obj.mask[channel_idx] = 0.0
    #                                 pruned_count += 1
            
    #         print(f"[DEBUG] Actual masks changed to zero: {pruned_count}")
    #     else:
    #         print("[DEBUG] 🚨 ERROR: target_unit_scores is empty! Check find_layer function.")

    #     # --- [6] 결과 출력 ---
    #     print(f"\n{'='*30} PDT Pruning Status: Epoch {current_epoch} {'='*30}")
    #     print(f" [*] Target Ratio: {total_target_keep_ratio:.4f} | Total Pruned Channels: {pruned_count}")
    #     # ... (이후 출력 코드는 동일)
    #     # --- [6] 결과 출력 ---
    #     print(f"\n{'='*30} PDT Pruning: Epoch {current_epoch} {'='*30}")
    #     print(f" [*] Target Keep Ratio: {total_target_keep_ratio:.4f} | Pruned: {pruned_count}")
    #     print(f" {'Group ID':<10} | {'Alive/Total':>12} | {'Sparsity':>8} | {'Hessian(avg)':>12} | {'Status'}")
    #     print(f" {'-'*87}")
        
    #     group_info_list.sort(key=lambda x: x['id'])
    #     for g in group_info_list:
    #         m = g['layers'][0]
    #         total, alive = m.mask.numel(), int(m.mask.sum().item())
    #         sparsity = (1 - alive/total) * 100
    #         h_avg = g['layers'][0].hessian_score.mean().item() if g['id'] in target_group_ids else 0.0
            
    #         status = "TARGET" if g['id'] in target_group_ids else "FIXED"
    #         if (alive/total) <= self.min_survival_ratio: status = "MIN-SURV"
    #         print(f" Group {g['id']:2d}      | {alive:4d}/{total:4d}      | {sparsity:>7.1f}% | {h_avg:>12.6f} | [{status}]")
    #     print(f"{'='*89}\n")

    #     # 학회용 리소스 분석 출력
    #     eff = self.get_model_efficiency()
    #     print(f"\n[Scientific Metrics - Epoch {current_epoch}]")
    #     print(f" 🟢 Model Size: {eff['orig_mb']:.2f} MB -> {eff['curr_mb']:.2f} MB (Reduction: {eff['orig_mb'] - eff['curr_mb']:.2f} MB)")
    #     print(f" 🔵 Sparsity (Params/FLOPs): {eff['sparsity']:.2f} %")
    #     print(f" 🟡 Theoretical Speedup: {eff['speedup']:.2f}x")
    #     print(f" 🟠 Current GPU Mem (Allocated): {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    #     print(f"{'='*89}\n")
    #     torch.cuda.empty_cache()


    # def step_pruning(self, loss, current_epoch, total_epochs):
    #     print(f"\n[DEBUG] === step_pruning Started at Epoch {current_epoch} ===")
    #     all_modules = dict(self.model.named_modules())
        
    #     # [CCTV] 이름 차이 분석기
    #     if self.topology_groups:
    #         first_group = self.topology_groups[0]
    #         search_target = first_group[0]
    #         print(f"[DEBUG] I am looking for: '{search_target}'")
            
    #         # 모델에서 가장 비슷하게 생긴 이름들 3개 추출
    #         import difflib
    #         closest = difflib.get_close_matches(search_target, all_modules.keys(), n=3, cutoff=0.1)
    #         print(f"[DEBUG] Closest names in actual model: {closest}")

    #     def find_layer(name):
    #         # 1. 원본 그대로
    #         if name in all_modules: return all_modules[name]
    #         # 2. 언더바 <-> 점 교체
    #         name_dot = name.replace('_', '.')
    #         if name_dot in all_modules: return all_modules[name_dot]
    #         name_under = name.replace('.', '_')
    #         if name_under in all_modules: return all_modules[name_under]
    #         # 3. DataParallel 접두사 (module.) 대응
    #         if name.startswith('module.'): return find_layer(name.replace('module.', ''))
    #         if f"module.{name}" in all_modules: return all_modules[f"module.{name}"]
    #         return None

    #     # --- 그룹 수집 로직 (느슨하게) ---
    #     group_info_list = []
    #     for idx, group in enumerate(self.topology_groups):
    #         score_layers = []
    #         for ln in group:
    #             layer = find_layer(ln)
    #             # weight가 있는 모듈이면 일단 OK (Conv, Linear, BN 모두 포함)
    #             if layer and hasattr(layer, 'weight'):
    #                 score_layers.append(layer)
            
    #         if not score_layers:
    #             # 50개 그룹 다 나오면 너무 길어서 1개만 자세히 출력
    #             if idx == 0:
    #                 print(f"[DEBUG] Warning: Group {idx+1} failed. Target names: {group}")
    #             continue

    #         # grad_ema 안전하게 가져오기
    #         grad_vals = []
    #         for m in score_layers:
    #             val = m.grad_ema.mean() if hasattr(m, 'grad_ema') else torch.tensor(1e-9).to(self.device)
    #             grad_vals.append(val)
            
    #         w_g = torch.mean(torch.stack(grad_vals)).item() + 1e-9
    #         group_info_list.append({'id': idx+1, 'layers': score_layers, 'w_g': w_g, 'names': group})

    #     if not group_info_list:
    #         print("[DEBUG] 🚨 ERROR: No valid group_info_list created. Naming mismatch is critical!")
    #         return

    #     # 3. 중요도 정렬 및 타겟 선정
    #     sorted_groups = sorted(group_info_list, key=lambda x: x['w_g'])
    #     num_targets = int(len(sorted_groups) * self.group_selection_ratio)
    #     num_targets = max(1, num_targets) # 최소 1개 그룹은 보장
    #     target_groups = sorted_groups[:num_targets]
    #     target_group_ids = [g['id'] for g in target_groups]
    #     print(f"[DEBUG] Target Groups for Hessian: {target_group_ids}")

    #     # 4. Hessian 계산
    #     target_params = [m.weight for g in target_groups for m in g['layers']]
    #     print(f"[DEBUG] Calculating Hessian for {len(target_params)} parameters...")
    #     hv_list = self.engine.get_k_step_hessian_selective(loss, target_params, self.k_horizon)

    #     target_unit_scores = []
    #     target_unit_costs = []
    #     target_unit_metadata = []

    #     hv_idx = 0


    #     for g in target_groups:
    #         # [수정] mask가 있는 첫 번째 레이어를 찾음
    #         base_mask = None
    #         for m in g['layers']:
    #             if hasattr(m, 'mask'):
    #                 base_mask = m.mask
    #                 break
            
    #         if base_mask is None:
    #             print(f"[DEBUG] Group {g['id']} has no maskable layers. Skipping...")
    #             continue

    #         # [이후 로직] base_mask를 사용하여 alive_indices 계산
    #         if base_mask.sum() == 0:
    #             base_mask.fill_(1.0)
            
    #         alive_indices = torch.where(base_mask > 0.5)[0].cpu().numpy()
        
            
    #         # 그룹 내 레이어별 Hessian 점수 계산 및 저장
    #         group_h_energies = []
    #         for m in g['layers']:
    #             hv = hv_list[hv_idx]; hv_idx += 1
    #             h_energy = hv.pow(2).reshape(hv.shape[0], -1).mean(1)
    #             # 정규화
    #             if h_energy.max() > h_energy.min():
    #                 h_energy = (h_energy - h_energy.min()) / (h_energy.max() - h_energy.min() + 1e-8)
    #             m.hessian_score.copy_(h_energy)
    #             group_h_energies.append(h_energy)

    #         # 스코어 수집
    #         if len(alive_indices) > 0:
    #             for i in alive_indices:
    #                 # 필터 i에 대한 그룹 평균 Hessian
    #                 s_gc = sum([he[i].item() for he in group_h_energies]) / len(group_h_energies)
    #                 raw_score = (g['w_g'] * (s_gc * self.lambda_h)) + 1e-9
                    
    #                 target_unit_scores.append(raw_score) 
    #                 target_unit_costs.append(sum(m.weight.nelement()/m.weight.shape[0] for m in g['layers']))
    #                 target_unit_metadata.append((g, i))

    #     print(f"[DEBUG] Total Unit Candidates: {len(target_unit_scores)}")

    #     # 5. 프루닝 최적화 (라그랑주)
    #     pruned_count = 0
    #     if target_unit_scores:
    #         start_ep = 120
    #         eff_progress = (current_epoch - start_ep + 1) / (total_epochs - start_ep + 1e-8)
    #         eff_progress = max(0.15, min(1.0, eff_progress)) 
            
    #         calculated_keep_ratio = 1.0 - (eff_progress * (1.0 - self.final_keep_ratio))
    #         # [핵심] 현재보다 무조건 10% 더 자르도록 강제 제약
    #         total_target_keep_ratio = min(calculated_keep_ratio, current_alive_ratio - 0.10)
            
    #         incremental_keep_ratio = total_target_keep_ratio / (current_alive_ratio + 1e-8)
    #         total_budget = np.sum(target_unit_costs) * incremental_keep_ratio
            
    #         print(f"[DEBUG] Target Keep Ratio: {total_target_keep_ratio:.4f} (Inc: {incremental_keep_ratio:.4f})")
            
    #         optimal_mask_flags = lagrangian_optimization(np.array(target_unit_scores), np.array(target_unit_costs), total_budget)
            
    #         # 마스크 실제 적용
    #         with torch.no_grad():
    #             for idx, is_alive in enumerate(optimal_mask_flags):
    #                 if not is_alive:
    #                     group_obj, channel_idx = target_unit_metadata[idx]
    #                     for layer_obj in group_obj['layers']:
    #                         if channel_idx < layer_obj.mask.size(0):
    #                             layer_obj.mask[channel_idx] = 0.0
    #                             pruned_count += 1
            
    #         print(f"[DEBUG] Lagrangian Decision: Pruned {np.sum(optimal_mask_flags == 0)} units.")
    #         print(f"[DEBUG] Actual Mask Updates: {pruned_count} channels.")

    #     # 6. 결과 출력
    #     print(f"\n{'='*30} PDT Pruning Results: Epoch {current_epoch} {'='*30}")
    #     print(f" [*] Pruned this step: {pruned_count}")
        
    #     group_info_list.sort(key=lambda x: x['id'])
    #     for g in group_info_list:
    #         m = g['layers'][0]
    #         total, alive = m.mask.numel(), int(m.mask.sum().item())
    #         sparsity = (1 - alive/total) * 100
    #         h_avg = m.hessian_score.mean().item() if g['id'] in target_group_ids else 0.0
    #         status = "TARGET" if g['id'] in target_group_ids else "FIXED"
    #         print(f" Group {g['id']:2d} | {alive:4d}/{total:4d} | {sparsity:>7.1f}% | {h_avg:>12.6f} | [{status}]")

    #     eff = self.get_model_efficiency()
    #     print(f"\n[Scientific Metrics]")
    #     print(f" 🟢 Model Size: {eff['curr_mb']:.2f} MB / 🔵 Sparsity: {eff['sparsity']:.2f} %")
    #     print(f" 🟠 GPU Mem: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    #     print(f"{'='*89}\n")
    #     torch.cuda.empty_cache()
    def step_pruning(self, loss, current_epoch, total_epochs):
        print(f"\n[DEBUG] === step_pruning Started at Epoch {current_epoch} ===")
        all_modules = dict(self.model.named_modules())
        
        # 1. 현재 모델 상태 파악 (current_alive_ratio 정의 시점 최상단 이동)
        current_sparsity = self.get_current_sparsity() / 100.0
        current_alive_ratio = 1.0 - current_sparsity
        print(f"[DEBUG] Current Model Sparsity: {current_sparsity*100:.2f}% (Alive: {current_alive_ratio*100:.2f}%)")

        def find_layer(name):
            # 1. 원본 그대로 / 2. 언더바 <-> 점 교체 / 3. DataParallel 대응
            if name in all_modules: return all_modules[name]
            dot_name = name.replace('_', '.')
            if dot_name in all_modules: return all_modules[dot_name]
            name_under = name.replace('.', '_')
            if name_under in all_modules: return all_modules[name_under]
            if name.startswith('module.'): return find_layer(name.replace('module.', ''))
            if f"module.{name}" in all_modules: return all_modules[f"module.{name}"]
            return None

        # 2. 토폴로지 그룹 수집 로직
        group_info_list = []
        if self.topology_groups:
            # 이름 차이 분석기 (CCTV)
            first_group_name = self.topology_groups[0][0]
            print(f"[DEBUG] I am looking for: '{first_group_name}'")
            import difflib
            closest = difflib.get_close_matches(first_group_name, all_modules.keys(), n=3, cutoff=0.1)
            print(f"[DEBUG] Closest names in actual model: {closest}")

            for idx, group in enumerate(self.topology_groups):
                score_layers = []
                for ln in group:
                    layer = find_layer(ln)
                    # weight가 있는 모듈이면 일단 OK (Conv, Linear, BN 모두 포함)
                    if layer and hasattr(layer, 'weight'):
                        score_layers.append(layer)
                
                if not score_layers:
                    if idx == 0: print(f"[DEBUG] Warning: Group {idx+1} failed. Target names: {group}")
                    continue

                # grad_ema 안전하게 가져오기
                grad_vals = []
                for m in score_layers:
                    val = m.grad_ema.mean() if hasattr(m, 'grad_ema') else torch.tensor(1e-9).to(self.device)
                    grad_vals.append(val)
                
                w_g = torch.mean(torch.stack(grad_vals)).item() + 1e-9
                group_info_list.append({'id': idx+1, 'layers': score_layers, 'w_g': w_g, 'names': group})

        if not group_info_list:
            print("[DEBUG] 🚨 ERROR: No valid group_info_list created. Naming mismatch is critical!")
            return

        # 3. 중요도 정렬 및 타겟 선정
        sorted_groups = sorted(group_info_list, key=lambda x: x['w_g'])
        num_targets = max(1, int(len(sorted_groups) * self.group_selection_ratio))
        target_groups = sorted_groups[:num_targets]
        target_group_ids = [g['id'] for g in target_groups]
        print(f"[DEBUG] Target Groups for Hessian: {target_group_ids}")

        # 4. Hessian 계산
        target_params = [m.weight for g in target_groups for m in g['layers'] if hasattr(m, 'weight')]
        print(f"[DEBUG] Calculating Hessian for {len(target_params)} parameters...")
        hv_list = self.engine.get_k_step_hessian_selective(loss, target_params, self.k_horizon)

        target_unit_scores = []
        target_unit_costs = []
        target_unit_metadata = []

        hv_idx = 0
        for g in target_groups:
            # [수정] mask가 있는 첫 번째 레이어를 찾음 (BatchNorm 등 방어)
            base_mask = None
            for m in g['layers']:
                if hasattr(m, 'mask'):
                    base_mask = m.mask
                    break
            
            if base_mask is None:
                print(f"[DEBUG] Group {g['id']} has no maskable layers. Skipping...")
                continue

            # 마스크가 0인 경우 로드 오류 방지를 위해 1.0으로 채움
            if base_mask.sum() == 0:
                print(f"[DEBUG] 🚨 ALERT: Group {g['id']} mask is ALL ZERO. Resetting to 1.0.")
                base_mask.fill_(1.0)
            
            alive_indices = torch.where(base_mask > 0.5)[0].cpu().numpy()
            
            # 그룹 내 레이어별 Hessian 점수 계산 및 저장
            group_h_energies = []
            for m in g['layers']:
                # weight가 있는 레이어에 대해서만 Hessian 인덱스 매칭
                if hasattr(m, 'weight'):
                    hv = hv_list[hv_idx]; hv_idx += 1
                    h_energy = hv.pow(2).reshape(hv.shape[0], -1).mean(1)
                    # 정규화
                    if h_energy.max() > h_energy.min():
                        h_energy = (h_energy - h_energy.min()) / (h_energy.max() - h_energy.min() + 1e-8)
                    m.hessian_score.copy_(h_energy)
                    group_h_energies.append(h_energy)

            # 스코어 수집
            if len(alive_indices) > 0:
                for i in alive_indices:
                    # 필터 i에 대한 그룹 평균 Hessian
                    s_gc_list = [he[i].item() for he in group_h_energies if i < he.size(0)]
                    s_gc = sum(s_gc_list) / len(s_gc_list) if s_gc_list else 0.0
                    raw_score = (g['w_g'] * (s_gc * self.lambda_h)) + 1e-9
                    
                    target_unit_scores.append(raw_score) 
                    target_unit_costs.append(sum(m.weight.nelement()/m.weight.shape[0] for m in g['layers'] if hasattr(m, 'weight')))
                    target_unit_metadata.append((g, i))

        print(f"[DEBUG] Total Unit Candidates: {len(target_unit_scores)}")

        # 5. 프루닝 최적화 (라그랑주)
        pruned_count = 0
        if target_unit_scores:
            start_ep = 120
            eff_progress = (current_epoch - start_ep + 1) / (total_epochs - start_ep + 1e-8)
            eff_progress = max(0.15, min(1.0, eff_progress)) 
            
            calculated_keep_ratio = 1.0 - (eff_progress * (1.0 - self.final_keep_ratio))
            # [핵심] 현재보다 무조건 10% 더 자르도록 강제 제약 (current_alive_ratio 활용)
            total_target_keep_ratio = min(calculated_keep_ratio, current_alive_ratio - 0.05)
            
            incremental_keep_ratio = total_target_keep_ratio / (current_alive_ratio + 1e-8)
            total_budget = np.sum(target_unit_costs) * incremental_keep_ratio
            
            print(f"[DEBUG] Target Keep Ratio: {total_target_keep_ratio:.4f} (Inc: {incremental_keep_ratio:.4f})")
            
            optimal_mask_flags = lagrangian_optimization(np.array(target_unit_scores), np.array(target_unit_costs), total_budget)
            
            # 마스크 실제 적용
            with torch.no_grad():
                for idx, is_alive in enumerate(optimal_mask_flags):
                    if not is_alive:
                        group_obj, channel_idx = target_unit_metadata[idx]
                        for layer_obj in group_obj['layers']:
                            if hasattr(layer_obj, 'mask') and channel_idx < layer_obj.mask.size(0):
                                layer_obj.mask[channel_idx] = 0.0
                                pruned_count += 1
            
            print(f"[DEBUG] Lagrangian Decision: Pruned {np.sum(optimal_mask_flags == 0)} units.")
            print(f"[DEBUG] Actual Mask Updates: {pruned_count} channels.")



        # 6. 결과 출력 및 RQ 대응 메트릭 수집
        print(f"\n{'='*30} PDT Pruning Results: Epoch {current_epoch} {'='*30}")
        print(f" [*] Pruned this step: {pruned_count}")
        
        group_info_list.sort(key=lambda x: x['id'])
        for g in group_info_list:
            # 그룹 내 maskable 레이어 찾기
            m = next((l for l in g['layers'] if hasattr(l, 'mask')), g['layers'][0])
            total = m.mask.numel()
            alive = int(m.mask.sum().item())
            sparsity = (1 - alive/total) * 100
            h_avg = m.hessian_score.mean().item() if g['id'] in target_group_ids else 0.0
            status = "TARGET" if g['id'] in target_group_ids else "FIXED"
            print(f" Group {g['id']:2d} | {alive:4d}/{total:4d} | {sparsity:>7.1f}% | {h_avg:>12.6f} | [{status}]")

        # 🚨 KeyError 방지를 위해 키 이름을 'Params_MB'에서 'Size_MB'로 통일하거나 아래처럼 직접 매칭하세요.
        eff = self.measure_performance_for_rqs() 

        print(f"\n[Scientific Metrics - RQ Analysis]")
        print(f" 🔵 Sparsity (Params): {eff.get('Sparsity', 0):.2f} %")
        # 💡 만약 함수 return에 'Params_MB'라고 되어있다면 eff['Params_MB']로 써야 합니다.
        # 안전하게 get()을 쓰거나 아래 키 이름을 확인하세요.
        print(f" 🟢 Model Size: {eff.get('Size_MB', eff.get('Params_MB', 0)):.2f} MB")
        print(f" 🟠 Peak VRAM (RQ2): {eff.get('Peak_VRAM_MB', 0):.2f} MB")
        print(f" ⚡ Inference Latency (RQ3): {eff.get('Latency_ms', 0):.4f} ms")
        print(f" 🚀 Theoretical Speedup: {eff.get('Theoretical_Speedup', eff.get('Reduction_Ratio', 1)):.2f}x")
        print(f"{'='*89}\n")
        torch.cuda.empty_cache()
        # # 6. 결과 출력
        # print(f"\n{'='*30} PDT Pruning Results: Epoch {current_epoch} {'='*30}")
        # print(f" [*] Pruned this step: {pruned_count}")
        
        # group_info_list.sort(key=lambda x: x['id'])
        # for g in group_info_list:
        #     m = g['layers'][0]
        #     total, alive = m.mask.numel(), int(m.mask.sum().item())
        #     sparsity = (1 - alive/total) * 100
        #     h_avg = m.hessian_score.mean().item() if g['id'] in target_group_ids else 0.0
        #     status = "TARGET" if g['id'] in target_group_ids else "FIXED"
        #     print(f" Group {g['id']:2d} | {alive:4d}/{total:4d} | {sparsity:>7.1f}% | {h_avg:>12.6f} | [{status}]")

        # eff = self.get_model_efficiency()
        # print(f"\n[Scientific Metrics]")
        # print(f" 🟢 Model Size: {eff['curr_mb']:.2f} MB / 🔵 Sparsity: {eff['sparsity']:.2f} %")
        # print(f" 🟠 GPU Mem: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        # print(f"{'='*89}\n")
        # torch.cuda.empty_cache()



    # def get_current_sparsity(self):
    #     total_p = sum(m.mask.numel() for m in self.layers)
    #     active_p = sum(m.mask.sum().item() for m in self.layers)
    #     return (1.0 - (active_p / total_p)) * 100.0 if total_p > 0 else 0.0

    def get_current_sparsity(self):
        """ResNet-152의 복잡한 레이어 이름을 추적하여 실제 파라미터 Sparsity 계산"""
        total_params = 0
        zero_params = 0
        all_modules = dict(self.model.named_modules())
        
        with torch.no_grad():
            for name, m in self.model.named_modules():
                # 파라미터가 있는 주요 레이어 (Conv, Linear, BN)
                if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                    if not hasattr(m, 'weight') or m.weight is None:
                        continue
                    
                    n_p = m.weight.numel()
                    if hasattr(m, 'bias') and m.bias is not None:
                        n_p += m.bias.numel()
                    
                    total_params += n_p
                    
                    # 1. 레이어 자체가 마스크를 가진 경우
                    if hasattr(m, 'mask'):
                        keep_ratio = m.mask.sum().item() / (m.mask.numel() + 1e-9)
                        zero_params += n_p * (1.0 - keep_ratio)
                    
                    # 2. 마스크는 없지만 토폴로지 그룹에 의해 잘리는 연쇄 레이어(Conv 등) 처리
                    else:
                        found_coupling = False
                        if self.topology_groups:
                            for group in self.topology_groups:
                                # '.' 형식을 '_' 형식으로 변환하여 그룹 내 존재 확인
                                normalized_name = name.replace('.', '_')
                                if name in group or normalized_name in group:
                                    # 그룹 내에서 마스크를 가진 대표 레이어(BN)를 찾아 상태 공유
                                    for member_name in group:
                                        # 그룹 내 멤버 이름도 원본 모델 이름으로 복구 시도
                                        original_member_name = member_name.replace('_', '.')
                                        target_m = all_modules.get(member_name) or all_modules.get(original_member_name)
                                        
                                        if target_m and hasattr(target_m, 'mask'):
                                            keep_ratio = target_m.mask.sum().item() / (target_m.mask.numel() + 1e-9)
                                            zero_params += n_p * (1.0 - keep_ratio)
                                            found_coupling = True
                                            break
                                if found_coupling: break
        
        return (zero_params / (total_params + 1e-9)) * 100.0

    # def get_model_efficiency(self, example_inputs=None):
    #     """FLOPs, Latency(Proxy), Memory, Size를 이론적으로 계산"""
    #     total_params = 0
    #     remaining_params = 0
    #     total_flops = 0
    #     remaining_flops = 0
        
    #     # 1. Parameter & FLOPs 계산 (VGG 기준)
    #     for name, m in self.model.named_modules():
    #         if isinstance(m, (nn.Conv2d, nn.Linear)):
    #             # 원본 수치
    #             n_p = m.weight.numel()
    #             total_params += n_p
                
    #             # FLOPs 근사 (H * W * C_in * C_out * K * K)
    #             # 여기서는 파라미터 감소 비율을 FLOPs 감소 비율의 프록시로 사용
    #             if hasattr(m, 'mask'):
    #                 keep_ratio = m.mask.sum().item() / m.mask.numel()
    #                 remaining_params += n_p * keep_ratio
    #                 # Conv의 경우 입력/출력이 같이 줄어들면 제곱으로 줄어들지만, 
    #                 # 마스킹 단계에서는 출력 채널 감소 비율을 기준으로 선형 근사하여 보수적 측정
    #                 remaining_flops += n_p * keep_ratio 
    #             else:
    #                 remaining_params += n_p
    #                 remaining_flops += n_p

    #     # 2. 지표 산출
    #     orig_size_mb = (total_params * 4) / (1024**2)
    #     curr_size_mb = (remaining_params * 4) / (1024**2)
    #     sparsity = (1 - remaining_params/total_params) * 100
        
    #     # 3. Latency & Energy (학술적 프록시 모델링)
    #     # 실제 Latency는 하드웨어 종속적이므로, FLOPs 감소량에 기반한 이론적 가속도를 출력
    #     theoretical_speedup = total_params / (remaining_params + 1e-8)
        
    #     return {
    #         'orig_mb': orig_size_mb,
    #         'curr_mb': curr_size_mb,
    #         'sparsity': sparsity,
    #         'speedup': theoretical_speedup
    #     }
    # def get_model_efficiency(self):
    #     total_params = 0
    #     remaining_params = 0
        
    #     for name, m in self.model.named_modules():
    #         # Conv, Linear, BN 모두 체크
    #         if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
    #             if not hasattr(m, 'weight') or m.weight is None: continue
                
    #             n_p = m.weight.numel()
    #             if hasattr(m, 'bias') and m.bias is not None:
    #                 n_p += m.bias.numel()
                
    #             total_params += n_p
                
    #             # 마스크가 있으면 잘린 비율만큼 차감, 없으면 전량 유지
    #             if hasattr(m, 'mask'):
    #                 # .item()을 써서 CPU로 가져와 정밀하게 계산
    #                 alive_ratio = m.mask.sum().item() / m.mask.numel()
    #                 remaining_params += n_p * alive_ratio
    #             else:
    #                 remaining_params += n_p

    #     sparsity = (1 - remaining_params / (total_params + 1e-9)) * 100
        
    #     # [CCTV] 너무 작아서 안 보일까 봐 실제 개수도 출력해봅니다.
    #     # print(f"[DEBUG] Remaining: {remaining_params} / Total: {total_params}") 
        
    #     return {
    #         'curr_mb': (remaining_params * 4) / (1024**2),
    #         'orig_mb': (total_params * 4) / (1024**2),
    #         'sparsity': sparsity,
    #         'speedup': total_params / (remaining_params + 1e-9)
    #     }
    # def get_model_efficiency(self):
    #     """가중치 내의 0의 개수를 직접 카운트하여 실제 압축률 산출"""
    #     total_params = 0
    #     remaining_params = 0
        
    #     with torch.no_grad():
    #         for name, m in self.model.named_modules():
    #             # 파라미터를 가진 주요 레이어들 대상
    #             if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
    #                 # 1. 원본 파라미터 수 계산
    #                 n_p = m.weight.numel()
    #                 if hasattr(m, 'bias') and m.bias is not None:
    #                     n_p += m.bias.numel()
                    
    #                 total_params += n_p
                    
    #                 # 2. 실제로 살아있는(0이 아닌) 파라미터 수 계산
    #                 # apply_mask_to_weights가 실행되었다면 실제 값이 0으로 밀려있음
    #                 active_w = torch.count_nonzero(m.weight.data).item()
                    
    #                 # 편향(Bias)도 체크
    #                 active_b = 0
    #                 if hasattr(m, 'bias') and m.bias is not None:
    #                     active_b = torch.count_nonzero(m.bias.data).item()
                    
    #                 remaining_params += (active_w + active_b)

    #     # 3. 지표 산출
    #     # 0이 나오는 걸 방지하기 위해 epsilon(1e-9) 추가
    #     sparsity = (1.0 - (remaining_params / (total_params + 1e-9))) * 100.0
        
    #     # 물리적 크기 계산 (float32 기준 4 bytes)
    #     orig_size_mb = (total_params * 4) / (1024**2)
    #     curr_size_mb = (remaining_params * 4) / (1024**2)
        
    #     # 이론적 속도 향상 (FLOPs 감소의 프록시)
    #     theoretical_speedup = total_params / (remaining_params + 1e-9)
        
    #     return {
    #         'orig_mb': orig_size_mb,
    #         'curr_mb': curr_size_mb,
    #         'sparsity': sparsity,
    #         'speedup': theoretical_speedup
    #     }


    def get_model_efficiency(self):
        """마스크 버퍼를 직접 참조하여 이론적 파라미터 및 Sparsity 산출"""
        total_params = 0
        remaining_params = 0
        
        # 모델의 모든 모듈을 순회
        all_modules = dict(self.model.named_modules())
        
        with torch.no_grad():
            for name, m in self.model.named_modules():
                # 파라미터를 가진 주요 레이어 대상 (Conv, Linear, BN)
                if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                    if not hasattr(m, 'weight') or m.weight is None:
                        continue
                        
                    # 1. 원본 파라미터 개수
                    n_p = m.weight.numel()
                    if hasattr(m, 'bias') and m.bias is not None:
                        n_p += m.bias.numel()
                    
                    total_params += n_p
                    
                    # 2. 살아있는 파라미터 개수 계산
                    # 해당 레이어에 직접 마스크가 있는 경우
                    if hasattr(m, 'mask'):
                        keep_ratio = m.mask.sum().item() / m.mask.numel()
                        remaining_params += (n_p * keep_ratio)
                    
                    # 마스크는 없지만 토폴로지 그룹에 속해 연쇄 삭제되는 경우 처리
                    else:
                        # 현재 레이어가 속한 그룹의 마스크가 있는지 확인 (가장 확실한 방법)
                        found_coupling = False
                        if self.topology_groups:
                            for group in self.topology_groups:
                                if name in group or name.replace('.', '_') in group:
                                    # 그룹 내 마스크를 가진 대표 레이어(BN 등) 탐색
                                    for member_name in group:
                                        target_m = all_modules.get(member_name.replace('_', '.'))
                                        if target_m and hasattr(target_m, 'mask'):
                                            keep_ratio = target_m.mask.sum().item() / target_m.mask.numel()
                                            remaining_params += (n_p * keep_ratio)
                                            found_coupling = True
                                            break
                                if found_coupling: break
                        
                        if not found_coupling:
                            # 그룹에도 없고 마스크도 없으면 100% 살아있는 것으로 간주
                            remaining_params += n_p

        # 3. 최종 지표 계산
        sparsity = (1.0 - (remaining_params / (total_params + 1e-9))) * 100.0
        
        # 물리적 크기 (MB)
        orig_mb = (total_params * 4) / (1024**2)
        curr_mb = (remaining_params * 4) / (1024**2)
        
        # 이론적 속도 향상
        speedup = total_params / (remaining_params + 1e-9)
        
        return {
            'orig_mb': orig_mb,
            'curr_mb': curr_mb,
            'sparsity': sparsity,
            'speedup': speedup
        }
    

    def measure_performance_for_rqs(self, example_inputs=None):
        """
        RQ 1, 2, 3에 답변하기 위한 모든 수치를 프루닝 직후 추출합니다.
        """

        # 만약 입력이 없으면 CIFAR-100 표준 사이즈로 자동 생성
        if example_inputs is None:
            # (배치1, 채널3, 높이32, 너비32)
            example_inputs = torch.randn(1, 3, 32, 32).to(self.device)
        
        
        self.model.eval()
        device = next(self.model.parameters()).device
        example_inputs = example_inputs.to(device)

        # --- [RQ 2: Memory Efficiency] Peak/Allocated Memory 측정 ---
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            _ = self.model(example_inputs)
        
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2) # MB
        curr_mem = torch.cuda.memory_allocated(device) / (1024**2) # MB

        # --- [RQ 3: Real Latency] 추론 속도 측정 (Warming up 포함) ---
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 50
        timings = np.zeros((repetitions, 1))
        
        with torch.no_grad():
            for _ in range(10): # Warm up
                _ = self.model(example_inputs)
            
            for rep in range(repetitions):
                starter.record()
                _ = self.model(example_inputs)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time

        avg_latency = np.mean(timings) # ms

        # --- [RQ 1 & 2: Sparsity & Size] 기존 정밀 계산 로직 활용 ---
        total_params = 0
        remaining_params = 0
        all_modules = dict(self.model.named_modules())
        
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                if not hasattr(m, 'weight') or m.weight is None: continue
                n_p = m.weight.numel()
                if hasattr(m, 'bias') and m.bias is not None: n_p += m.bias.numel()
                total_params += n_p
                
                # 마스크 기반 생존율 계산
                if hasattr(m, 'mask'):
                    keep_ratio = m.mask.sum().item() / m.mask.numel()
                    remaining_params += (n_p * keep_ratio)
                else:
                    # Coupling된 레이어 체크 (이전 코드 로직)
                    found_coupling = False
                    for group in self.topology_groups:
                        if name in group or name.replace('.', '_') in group:
                            for member in group:
                                target_m = all_modules.get(member.replace('_', '.'))
                                if target_m and hasattr(target_m, 'mask'):
                                    keep_ratio = target_m.mask.sum().item() / target_m.mask.numel()
                                    remaining_params += (n_p * keep_ratio)
                                    found_coupling = True; break
                        if found_coupling: break
                    if not found_coupling: remaining_params += n_p

        sparsity = (1.0 - (remaining_params / (total_params + 1e-9))) * 100
        
        return {
            'Sparsity': sparsity,
            'Params_MB': (remaining_params * 4) / (1024**2),
            'Peak_VRAM_MB': peak_mem,
            'Latency_ms': avg_latency,
            'Reduction_Ratio': total_params / (remaining_params + 1e-9)
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
    HAP 정석 로직:
    1. 레이어별(그룹별) Hessian Trace의 역수를 이용해 Sparsity 비율을 동적으로 할당.
    2. 각 그룹 내에서 제거할 채널을 선정할 때도 Hessian Trace(Energy) 점수를 기준으로 사용.
    """
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

        # 2. Hessian 계산 (전체 레이어 대상)
        target_params = [m.weight for g in group_info_list for m in g['layers']]
        hv_list = self.engine.get_k_step_hessian_selective(loss, target_params, self.k_horizon)

        # 3. 레이어별 Hessian Trace 및 필터별 Hessian Score 수집
        group_traces = []
        hv_idx = 0
        for g in group_info_list:
            layer_traces = []
            for m in g['layers']:
                hv = hv_list[hv_idx]
                # 필터별 Hessian Energy 계산 및 저장
                h_energy = hv.pow(2).reshape(hv.shape[0], -1).mean(1)
                m.hessian_score.copy_(h_energy)
                
                # 레이어 전체의 Trace (평균 에너지)
                layer_traces.append(h_energy.mean().item())
                hv_idx += 1
            group_traces.append(np.mean(layer_traces))

        # 4. [HAP 핵심] Hessian에 반비례하게 Sparsity 할당
        # 민감도(Sensitivity) S_i = 1 / Trace_i
        sensitivities = [1.0 / (t + 1e-8) for t in group_traces]
        total_sens = sum(sensitivities)
        avg_sens = total_sens / len(group_info_list)
        
        pruned_count_total = 0
        print(f"\n{'='*30} HAP Standard Pruning: Epoch {current_epoch} {'='*30}")
        
        for i, g in enumerate(group_info_list):
            # 그룹별 할당 Sparsity = (전체 목표) * (상대적 민감도 비율)
            group_sparsity = total_target_sparsity * (sensitivities[i] / avg_sens)
            group_sparsity = min(group_sparsity, 1.0 - self.min_survival_ratio) # 최소 생존 보장
            
            mask = g['layers'][0].mask
            num_channels = mask.numel()
            num_prune = int(num_channels * group_sparsity)

            if num_prune > 0:
                # [수정 지점] 가중치 크기(L1) 대신 Hessian Score를 기준으로 필터 선정
                # 그룹 내 레이어들의 채널 크기가 다를 수 있으므로(EfficientNet 등), 
                # 첫 번째 레이어(기준)와 크기가 같은 레이어들의 Hessian 점수만 평균냄
                base_size = num_channels
                hessian_mags = [m.hessian_score for m in g['layers'] if m.hessian_score.size(0) == base_size]
                
                if hessian_mags:
                    avg_hessian_score = torch.mean(torch.stack(hessian_mags), dim=0)
                else:
                    avg_hessian_score = g['layers'][0].hessian_score

                # Hessian 점수가 낮은(중요도가 낮은) 순서대로 인덱스 추출
                _, prune_indices = torch.topk(avg_hessian_score, k=num_prune, largest=False)
                
                with torch.no_grad():
                    for ln in g['names']:
                        layer_obj = find_layer(ln)
                        if layer_obj is not None and hasattr(layer_obj, 'mask'):
                            # 채널 수가 맞는 레이어에만 해당 인덱스 마스킹
                            if layer_obj.mask.size(0) == base_size:
                                layer_obj.mask[prune_indices] = 0.0
                
                pruned_count_total += num_prune

            print(f" Group {g['id']:2d} | Assigned Sparsity: {group_sparsity*100:4.1f}% | Alive: {num_channels-num_prune:4d}/{num_channels:4d}")

        print(f" [*] Total Pruned in this step: {pruned_count_total}")
        
        # 5. 학회용 리소스 분석 출력
        eff = self.get_model_efficiency()
        print(f"\n[Scientific Metrics - Epoch {current_epoch}]")
        print(f" 🟢 Model Size: {eff['orig_mb']:.2f} MB -> {eff['curr_mb']:.2f} MB")
        print(f" 🔵 Sparsity (Params/FLOPs): {eff['sparsity']:.2f} %")
        print(f" 🟡 Theoretical Speedup: {eff['speedup']:.2f}x")
        print(f" 🟠 Current GPU Mem (Allocated): {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"{'='*89}\n")
        torch.cuda.empty_cache()
class SNOWSPruner(PDTPruner):
    """
    SNOWS 논문 로직: 중요도 = 순수 Hessian_Trace
    (Grad EMA 보정 없이 순수한 현재 배치의 Hessian 에너지만 사용하며, Global Ranking으로 프루닝)
    """
    def step_pruning(self, loss, current_epoch, total_epochs):
        all_modules = dict(self.model.named_modules())
        def find_layer(name):
            if name in all_modules: return all_modules[name]
            return all_modules.get(name.replace('_', '.'))

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
        target_unit_metadata = []

        hv_idx = 0
        for g in group_info_list:
            mask = g['layers'][0].mask
            alive_indices = torch.where(mask > 0.5)[0].cpu().numpy()
            
            # 각 레이어의 Hessian Trace 계산 (H-Vector Product의 2-norm 제곱 평균)
            layer_hessians = []
            for m in g['layers']:
                hv = hv_list[hv_idx]; hv_idx += 1
                h_energy = hv.pow(2).reshape(hv.shape[0], -1).mean(1)
                layer_hessians.append(h_energy)
            
            # [수정 지점] 그룹 내 레이어들의 채널 수가 달라도 인덱스 에러가 나지 않도록 처리
            # 1. 각 레이어별로 채널 평균 Hessian 에너지를 구함
            # 2. 그룹 전체의 평균 에너지를 하나의 대표값으로 산출
            layer_means = [torch.mean(h) for h in layer_hessians]
            avg_group_hessian_value = torch.mean(torch.stack(layer_means)).item()

            if len(alive_indices) > 0:
                for i in alive_indices:
                    # [수정] 인덱싱([i]) 대신, 위에서 구한 그룹 대표값을 모든 채널에 동일하게 부여
                    # SNOWS의 Global Ranking을 위해 그룹 단위의 중요도를 채널 점수로 사용합니다.
                    s_gc = avg_group_hessian_value 
                    
                    target_unit_scores.append(s_gc) 
                    target_unit_metadata.append((g, i))

        # [변경 지점] 라그랑주 엔진 대신, 부모 클래스의 Global Ranking 함수 호출
        # 이 함수 내부에서 정렬, 마스킹, 학회용 리소스 출력까지 한 번에 수행
        self._global_rank_prune(
            scores=target_unit_scores, 
            metadata=target_unit_metadata, 
            epoch=current_epoch, 
            total_epochs=total_epochs, 
            method_name="SNOWS"
        )


class ATOPruner(PDTPruner):
    """
    ATO 논문의 핵심 개념: Magnitude(L1-norm) 기반 점진적 프루닝
    Hessian을 계산하지 않고, 가중치의 크기(L1-norm)를 중요도로 사용합니다.
    라그랑주 최적화 대신 Global Ranking 방식을 사용하여 대조군으로서의 신빙성을 확보합니다.
    """
    def step_pruning(self, loss, current_epoch, total_epochs):
        all_modules = dict(self.model.named_modules())
        def find_layer(name):
            if name in all_modules: return all_modules[name]
            return all_modules.get(name.replace('_', '.'))

        group_info_list = []
        if self.topology_groups:
            for idx, group in enumerate(self.topology_groups):
                score_layers = [find_layer(ln) for ln in group if isinstance(find_layer(ln), (nn.Conv2d, nn.Linear))]
                if not score_layers: continue
                group_info_list.append({'id': idx+1, 'layers': score_layers, 'names': group})

        if not group_info_list: return

        target_unit_scores = []
        target_unit_metadata = []

        # [ATO 핵심 로직] Hessian 대신 L1-norm(Magnitude) 계산
        for g in group_info_list:
            mask = g['layers'][0].mask
            alive_indices = torch.where(mask > 0.5)[0].cpu().numpy()
            
            # 그룹 내 모든 레이어의 Magnitude를 합산하여 중요도 판단
            magnitude_scores = []
            for m in g['layers']:
                # (out_channels, in_channels, k, k) -> (out_channels,)
                m_score = m.weight.data.abs().reshape(m.weight.shape[0], -1).mean(1)
                magnitude_scores.append(m_score)
            
            # --- [수정 지점] 채널 수가 다른 레이어 간의 stack 에러 방지 ---
            layer_mag_means = [torch.mean(ms) for ms in magnitude_scores]
            group_magnitude_value = torch.mean(torch.stack(layer_mag_means)).item()

            if len(alive_indices) > 0:
                for i in alive_indices:
                    # 해당 채널의 중요도 점수: 그룹 전체의 평균 Magnitude 부여
                    s_gc = group_magnitude_value
                    
                    target_unit_scores.append(s_gc) 
                    target_unit_metadata.append((g, i))

        # 부모 클래스의 Global Ranking 함수 호출
        self._global_rank_prune(
            scores=target_unit_scores, 
            metadata=target_unit_metadata, 
            epoch=current_epoch, 
            total_epochs=total_epochs, 
            method_name="ATO"
        )

    
# ==============================================================================
# SuperTickets (ST - Gradient-Weight Product based) 비교 실험용 Pruner
# ==============================================================================
class STPruner(PDTPruner):
    """
    SuperTickets 개념: Gradient와 Weight의 곱을 중요도로 사용
    단순 Magnitude보다 학습의 기여도를 더 정확히 포착한다고 가정.
    라그랑주 최적화 대신 Global Ranking 방식을 사용하여 ST 논문의 정석 로직을 재현.
    """
    def step_pruning(self, loss, current_epoch, total_epochs):
        all_modules = dict(self.model.named_modules())
        def find_layer(name):
            if name in all_modules: return all_modules[name]
            return all_modules.get(name.replace('_', '.'))

        group_info_list = []
        if self.topology_groups:
            for idx, group in enumerate(self.topology_groups):
                score_layers = [find_layer(ln) for ln in group if isinstance(find_layer(ln), (nn.Conv2d, nn.Linear))]
                if not score_layers: continue
                group_info_list.append({'id': idx+1, 'layers': score_layers, 'names': group})

        if not group_info_list: return

        target_unit_scores = []
        target_unit_metadata = []

        # [ST 핵심 로직] Weight * Gradient_EMA 를 중요도로 사용
        for g in group_info_list:
            mask = g['layers'][0].mask
            alive_indices = torch.where(mask > 0.5)[0].cpu().numpy()
            
            st_scores = []
            for m in g['layers']:
                # SuperTickets의 핵심: |W * dL/dW|
                # (W의 절대값) * (Grad EMA의 root) -> 점수 산출
                # view(-1)이나 reshape을 통해 채널별 점수로 요약
                score = (m.weight.data.abs() * m.grad_ema.reshape(m.weight.shape[0], 1, 1, 1).sqrt()).reshape(m.weight.shape[0], -1).mean(1)
                st_scores.append(score)
            
            # --- [핵심 수정 지점] stack 에러 원천 차단 ---
            # 레이어마다 채널 수가 달라도 동작하도록 각 레이어의 평균 점수를 먼저 계산
            layer_st_means = [torch.mean(s) for s in st_scores]
            # 단일 값(Scalar)들을 stack 하여 그룹 전체 평균값 산출
            group_st_value = torch.mean(torch.stack(layer_st_means)).item()

            if len(alive_indices) > 0:
                for i in alive_indices:
                    # 인덱스 에러 방지를 위해 그룹 대표값을 모든 채널에 동일하게 부여
                    s_gc = group_st_value 
                    
                    target_unit_scores.append(s_gc) 
                    target_unit_metadata.append((g, i))

        # 부모 클래스의 Global Ranking 함수 호출 (이 안에서 pruning 수행)
        self._global_rank_prune(
            scores=target_unit_scores, 
            metadata=target_unit_metadata, 
            epoch=current_epoch, 
            total_epochs=total_epochs, 
            method_name="ST"
        )

# ==============================================================================
# DFPC (Data-Free Parameter Compensation - Similarity based) 비교 실험용 Pruner
# ==============================================================================
class DFPCPruner(PDTPruner):
    """
    DFPC 개념: 데이터 없이 필터 자체의 기하학적 분포를 분석
    필터 간의 거리가 멀수록(고유할수록) 중요하다고 판단
    라그랑주 최적화 대신 Global Ranking 방식을 사용하여 필터 고유성 기반 프루닝을 재현
    """
    def step_pruning(self, loss, current_epoch, total_epochs):
        all_modules = dict(self.model.named_modules())
        def find_layer(name):
            if name in all_modules: return all_modules[name]
            return all_modules.get(name.replace('_', '.'))

        group_info_list = []
        if self.topology_groups:
            for idx, group in enumerate(self.topology_groups):
                score_layers = [find_layer(ln) for ln in group if isinstance(find_layer(ln), (nn.Conv2d, nn.Linear))]
                if not score_layers: continue
                group_info_list.append({'id': idx+1, 'layers': score_layers, 'names': group})

        if not group_info_list: return

        target_unit_scores = []
        target_unit_metadata = []

        # [DFPC 핵심 로직] 필터 간의 L2 Distance (Uniqueness)를 중요도로 사용
        for g in group_info_list:
            mask = g['layers'][0].mask
            alive_indices = torch.where(mask > 0.5)[0].cpu().numpy()
            # DFPCPruner의 step_pruning 내 수정
            group_dfpc_scores = []
            for m in g['layers']:
                w = m.weight.data.reshape(m.weight.shape[0], -1)
                dist_matrix = torch.cdist(w, w, p=2)
                importance = dist_matrix.sum(dim=1) 
                group_dfpc_scores.append(importance.mean().item()) # 레이어별 평균값 저장

            # 그룹 전체의 평균 고유성 점수 산출
            group_score_value = np.mean(group_dfpc_scores)

            if len(alive_indices) > 0:
                for i in alive_indices:
                    # 모든 채널 i에 대해 동일한 그룹 평균 점수를 부여하여 랭킹 경쟁
                    target_unit_scores.append(group_score_value) 
                    target_unit_metadata.append((g, i))



            #group_dfpc_scores = []
            #for m in g['layers']:
                # weight shape: [out_channels, in_channels * k * k]
             #   w = m.weight.data.reshape(m.weight.shape[0], -1)
                
                # 각 필터가 다른 모든 필터들과 얼마나 다른지(L2 distance의 합) 계산
                # 거리가 가까울수록(점수가 낮을수록) 중복된 정보라고 판단하여 프루닝 대상이 됨
            #    dist_matrix = torch.cdist(w, w, p=2)
            #    importance = dist_matrix.sum(dim=1) 
            #    group_dfpc_scores.append(importance)
            
            # 그룹 내 레이어들의 유사도 점수 평균 산출
            # 수정된 코드: 각 레이어의 중요도 점수를 스칼라(평균값)로 변환 후 평균 산출
            #layer_score_means = [torch.mean(score) for score in group_dfpc_scores]
            #group_score_value = torch.mean(torch.stack(layer_score_means)).item()

            # 이후 루프에서 s_gc를 단일 평균값으로 할당
            #if len(alive_indices) > 0:
           #     for i in alive_indices:
                    # 그룹 내 모든 레이어의 기하학적 유사도 평균을 채널 점수로 사용
          #          s_gc = group_score_value 
                    
         #           target_unit_scores.append(s_gc) 
        #            target_unit_metadata.append((g, i))


            #group_score = torch.mean(torch.stack(group_dfpc_scores), dim=0)

            #if len(alive_indices) > 0:
             #   for i in alive_indices:
                    # 해당 채널의 중요도 점수: Geometric Uniqueness
              #      s_gc = group_score[i].item()
                    
               #     target_unit_scores.append(s_gc) 
                #    target_unit_metadata.append((g, i))

        # [변경 지점] 라그랑주 엔진 대신 Global Ranking 함수 호출
        # 모델 전체에서 기하학적으로 가장 '뻔한(유사한)' 필터들부터 차례대로 제거합니다.
        self._global_rank_prune(
            scores=target_unit_scores, 
            metadata=target_unit_metadata, 
            epoch=current_epoch, 
            total_epochs=total_epochs, 
            method_name="DFPC"
        )

# ==============================================================================
# TPP (Towards Personalized Pruning - Weight-Activation Interaction) 비교 실험용 Pruner
# ==============================================================================
class TPPPruner(PDTPruner):
    """
    TPP 개념: 가중치의 절대값과 활성화 기여도(Gradient 활용)의 곱을 통해
    채널별 '개인화된' 중요도를 산출
    라그랑주 최적화 대신 Global Ranking 방식을 사용하여 TPP 논문 원안의 중요도 배분 로직을 재현
    """
    def step_pruning(self, loss, current_epoch, total_epochs):
        all_modules = dict(self.model.named_modules())
        def find_layer(name):
            if name in all_modules: return all_modules[name]
            return all_modules.get(name.replace('_', '.'))

        group_info_list = []
        if self.topology_groups:
            for idx, group in enumerate(self.topology_groups):
                score_layers = [find_layer(ln) for ln in group if isinstance(find_layer(ln), (nn.Conv2d, nn.Linear))]
                if not score_layers: continue
                group_info_list.append({'id': idx+1, 'layers': score_layers, 'names': group})

        if not group_info_list: return

        target_unit_scores = []
        target_unit_metadata = []

        # [TPP 핵심 로직] Weight Magnitude * Gradient Persistence (Grad-EMA)
        for g in group_info_list:
            mask = g['layers'][0].mask
            alive_indices = torch.where(mask > 0.5)[0].cpu().numpy()
            
            tpp_scores = []
            for m in g['layers']:
                # TPP 논문의 핵심: Weight-Activation Interaction
                w_abs = m.weight.data.abs().reshape(m.weight.shape[0], -1).mean(1)
                g_ema = m.grad_ema.reshape(m.weight.shape[0], -1).mean(1)
                
                # 가중치와 그래디언트 영향력을 결합
                score = w_abs * torch.sqrt(g_ema + 1e-8)
                tpp_scores.append(score)
            
            # --- [수정 지점] 채널 수가 다른 레이어 간의 stack 에러 방지 ---
            # 각 레이어별 TPP 점수의 평균을 구한 뒤, 그 값들의 평균을 그룹 대표값으로 사용
            layer_tpp_means = [torch.mean(s) for s in tpp_scores]
            group_tpp_value = torch.mean(torch.stack(layer_tpp_means)).item()

            if len(alive_indices) > 0:
                for i in alive_indices:
                    # 해당 채널의 중요도 점수: 그룹 전체의 Personalized Interaction Score 부여
                    s_gc = group_tpp_value
                    
                    target_unit_scores.append(s_gc) 
                    target_unit_metadata.append((g, i))

        # 부모 클래스의 Global Ranking 함수 호출
        self._global_rank_prune(
            scores=target_unit_scores, 
            metadata=target_unit_metadata, 
            epoch=current_epoch, 
            total_epochs=total_epochs, 
            method_name="TPP"
        )