import torch
import torch.nn as nn
from .base import BasePruner
from .engine.hessian_free import SNOWSEngine
import numpy as np
from .optimizer import lagrangian_optimization 
import sys
import time
import gc
import os
class PDTPruner(BasePruner):


    def __init__(self, model, config, args=None, topology_groups=None):
        super().__init__(model, config)
        
        strat_cfg = config.get('strategy', {})
        self.group_selection_ratio = float(strat_cfg.get('group_selection_ratio', 1.0))
        self.group_selection_ratio = min(max(self.group_selection_ratio, 0.0), 1.0)
        self.pruning_ratio = float(strat_cfg.get('pruning_ratio', 0.8))
        self.min_survival_ratio = strat_cfg.get('min_survival_ratio', 0.4)

        # args 우선순위 적용

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
        debug_mask_consistency = os.getenv('MCPRUNE_DEBUG_MASK_CONSISTENCY') == '1'
        if debug_mask_consistency and not hasattr(self, '_mask_debug_count'):
            self._mask_debug_count = 0
        max_debug_events = int(os.getenv('MCPRUNE_DEBUG_MASK_CONSISTENCY_MAX', '8'))
        debug_this_call = (
            debug_mask_consistency
            and self._mask_debug_count < max_debug_events
        )
        pruned_units = 0
        weight_nonzero_before = 0
        weight_nonzero_after = 0
        state_nonzero_before = 0
        state_nonzero_after = 0

        with torch.no_grad():
            for m in self.layers:
                mask = m.mask
                m_view = mask.view(-1, 1, 1, 1) if m.weight.dim() == 4 else mask.view(-1, 1)
                if debug_this_call:
                    pruned_mask = mask <= 0.5
                    if pruned_mask.any():
                        pruned_units += int(pruned_mask.sum().item())
                        w_pruned_view = (
                            pruned_mask.view(-1, 1, 1, 1)
                            if m.weight.dim() == 4
                            else pruned_mask.view(-1, 1)
                        )
                        w_pruned_view = w_pruned_view.expand_as(m.weight)
                        weight_nonzero_before += int(
                            (m.weight.data[w_pruned_view] != 0).sum().item()
                        )
                        if optimizer is not None and m.weight in optimizer.state:
                            for state_value in optimizer.state[m.weight].values():
                                if torch.is_tensor(state_value) and state_value.shape == m.weight.shape:
                                    state_nonzero_before += int(
                                        (state_value[w_pruned_view] != 0).sum().item()
                                    )

                m.weight.data.mul_(m_view)
                if optimizer is not None and m.weight in optimizer.state:
                    for state_value in optimizer.state[m.weight].values():
                        if torch.is_tensor(state_value) and state_value.shape == m.weight.shape:
                            state_value.mul_(m_view)
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.mul_(mask)
                    if optimizer is not None and m.bias in optimizer.state:
                        for state_value in optimizer.state[m.bias].values():
                            if torch.is_tensor(state_value) and state_value.shape == m.bias.shape:
                                state_value.mul_(mask)

                if debug_this_call:
                    pruned_mask = mask <= 0.5
                    if pruned_mask.any():
                        w_pruned_view = (
                            pruned_mask.view(-1, 1, 1, 1)
                            if m.weight.dim() == 4
                            else pruned_mask.view(-1, 1)
                        )
                        w_pruned_view = w_pruned_view.expand_as(m.weight)
                        weight_nonzero_after += int(
                            (m.weight.data[w_pruned_view] != 0).sum().item()
                        )
                        if optimizer is not None and m.weight in optimizer.state:
                            for state_value in optimizer.state[m.weight].values():
                                if torch.is_tensor(state_value) and state_value.shape == m.weight.shape:
                                    state_nonzero_after += int(
                                        (state_value[w_pruned_view] != 0).sum().item()
                                    )

        if debug_this_call and pruned_units > 0:
            print(
                "[Mask Consistency] "
                f"pruned_units={pruned_units} "
                f"weight_nonzero_before={weight_nonzero_before} "
                f"weight_nonzero_after={weight_nonzero_after} "
                f"optimizer_state_nonzero_before={state_nonzero_before} "
                f"optimizer_state_nonzero_after={state_nonzero_after} "
                f"optimizer={'enabled' if optimizer is not None else 'missing'}"
            )
            self._mask_debug_count += 1

    def update_ema_and_mask_grad(self):
        with torch.no_grad():
            for m in self.layers:
                if hasattr(m, 'weight') and m.weight.grad is not None:
                    m_view = m.mask.view(-1, 1, 1, 1) if m.weight.dim()==4 else m.mask.view(-1, 1)
                    m.weight.grad.mul_(m_view)
                    g = m.weight.grad.pow(2).reshape(m.weight.shape[0], -1).mean(1)
                    m.grad_ema.mul_(self.ema_decay).add_(g, alpha=1 - self.ema_decay)

    def _select_candidate_groups(self, group_info_list):
        if not group_info_list or self.group_selection_ratio >= 1.0:
            return group_info_list

        scores = [g['w_g'] for g in group_info_list]
        if max(scores) - min(scores) <= 1e-12:
            print("[Group Selection] Skipped: Grad-EMA scores are not differentiated yet.")
            return group_info_list

        n_select = max(1, int(np.ceil(len(group_info_list) * self.group_selection_ratio)))
        ranked_groups = sorted(group_info_list, key=lambda g: g['w_g'])
        if os.getenv('MCPRUNE_DEBUG_GROUP_SELECTION') == '1':
            print("[Group Selection] Grad-EMA ranking, low to high")
            for rank, g in enumerate(ranked_groups, 1):
                names = ", ".join(g['names'])
                print(f"  rank {rank:02d} | group {g['id']:02d} | grad_ema={g['w_g']:.6e} | {names}")

        selected = ranked_groups[:n_select]
        selected_ids = {id(g) for g in selected}
        print(
            f"[Group Selection] Selected {n_select}/{len(group_info_list)} "
            f"low-Grad-EMA groups as pruning candidates."
        )
        for g in selected[:10]:
            names = ", ".join(g['names'][:3])
            if len(g['names']) > 3:
                names += ", ..."
            print(f"  - group {g['id']:02d} | grad_ema={g['w_g']:.6e} | {names}")
        return [g for g in group_info_list if id(g) in selected_ids]

    def step_pruning(self, loss, current_epoch, total_epochs):
        all_modules = dict(self.model.named_modules())
        def find_layer(name):
            if name in all_modules: return all_modules[name]
            return all_modules.get(name.replace('_', '.'))

        

        # ---  [최종 수정] ---
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
        target_remaining_ratio = 1.0 - (progress * self.pruning_ratio)
        
        print(f"\n[DEBUG] !!! PRUNING TRIGGERED !!! Epoch {current_epoch}/{actual_total}")

        # [핵심 수정] 모든 타입의 레이어를 일단 수집
        group_info_list = []
        module_name_by_id = {id(m): name for name, m in all_modules.items()}
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

        group_info_list = self._select_candidate_groups(group_info_list)

        # --- [이후 Hessian 및 Pruning 로직] ---
        debug_hvp_alignment = os.getenv('MCPRUNE_DEBUG_HVP_ALIGNMENT') == '1'
        if debug_hvp_alignment:
            print("[HVP Alignment] Selected scoring layers by group")
            for g in group_info_list:
                layer_names = [
                    module_name_by_id.get(id(m), "<unnamed>")
                    for m in g['layers']
                    if hasattr(m, 'weight')
                ]
                print(f"  group {g['id']:02d}: {layer_names}")

        target_param_layers = [
            (module_name_by_id.get(id(m), "<unnamed>"), m.weight)
            for g in group_info_list
            for m in g['layers']
            if hasattr(m, 'weight')
        ]
        target_params = [param for _, param in target_param_layers]
        print(f"[Hessian Input] Passing {len(target_params)} layer weights to HVP.")
        for name, _ in target_param_layers[:15]:
            print(f"  - {name}")
        if len(target_param_layers) > 15:
            print(f"  - ... ({len(target_param_layers) - 15} more)")
        if debug_hvp_alignment:
            hvp_names = [name for name, _ in target_param_layers]
            duplicate_hvp_names = sorted({name for name in hvp_names if hvp_names.count(name) > 1})
            print(f"[HVP Alignment] HVP input layers: {hvp_names}")
            print(f"[HVP Alignment] Duplicate HVP input layers: {duplicate_hvp_names or 'none'}")
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

        if target_unit_scores and (current_alive_ratio > target_remaining_ratio - 0.05):
            target_ratio = target_remaining_ratio / (current_alive_ratio + 1e-7)
            optimal_mask_flags = lagrangian_optimization(
                np.array(target_unit_scores),
                np.array(target_unit_costs),
                np.sum(target_unit_costs) * target_ratio,
                unit_metadata=target_unit_metadata
            )
            
            # FORCE
            if np.all(optimal_mask_flags == 1) and target_ratio < 0.999:
                num_force = int(len(optimal_mask_flags) * (1 - target_ratio))
                optimal_mask_flags[np.argsort(target_unit_scores)[:num_force]] = 0

            remaining_by_layer = {}
            min_keep_by_layer = {}
            layer_name_by_id = {id(m): name for name, m in all_modules.items()}
            debug_min_survival = os.getenv('MCPRUNE_DEBUG_MIN_SURVIVAL') == '1'
            for g in group_info_list:
                for layer in g['layers']:
                    if not hasattr(layer, 'mask'):
                        continue
                    layer_key = id(layer)
                    if layer_key in remaining_by_layer:
                        continue
                    remaining_by_layer[layer_key] = int(layer.mask.sum().item())
                    min_keep_by_layer[layer_key] = max(
                        1,
                        int(layer.mask.numel() * self.min_survival_ratio)
                    )
                    if debug_min_survival:
                        print(
                            f"[Min Survival] layer={layer_name_by_id.get(layer_key, '<unnamed>')} "
                            f"alive_before={remaining_by_layer[layer_key]} "
                            f"min_keep={min_keep_by_layer[layer_key]} "
                            f"ratio={self.min_survival_ratio:.2f}"
                        )

            with torch.no_grad():
                for idx, is_alive in enumerate(optimal_mask_flags):
                    if not is_alive:
                        g_obj, ch_idx = target_unit_metadata[idx]
                        affected_layers = []
                        for ln in g_obj['names']:
                            l_obj = find_layer(ln)
                            if l_obj is not None and hasattr(l_obj, 'mask') and ch_idx < l_obj.mask.size(0):
                                affected_layers.append(l_obj)

                        if any(
                            remaining_by_layer.get(id(layer), 0) <= min_keep_by_layer.get(id(layer), 1)
                            for layer in affected_layers
                        ):
                            if debug_min_survival:
                                blocked = [
                                    layer_name_by_id.get(id(layer), '<unnamed>')
                                    for layer in affected_layers
                                    if remaining_by_layer.get(id(layer), 0) <= min_keep_by_layer.get(id(layer), 1)
                                ]
                                print(
                                    f"[Min Survival] block prune group={g_obj['id']:02d} "
                                    f"unit={ch_idx} blocked_layers={blocked}"
                                )
                            continue
                        for l_obj in affected_layers:
                            l_obj.mask.data[ch_idx] = 0.0
                            remaining_by_layer[id(l_obj)] -= 1
                        pruned_count += 1

                total_param_cost = sum(m.weight.numel() for m in self.layers)
                current_pruned_cost = sum(
                    m.weight.numel() * (1.0 - m.mask.float().mean().item())
                    for m in self.layers
                )
                target_pruned_cost = (
                    total_param_cost * progress * self.pruning_ratio
                )
                repair_count = 0

                if current_pruned_cost < target_pruned_cost:
                    repair_order = np.argsort(
                        np.asarray(target_unit_scores)
                        / (np.asarray(target_unit_costs) + 1e-12)
                    )
                    for idx in repair_order:
                        if current_pruned_cost >= target_pruned_cost:
                            break

                        g_obj, ch_idx = target_unit_metadata[idx]
                        affected_layers = [
                            layer for layer in g_obj['layers']
                            if hasattr(layer, 'mask') and ch_idx < layer.mask.size(0)
                        ]
                        if not affected_layers or all(
                            layer.mask[ch_idx] <= 0.5 for layer in affected_layers
                        ):
                            continue
                        if any(
                            remaining_by_layer.get(id(layer), 0)
                            <= min_keep_by_layer.get(id(layer), 1)
                            for layer in affected_layers
                        ):
                            continue

                        unit_cost = 0.0
                        for layer in affected_layers:
                            if layer.mask[ch_idx] > 0.5:
                                layer.mask.data[ch_idx] = 0.0
                                remaining_by_layer[id(layer)] -= 1
                                unit_cost += layer.weight.numel() / layer.weight.shape[0]
                        current_pruned_cost += unit_cost
                        pruned_count += 1
                        repair_count += 1

                print(
                    f"[Ratio Repair] additional_units={repair_count} "
                    f"target_param_cost={target_pruned_cost:.0f} "
                    f"actual_param_cost={current_pruned_cost:.0f}"
                )
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
        print(
            f"\n[Scientific Metrics] Size: {eff['curr_mb']:.2f}MB | "
            f"Sparsity: {eff['sparsity']:.2f}% | Speedup: {eff['speedup']:.2f}x"
        )
        print(f"{'='*89}\n")

    
    
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
        total_target_sparsity = progress * self.pruning_ratio
        
        num_total = len(scores)
        total_param_cost = sum(m.weight.numel() for m in self.layers)
        current_pruned_cost = sum(
            m.weight.numel() * (1.0 - m.mask.float().mean().item())
            for m in self.layers
        )
        target_pruned_cost = total_param_cost * total_target_sparsity
        selected_indices = []

        for idx in np.argsort(scores):
            if current_pruned_cost >= target_pruned_cost:
                break
            group_obj, channel_idx = metadata[idx]
            unit_cost = 0.0
            for layer_name in group_obj['names']:
                layer = self.model.get_submodule(layer_name.replace('_', '.'))
                if (
                    hasattr(layer, 'mask')
                    and channel_idx < layer.mask.size(0)
                    and layer.mask[channel_idx] > 0.5
                ):
                    unit_cost += layer.weight.numel() / layer.weight.shape[0]
            if unit_cost > 0:
                selected_indices.append(idx)
                current_pruned_cost += unit_cost

        num_to_prune = len(selected_indices)

        if num_to_prune > 0:
            # 2. 점수가 낮은 순서대로 정렬 (Global Ranking)
            indices = selected_indices
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
        print(
            f" [Ratio Budget] target_param_cost={target_pruned_cost:.0f} "
            f"actual_param_cost={current_pruned_cost:.0f}"
        )
        
        # 3.학회용 측정치 출력 (통합 호출)
        eff = self.get_model_efficiency()
        print(f"\n[Scientific Metrics - Epoch {epoch}]")
        print(f" 🟢 Model Size: {eff['orig_mb']:.2f} MB -> {eff['curr_mb']:.2f} MB")
        print(f" 🔵 Sparsity (Params/FLOPs): {eff['sparsity']:.2f} %")
        print(f" 🟡 Theoretical Speedup: {eff['speedup']:.2f}x")
        print(f" 🟠 Current GPU Mem (Allocated): {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"{'='*89}\n")
        torch.cuda.empty_cache()


class ViTPDTPruner(PDTPruner):
    """
    ViT/DeiT 전용 PDT Pruner.

    topology_groups 형태:
      {'type': 'ffn',  'names': [fc1, fc2]}
      {'type': 'attn', 'names': [qkv, proj], 'num_heads': 6, 'head_dim': 64}

    마스킹 단위:
      ffn  → 뉴런 단위 (fc1 out = fc2 in 동기화)
      attn → head 단위 (qkv 3슬라이스 + proj 열 동기화)
    """

    def _check_buffers(self):
        """
        Linear 레이어에 버퍼 등록.
        ffn:  mask 크기 = fc1.out_features (중간 차원)
        attn: mask 크기 = num_heads
        """
        if not self.topology_groups:
            return

        all_modules = dict(self.model.named_modules())

        for g in self.topology_groups:
            names = g['names']

            if g['type'] == 'ffn':
                # fc1의 출력 차원 기준
                fc1 = all_modules.get(names[0])
                if fc1 is None:
                    continue
                n = fc1.out_features  # 1536
                for name in names:
                    m = all_modules.get(name)
                    if m is None:
                        continue
                    if not hasattr(m, 'mask'):
                        m.register_buffer('mask',
                            torch.ones(n, device=m.weight.device))
                    if not hasattr(m, 'grad_ema'):
                        m.register_buffer('grad_ema',
                            torch.zeros(n, device=m.weight.device))
                    if not hasattr(m, 'hessian_score'):
                        m.register_buffer('hessian_score',
                            torch.zeros(n, device=m.weight.device))

            elif g['type'] == 'attn':
                # head 수 기준
                num_heads = g['num_heads']
                for name in names:
                    m = all_modules.get(name)
                    if m is None:
                        continue
                    if not hasattr(m, 'mask'):
                        m.register_buffer('mask',
                            torch.ones(num_heads, device=m.weight.device))
                    if not hasattr(m, 'grad_ema'):
                        m.register_buffer('grad_ema',
                            torch.zeros(num_heads, device=m.weight.device))
                    if not hasattr(m, 'hessian_score'):
                        m.register_buffer('hessian_score',
                            torch.zeros(num_heads, device=m.weight.device))

        # self.layers: sparsity 계산용 (마스크 있는 레이어만)
        self.layers = [m for _, m in self.model.named_modules()
                       if isinstance(m, nn.Linear) and hasattr(m, 'mask')]

    def apply_mask_to_weights(self, optimizer=None):
        """
        타입별 마스킹:
        - ffn:  fc1 행 마스킹 + fc2 열 마스킹
        - attn: qkv 행 슬라이스 마스킹 + proj 열 슬라이스 마스킹
        """
        if not self.topology_groups:
            return

        all_modules = dict(self.model.named_modules())

        with torch.no_grad():
            for g in self.topology_groups:

                if g['type'] == 'ffn':
                    fc1 = all_modules.get(g['names'][0])
                    fc2 = all_modules.get(g['names'][1])
                    if fc1 is None or fc2 is None:
                        continue
                    mask = fc1.mask  # [1536]

                    # fc1: 행(출력 뉴런) 마스킹
                    fc1.weight.data.mul_(mask.view(-1, 1))
                    if fc1.bias is not None:
                        fc1.bias.data.mul_(mask)

                    # fc2: 열(입력 뉴런) 마스킹
                    fc2.weight.data.mul_(mask.view(1, -1))
                    # fc2 bias는 출력 방향 (384) 건드리지 않음

                elif g['type'] == 'attn':
                    qkv  = all_modules.get(g['names'][0])
                    proj = all_modules.get(g['names'][1])
                    if qkv is None or proj is None:
                        continue

                    num_heads = g['num_heads']
                    head_dim  = g['head_dim']
                    mask = qkv.mask  # [num_heads]

                    # qkv: Q/K/V 각 슬라이스의 해당 head 행 마스킹
                    for h in range(num_heads):
                        if mask[h].item() < 0.5:
                            for offset in range(3):  # Q, K, V
                                start = offset * (num_heads * head_dim) + h * head_dim
                                end   = start + head_dim
                                qkv.weight.data[start:end, :] = 0.0
                            if qkv.bias is not None:
                                for offset in range(3):
                                    start = offset * (num_heads * head_dim) + h * head_dim
                                    end   = start + head_dim
                                    qkv.bias.data[start:end] = 0.0

                            # proj: 해당 head의 입력 열 마스킹
                            start = h * head_dim
                            end   = start + head_dim
                            proj.weight.data[:, start:end] = 0.0

    def update_ema_and_mask_grad(self):
        """
        타입별 EMA 업데이트.
        - ffn:  fc1의 뉴런별 grad 에너지
        - attn: qkv의 head별 grad 에너지 (Q슬라이스 기준)
        """
        if not self.topology_groups:
            return

        all_modules = dict(self.model.named_modules())

        with torch.no_grad():
            for g in self.topology_groups:

                if g['type'] == 'ffn':
                    fc1 = all_modules.get(g['names'][0])
                    if fc1 is None or fc1.weight.grad is None:
                        continue
                    # grad 마스킹 후 EMA
                    fc1.weight.grad.data.mul_(fc1.mask.view(-1, 1))
                    g_energy = fc1.weight.grad.pow(2).mean(dim=1)  # [1536]
                    fc1.grad_ema.mul_(self.ema_decay).add_(
                        g_energy, alpha=1 - self.ema_decay)

                elif g['type'] == 'attn':
                    qkv = all_modules.get(g['names'][0])
                    if qkv is None or qkv.weight.grad is None:
                        continue
                    num_heads = g['num_heads']
                    head_dim  = g['head_dim']

                    # head별 grad 에너지 (Q/K/V 세 슬라이스 평균)
                    g_energy = torch.zeros(num_heads, device=qkv.weight.device)
                    for h in range(num_heads):
                        slice_energies = []
                        for offset in range(3):  # Q, K, V
                            start = offset * (num_heads * head_dim) + h * head_dim
                            end   = start + head_dim
                            slice_energies.append(
                                qkv.weight.grad[start:end, :].pow(2).mean()
                            )
                        g_energy[h] = torch.stack(slice_energies).mean()

                    qkv.grad_ema.mul_(self.ema_decay).add_(
                        g_energy, alpha=1 - self.ema_decay)

    def get_model_efficiency(self, example_inputs=None):
        """Linear 기준 efficiency 계산"""
        total_params, remaining_params = 0, 0
        all_modules = dict(self.model.named_modules())

        for g in (self.topology_groups or []):
            for name in g['names']:
                m = all_modules.get(name)
                if m is None:
                    continue
                n_p = m.weight.numel()
                total_params += n_p
                if hasattr(m, 'mask'):
                    keep = m.mask.sum().item() / (m.mask.numel() + 1e-8)
                    remaining_params += n_p * keep
                else:
                    remaining_params += n_p

        orig_mb = (total_params * 4) / (1024 ** 2)
        curr_mb = (remaining_params * 4) / (1024 ** 2)
        sparsity = (1 - remaining_params / (total_params + 1e-8)) * 100
        speedup  = total_params / (remaining_params + 1e-8)

        return {'orig_mb': orig_mb, 'curr_mb': curr_mb,
                'sparsity': sparsity, 'speedup': speedup}

    def get_true_sparsity(self):
        """논문 기준 실제 weight sparsity 계산"""
        total = 0
        zero = 0
        all_modules = dict(self.model.named_modules())
        for g in (self.topology_groups or []):
            for name in g['names']:
                m = all_modules.get(name)
                if m is None:
                    continue
                total += m.weight.numel()
                zero += (m.weight.data == 0).sum().item()
        return (zero / total) * 100.0 if total > 0 else 0.0

    def step_pruning(self, loss, current_epoch, total_epochs):
        t_start = time.time()
        print(f"\n[DEBUG] === ViT PDT Pruning (k-horizon={self.k_horizon}): Epoch {current_epoch} ===")

        all_modules = dict(self.model.named_modules())

        if not self.topology_groups:
            print("[WARNING] No ViT topology groups.")
            return

        progress = current_epoch / total_epochs
        target_remaining_ratio = 1.0 - (progress * self.pruning_ratio)

        # topology_groups의 대표 레이어 순서 리스트
        # blocks.0.attn.qkv → blocks.0.mlp.fc1 → blocks.1.attn.qkv → ...
        rep_layer_names = [g['names'][0] for g in self.topology_groups]

        # ────────────────────────────────────────────────
        # k-horizon sensitivity 계산
        # s(g,c,K) = Σᵢ₌₀ᴷ | g(l+i,c)ᵀΔw + ½ΔwᵀH(l+i,c)Δw |
        # ────────────────────────────────────────────────
        t_khorizon_start = time.time()
        import gc
        group_k_sensitivity = {}  # {g_idx: tensor [n_units]}

        for g_idx, g in enumerate(self.topology_groups):
            rep = all_modules.get(g['names'][0])
            if rep is None or not hasattr(rep, 'weight'):
                continue

            n_units = rep.mask.size(0)  # ffn: 1536, attn: 6
            cumulative = torch.zeros(n_units, device=rep.weight.device)

            for step in range(self.k_horizon):
                look_idx = g_idx + step  # 현재 그룹에서 k step 앞
                if look_idx >= len(self.topology_groups):
                    break

                look_g    = self.topology_groups[look_idx]
                look_name = look_g['names'][0]
                look_m    = all_modules.get(look_name)
                if look_m is None or not hasattr(look_m, 'weight'):
                    continue

                try:
                    # 1차 gradient g(l+i,c)
                    grad = torch.autograd.grad(
                        loss, look_m.weight,
                        create_graph=True, retain_graph=True
                    )[0]

                    # Δw(g,c): 채널 c 제거 시 가중치 변화 = -현재 가중치
                    delta_w = -look_m.weight.detach()

                    # ── 1차 항: g(l+i)ᵀ · Δw ──────────────────────
                    if g['type'] == 'ffn':
                        # [out_features] 단위
                        grad_term = (grad * delta_w).reshape(
                            look_m.weight.shape[0], -1).sum(1).abs()

                    elif g['type'] == 'attn':
                        num_heads = g['num_heads']
                        head_dim  = g['head_dim']
                        grad_term = torch.zeros(num_heads, device=grad.device)
                        if look_g['type'] == 'attn':
                            # qkv: Q 슬라이스 기준
                            for h in range(num_heads):
                                s, e = h * head_dim, (h + 1) * head_dim
                                if e <= grad.shape[0]:
                                    grad_term[h] = (grad[s:e] * delta_w[s:e]).sum().abs()
                        else:
                            # look 레이어가 ffn 타입이면 head 단위로 묶어서 평균
                            chunk = grad.reshape(num_heads, -1)
                            dw_chunk = delta_w.reshape(num_heads, -1)
                            grad_term = (chunk * dw_chunk).sum(1).abs()

                    # ── 2차 항: ½ · Δw ᵀ · H(l+i) · Δw ───────────
                    v  = torch.randn_like(grad)
                    hv = torch.autograd.grad(
                        (grad * v).sum(), look_m.weight,
                        retain_graph=True
                    )[0]

                    if g['type'] == 'ffn':
                        hess_term = (0.5 * delta_w * hv).reshape(
                            look_m.weight.shape[0], -1).sum(1).abs()

                    elif g['type'] == 'attn':
                        hess_term = torch.zeros(num_heads, device=hv.device)
                        if look_g['type'] == 'attn':
                            for h in range(num_heads):
                                s, e = h * head_dim, (h + 1) * head_dim
                                if e <= hv.shape[0]:
                                    hess_term[h] = (0.5 * delta_w[s:e] * hv[s:e]).sum().abs()
                        else:
                            hv_chunk  = hv.reshape(num_heads, -1)
                            dw_chunk  = delta_w.reshape(num_heads, -1)
                            hess_term = (0.5 * dw_chunk * hv_chunk).sum(1).abs()

                    # ── 누적: Σᵢ₌₀ᴷ ─────────────────────────────
                    step_score = grad_term + hess_term
                    # 크기 맞추기 (look 레이어와 현재 그룹 단위가 다를 수 있음)
                    if step_score.size(0) != n_units:
                        step_score = step_score[:n_units] if step_score.size(0) > n_units \
                                    else torch.nn.functional.pad(step_score, (0, n_units - step_score.size(0)))

                    cumulative += step_score.detach()

                    del grad, hv, v, delta_w
                    if step % 3 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()

                except RuntimeError as e:
                    print(f"[WARNING] k-horizon step {step} skipped ({look_name}): {e}")
                    break

            # 정규화 후 hessian_score 버퍼에 저장
            avg_sensitivity = cumulative / max(self.k_horizon, 1)
            if avg_sensitivity.max() > avg_sensitivity.min():
                avg_sensitivity = (avg_sensitivity - avg_sensitivity.min()) / \
                                (avg_sensitivity.max() - avg_sensitivity.min() + 1e-8)

            rep.hessian_score.copy_(avg_sensitivity)
            group_k_sensitivity[g_idx] = avg_sensitivity
        t_khorizon_end = time.time()
        print(f" [k-horizon time]: {t_khorizon_end - t_khorizon_start:.2f}s")


        # ────────────────────────────────────────────────
        # 점수 산출: S(g,c,K) = Ŵ_g^(EMA) × s(g,c,K)
        # ────────────────────────────────────────────────
        target_unit_scores   = []
        target_unit_costs    = []
        target_unit_metadata = []

        for g_idx, g in enumerate(self.topology_groups):
            rep = all_modules.get(g['names'][0])
            if rep is None or not hasattr(rep, 'mask'):
                continue

            alive = torch.where(rep.mask > 0.5)[0].cpu().numpy()

            for i in alive:
                # S(g,c,K) = grad_ema(그룹 rank) × hessian_score(채널 score) × λ
                score = rep.grad_ema[i].item() * (
                    rep.hessian_score[i].item() * self.lambda_h)

                if g['type'] == 'ffn':
                    cost = sum(
                        all_modules[n].weight.nelement() / all_modules[n].weight.shape[0]
                        for n in g['names'] if n in all_modules)
                else:
                    head_dim = g['head_dim']
                    cost = head_dim * (
                        all_modules[g['names'][0]].weight.shape[1] * 3 +
                        all_modules[g['names'][1]].weight.shape[0])

                target_unit_scores.append(score)
                target_unit_costs.append(cost)
                target_unit_metadata.append((g, int(i)))

        # ────────────────────────────────────────────────
        # Lagrangian 최적화 및 마스크 집행 (기존과 동일)
        # ────────────────────────────────────────────────
        if not target_unit_scores:
            return

        current_alive_ratio = 1.0 - (self.get_current_sparsity() / 100.0)
        target_ratio = target_remaining_ratio / (current_alive_ratio + 1e-7)

        optimal_mask_flags = lagrangian_optimization(
            np.array(target_unit_scores),
            np.array(target_unit_costs),
            np.sum(target_unit_costs) * target_ratio
        )

        if np.all(optimal_mask_flags == 1) and target_ratio < 0.999:
            num_force = int(len(optimal_mask_flags) * (1 - target_ratio))
            optimal_mask_flags[np.argsort(target_unit_scores)[:num_force]] = 0

        pruned_count = 0
        with torch.no_grad():
            for idx, is_alive in enumerate(optimal_mask_flags):
                if not is_alive:
                    g_obj, unit_idx = target_unit_metadata[idx]
                    for name in g_obj['names']:
                        m = all_modules.get(name)
                        if m is not None and hasattr(m, 'mask') and unit_idx < m.mask.size(0):
                            m.mask.data[unit_idx] = 0.0
                            pruned_count += 1

        self.apply_mask_to_weights()
        torch.cuda.empty_cache()
        t_end = time.time()

        # 리포트
        eff = self.get_model_efficiency()
        print(f"\n{'='*30} ViT PDT k-horizon Pruning Report {'='*30}")
        # print(f" k_horizon={self.k_horizon} | Pruned: {pruned_count} units")
        print(f" k_horizon={self.k_horizon} | Pruned: {pruned_count} units | Overhead: {t_end - t_start:.2f}s")
        for g in self.topology_groups:
            m = all_modules.get(g['names'][0])
            if m is None or not hasattr(m, 'mask'):
                continue
            t = m.mask.numel()
            a = int(m.mask.sum().item())
            print(f" [{g['type'].upper()}] {g['names'][0]:40s} | {a:4d}/{t:4d} | {(1-a/t)*100:5.1f}% pruned")
        # print(f"\n Size: {eff['curr_mb']:.2f}MB | Sparsity: {eff['sparsity']:.2f}% | Speedup: {eff['speedup']:.2f}x")
        true_sp = self.get_true_sparsity()
        print(f"\n Size: {eff['curr_mb']:.2f}MB | Sparsity(weight): {true_sp:.2f}% | Speedup: {eff['speedup']:.2f}x")
        print('='*67)


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
        total_target_sparsity = min(progress * 2.0, 1.0) * self.pruning_ratio

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
        total_param_cost = sum(m.weight.numel() for _, m in target_layers)
        target_pruned_cost = total_param_cost * total_target_sparsity
        current_pruned_cost = sum(
            m.weight.numel() * (1.0 - m.mask.float().mean().item())
            for _, m in target_layers
        )
        restored_units = 0
        added_units = 0

        with torch.no_grad():
            if current_pruned_cost > target_pruned_cost:
                pruned_candidates = []
                for _, layer in target_layers:
                    unit_cost = layer.weight.numel() / layer.weight.shape[0]
                    for channel_idx in torch.where(layer.mask <= 0.5)[0].tolist():
                        pruned_candidates.append(
                            (layer.hessian_score[channel_idx].item(), layer, channel_idx, unit_cost)
                        )
                for _, layer, channel_idx, unit_cost in sorted(
                    pruned_candidates, key=lambda item: item[0], reverse=True
                ):
                    if current_pruned_cost <= target_pruned_cost:
                        break
                    layer.mask[channel_idx] = 1.0
                    current_pruned_cost -= unit_cost
                    actual_pruned_total -= 1
                    restored_units += 1
            elif current_pruned_cost < target_pruned_cost:
                alive_candidates = []
                for _, layer in target_layers:
                    unit_cost = layer.weight.numel() / layer.weight.shape[0]
                    for channel_idx in torch.where(layer.mask > 0.5)[0].tolist():
                        alive_candidates.append(
                            (layer.hessian_score[channel_idx].item(), layer, channel_idx, unit_cost)
                        )
                for _, layer, channel_idx, unit_cost in sorted(
                    alive_candidates, key=lambda item: item[0]
                ):
                    if current_pruned_cost >= target_pruned_cost:
                        break
                    layer.mask[channel_idx] = 0.0
                    current_pruned_cost += unit_cost
                    actual_pruned_total += 1
                    added_units += 1

        print(
            f"[Ratio Repair] restored_units={restored_units} "
            f"additional_units={added_units} "
            f"target_param_cost={target_pruned_cost:.0f} "
            f"actual_param_cost={current_pruned_cost:.0f}"
        )
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
