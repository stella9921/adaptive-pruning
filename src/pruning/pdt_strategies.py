import torch
import torch.nn as nn
from .base import BasePruner
from .engine.hessian_free import SNOWSEngine
import numpy as np
from .optimizer import lagrangian_optimization 

class PDTPruner(BasePruner):
    def __init__(self, model, config, topology_groups=None):
        super().__init__(model, config)
        self.ema_decay = config['strategy'].get('ema_decay', 0.95)
        self.lambda_h = config['strategy'].get('lambda_h', 0.5) 
        self.engine = SNOWSEngine(n_iter=config['strategy'].get('hessian_iter', 5))
        
        self.topology_groups = topology_groups
        self.layers_dict = nn.ModuleDict()
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                self.layers_dict[name.replace('.', '_')] = m
        
        self.layers = list(self.layers_dict.values())
        self._check_buffers()

    def _check_buffers(self):
        for m in self.layers:
            n_f = m.weight.shape[0]
            if not hasattr(m, 'mask'):
                m.register_buffer("mask", torch.ones(n_f, device=self.device))
            if not hasattr(m, 'grad_ema'):
                m.register_buffer("grad_ema", torch.zeros(n_f, device=self.device))
            if not hasattr(m, 'hessian_score'):
                m.register_buffer("hessian_score", torch.zeros(n_f, device=self.device))

    def update_ema_and_mask_grad(self):
        """
        [수식 반영] W_g^(EMA)의 기초 데이터인 레이어별 그래디언트 EMA 업데이트
        """
        with torch.no_grad():
            for m in self.layers:
                if hasattr(m, 'weight') and m.weight.grad is not None:
                    m.weight.grad.mul_(m.mask.view(-1, 1, 1, 1) if m.weight.dim()==4 else m.mask.view(-1, 1))
                    g = m.weight.grad.pow(2).reshape(m.weight.shape[0], -1).mean(1)
                    m.grad_ema.mul_(self.ema_decay).add_(g, alpha=1 - self.ema_decay)

    def apply_mask_to_weights(self, optimizer=None):
        with torch.no_grad():
            for m in self.layers:
                mask = m.mask
                m_view = mask.view(-1, 1, 1, 1) if m.weight.dim() == 4 else mask.view(-1, 1)
                m.weight.data.mul_(m_view)
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.mul_(mask)
                if optimizer is not None:
                    for p in [m.weight, m.bias]:
                        if p is not None and p in optimizer.state:
                            state = optimizer.state[p]
                            if "momentum_buffer" in state:
                                p_view = mask.view(-1, 1, 1, 1) if p.dim() == 4 else mask.view(-1, 1)
                                state["momentum_buffer"].mul_(p_view)

    def step_pruning(self, loss, memory_budget=None):
        """
        [Conditional Hierarchical Optimization]
        1. 그룹별 위상(EMA)을 먼저 구함
        2. 위상이 낮은 하위 그룹만 헤시안(Hessian)으로 솎아냄
        """
        all_modules = dict(self.model.named_modules())
        def find_layer(name):
            if name in all_modules: return all_modules[name]
            return all_modules.get(name.replace('_', '.'))

        # 1. 모든 그룹의 EMA 위상(W_g) 정보 수집
        group_info_list = []
        processed_layers = set()

        if self.topology_groups:
            for group in self.topology_groups:
                layers = [find_layer(ln) for ln in group if isinstance(find_layer(ln), (nn.Conv2d, nn.Linear))]
                if not layers: continue
                for m in layers: processed_layers.add(m)
                
                # 그룹 위상 산출 (평균 EMA)
                w_g = torch.mean(torch.stack([m.grad_ema.mean() for m in layers])).item()
                group_info_list.append({'layers': layers, 'w_g': w_g, 'is_group': True})

        # 독립 레이어들도 그룹으로 취급하여 추가
        for m in self.layers:
            if m not in processed_layers:
                group_info_list.append({'layers': [m], 'w_g': m.grad_ema.mean().item(), 'is_group': False})

        # 2. 위상 기준 하위 그룹 선별 (Hessian 대상)
        group_info_list.sort(key=lambda x: x['w_g'])
        target_ratio = self.config['strategy'].get('hessian_target_ratio', 0.5) # 하위 50%
        num_targets = int(len(group_info_list) * target_ratio)
        
        target_groups = group_info_list[:num_targets]
        protected_groups = group_info_list[num_targets:]

        # 3. 선별된 타겟 그룹 파라미터만 헤시안 계산
        target_params = []
        for g in target_groups:
            for m in g['layers']:
                target_params.append(m.weight)

        K = self.config['strategy'].get('k_horizon', 10)
        # 엔진에서 특정 파라미터만 HVP 수행 (VRAM 대폭 절약)
        hv_list = self.engine.get_k_step_hessian_selective(loss, target_params, K)

        # 4. 최종 유닛별 점수(Score)와 비용(Cost) 구성
        all_unit_scores = []
        all_unit_costs = []
        unit_metadata = []

        # [Case A] 타겟 그룹: W_g * s_gc (Hessian 반영)
        hv_idx = 0
        for g in target_groups:
            for m in g['layers']:
                hv = hv_list[hv_idx]
                h_energy = hv.pow(2).reshape(hv.shape[0], -1).mean(1)
                if h_energy.max() > 0:
                    h_energy /= (h_energy.max() + 1e-8)
                m.hessian_score.copy_(h_energy)
                hv_idx += 1

            num_channels = g['layers'][0].weight.shape[0]
            for i in range(num_channels):
                s_gc = sum(m.hessian_score[i].item() for m in g['layers'])
                all_unit_scores.append(g['w_g'] * s_gc) # 계층적 곱셈
                all_unit_costs.append(sum(m.weight.nelement()/m.weight.shape[0] for m in g['layers']))
                unit_metadata.append((g['layers'], i))

        # [Case B] 보호 그룹: 헤시안 생략, 높은 보존 점수 부여
        for g in protected_groups:
            num_channels = g['layers'][0].weight.shape[0]
            for i in range(num_channels):
                # 헤시안 검사 없이 위상 점수로만 생존권 보장 (고정 가중치 1.0 적용)
                all_unit_scores.append(g['w_g'] * 1.0) 
                all_unit_costs.append(sum(m.weight.nelement()/m.weight.shape[0] for m in g['layers']))
                unit_metadata.append((g['layers'], i))

        # 5. Lagrangian Optimization (m* = argmax V(m))
        scores_np = np.array(all_unit_scores)
        costs_np = np.array(all_unit_costs)

        if memory_budget is None:
            target_ratio = self.config['strategy'].get('target_ratio', 0.6)
            memory_budget = np.sum(costs_np) * (1.0 - target_ratio)

        optimal_mask_flags = lagrangian_optimization(scores_np, costs_np, memory_budget)

        # 6. 마스크 적용
        with torch.no_grad():
            for m in self.layers:
                m.mask.fill_(0.0)
            
            active_count = 0
            for idx, is_alive in enumerate(optimal_mask_flags):
                if is_alive:
                    layers_list, channel_idx = unit_metadata[idx]
                    for layer_obj in layers_list:
                        layer_obj.mask[channel_idx] = 1.0
                    active_count += 1

        print(f"[PDT] Conditional Pruning Done. Target Groups: {num_targets}, Active Units: {active_count}/{len(all_unit_scores)}")
        torch.cuda.empty_cache()