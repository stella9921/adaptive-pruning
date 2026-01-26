import torch
import torch.nn as nn
from .base import BasePruner
from .engine.hessian_free import SNOWSEngine
import numpy as np
from .optimizer import lagrangian_optimization 

class PDTPruner(BasePruner):
    def __init__(self, model, config, topology_groups=None):
        super().__init__(model, config)
        self.ema_decay = config['strategy'].get('ema_decay', 0.9)
        self.lambda_h = config['strategy'].get('lambda_h', 0.5) 
        self.engine = SNOWSEngine(n_iter=config['strategy'].get('hessian_iter', 5))
        
        # [Stage 1] Topology Manager로부터 받은 그룹 정보 (연계 관계 지도)
        self.topology_groups = topology_groups
        
        self.layers_dict = nn.ModuleDict()
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
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
        with torch.no_grad():
            for m in self.layers:
                if hasattr(m, 'weight') and m.weight.grad is not None:
                    # 마스크가 적용된 그래디언트만 추적
                    m.weight.grad.mul_(m.mask.view(-1, 1, 1, 1))
                    g = m.weight.grad.pow(2).reshape(m.weight.shape[0], -1).mean(1)
                    m.grad_ema.mul_(self.ema_decay).add_(g, alpha=1 - self.ema_decay)

    def apply_mask_to_weights(self, optimizer=None):
        with torch.no_grad():
            for m in self.layers:
                mask = m.mask
                m.weight.data.mul_(mask.view(-1, 1, 1, 1))
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.mul_(mask)
                if optimizer is not None:
                    for p in [m.weight, m.bias]:
                        if p is not None and p in optimizer.state:
                            state = optimizer.state[p]
                            if "momentum_buffer" in state:
                                m_view = mask.view(-1, 1, 1, 1) if p.dim() == 4 else mask
                                state["momentum_buffer"].mul_(m_view)

    def step_pruning(self, loss, memory_budget=None):
        """
        [Topology-Aware Lagrangian Optimization - Accuracy Guard Version]
        정확도(Hessian), 메모리(Cost), 토폴로지(FX Group) 통합 최적화
        """
        # 1. Stage 2: Hessian-Vector Product 추출
        _, hv_list = self.engine.get_smart_direction_p(loss, self.model)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        with torch.no_grad():
            for param, hv in zip(trainable_params, hv_list):
                for m in self.layers:
                    if m.weight is param:
                        h_energy = hv.pow(2).reshape(hv.shape[0], -1).mean(1)
                        # [FIX] 레이어별 Hessian 정규화: 특정 레이어 몰살 방지
                        if h_energy.max() > 0:
                            h_energy = h_energy / (h_energy.max() + 1e-8)
                        m.hessian_score.copy_(h_energy)

        # 2. Stage 1 & 3 연계: 토폴로지 그룹 단위 데이터 합산
        all_unit_scores = []
        all_unit_costs = []
        unit_metadata = [] 
        all_modules = dict(self.model.named_modules())

        def find_layer(name):
            if name in all_modules: return all_modules[name]
            dot_name = name.replace('_', '.')
            return all_modules.get(dot_name)

        processed_layers = set()
        
        # [Topology Awareness] FX 그룹 처리
        if self.topology_groups:
            for group in self.topology_groups:
                layers_in_group = [find_layer(ln) for ln in group if isinstance(find_layer(ln), nn.Conv2d)]
                for m in layers_in_group: processed_layers.add(m)
                
                if layers_in_group:
                    num_channels = layers_in_group[0].weight.shape[0]
                    # 레이어 내 상대적 중요도 계산
                    for i in range(num_channels):
                        g_score = sum((m.grad_ema[i] + self.lambda_h * m.hessian_score[i]).item() for m in layers_in_group)
                        g_cost = sum(m.weight.nelement() / m.weight.shape[0] for m in layers_in_group)
                        
                        all_unit_scores.append(g_score)
                        all_unit_costs.append(g_cost)
                        unit_metadata.append((layers_in_group, i))

        # 독립 레이어 처리
        for m in self.layers:
            if m not in processed_layers:
                num_channels = m.weight.shape[0]
                unit_cost = m.weight.nelement() / num_channels
                combined_scores = m.grad_ema + (self.lambda_h * m.hessian_score)
                for i in range(num_channels):
                    all_unit_scores.append(combined_scores[i].item())
                    all_unit_costs.append(unit_cost)
                    unit_metadata.append(([m], i))

        # 3. Stage 3: Lagrangian Optimization 실행
        scores_np = np.array(all_unit_scores)
        costs_np = np.array(all_unit_costs)

        if memory_budget is None:
            target_ratio = self.config['strategy'].get('target_ratio', 0.6)
            memory_budget = np.sum(costs_np) * (1.0 - target_ratio)

        # 라그랑주 엔진 호출
        optimal_mask_flags = lagrangian_optimization(scores_np, costs_np, memory_budget)

        # 4. 결과 적용 및 정확도 방어 (Connectivity 보존)
        with torch.no_grad():
            for m in self.layers:
                m.mask.fill_(0.0)
            
            # [FIX] 각 유닛별로 강제 생존 조건 체크
            # 점수가 상위 10% 안에 들면 optimal_mask_flags와 상관없이 살리는 로직을 적용할 수도 있음
            active_count = 0
            for idx, is_alive in enumerate(optimal_mask_flags):
                if is_alive:
                    layers_list, channel_idx = unit_metadata[idx]
                    for layer_obj in layers_list:
                        layer_obj.mask[channel_idx] = 1.0
                    active_count += 1

        print(f"[PDT] Pruning Done. Active Topology Units: {active_count}/{len(all_unit_scores)}")

    @torch.no_grad()
    def get_current_sparsity(self):
        total_params = sum(m.weight.nelement() for m in self.layers)
        active_params = sum(m.mask.sum().item() * (m.weight.nelement() / m.weight.shape[0]) for m in self.layers)
        return (1 - active_params / total_params) * 100 if total_params > 0 else 0.0