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
        [Topology-Aware Lagrangian Optimization]
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
                        m.hessian_score.copy_(h_energy)

        # 2. Stage 1 & 3 연계: 토폴로지 그룹 단위 데이터 합산
        all_unit_scores = []
        all_unit_costs = []
        unit_metadata = [] # (레이어 리스트, 채널 인덱스)
        all_modules = dict(self.model.named_modules())

        def find_layer(name):
            if name in all_modules: return all_modules[name]
            dot_name = name.replace('_', '.')
            return all_modules.get(dot_name)

        processed_layers = set()
        
        # [Topology Awareness] FX 그룹 처리 (운명 공동체 채널들)
        if self.topology_groups:
            for group in self.topology_groups:
                layers_in_group = [find_layer(ln) for ln in group if isinstance(find_layer(ln), nn.Conv2d)]
                for m in layers_in_group: processed_layers.add(m)
                
                if layers_in_group:
                    # 그룹 내 모든 레이어는 동일한 출력 채널 수를 가짐 (Add 연산 제약)
                    num_channels = layers_in_group[0].weight.shape[0]
                    for i in range(num_channels):
                        # 목적함수의 정확도 항: 그룹 내 모든 연계 채널의 Hessian+Grad 점수 합산
                        g_score = sum((m.grad_ema[i] + self.lambda_h * m.hessian_score[i]).item() for m in layers_in_group)
                        # 제약조건의 비용 항: 그룹 내 모든 연계 채널의 파라미터 비용 합산
                        g_cost = sum(m.weight.nelement() / m.weight.shape[0] for m in layers_in_group)
                        
                        all_unit_scores.append(g_score)
                        all_unit_costs.append(g_cost)
                        unit_metadata.append((layers_in_group, i))

        # 독립 레이어 처리 (연계 관계가 없는 레이어)
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

        # 라그랑주 엔진 호출 (이제 그룹 단위의 데이터를 처리함)
        optimal_mask_flags = lagrangian_optimization(scores_np, costs_np, memory_budget)

        # 4. 결과 적용: 의존성 그룹 내 모든 레이어에 마스크 전파
        with torch.no_grad():
            for m in self.layers:
                m.mask.fill_(0.0)
            
            active_count = 0
            for idx, is_alive in enumerate(optimal_mask_flags):
                if is_alive:
                    layers_list, channel_idx = unit_metadata[idx]
                    for layer_obj in layers_list:
                        layer_obj.mask[channel_idx] = 1.0 # 그룹 내 모든 레이어의 동일 채널 활성화
                    active_count += 1

        print(f"[PDT] Pruning Done. Active Topology Units: {active_count}/{len(all_unit_scores)}")

    @torch.no_grad()
    def get_current_sparsity(self):
        total_params = sum(m.weight.nelement() for m in self.layers)
        active_params = sum(m.mask.sum().item() * (m.weight.nelement() / m.weight.shape[0]) for m in self.layers)
        return (1 - active_params / total_params) * 100 if total_params > 0 else 0.0