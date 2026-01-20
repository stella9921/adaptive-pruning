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
        # 1. Hessian-Vector Product 추출
        _, hv_list = self.engine.get_smart_direction_p(loss, self.model)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # 2. 레이어별 Hessian 점수 업데이트
        with torch.no_grad():
            for param, hv in zip(trainable_params, hv_list):
                for m in self.layers:
                    if m.weight is param:
                        h_energy = hv.pow(2).reshape(hv.shape[0], -1).mean(1)
                        m.hessian_score.copy_(h_energy)

        # 3. 그룹 단위 스코어 합산
        group_scores = []
        group_mem_costs = []
        group_names = []
        all_modules = dict(self.model.named_modules())

        if self.topology_groups and len(self.topology_groups) > 0:
            for group in self.topology_groups:
                g_score = 0
                g_mem = 0
                for layer_name in group:
                    m = all_modules.get(layer_name)
                    if isinstance(m, nn.Conv2d):
                        combined = m.grad_ema + (self.lambda_h * m.hessian_score)
                        g_score += combined.mean().item()
                        g_mem += m.weight.nelement()
                
                if g_mem > 0: # 유효한 레이어가 있는 그룹만 추가
                    group_scores.append(g_score)
                    group_mem_costs.append(g_mem)
                    group_names.append(group)
        else:
            for name, m in self.layers_dict.items():
                combined = m.grad_ema + (self.lambda_h * m.hessian_score)
                group_scores.append(combined.mean().item())
                group_mem_costs.append(m.weight.nelement())
                group_names.append([name.replace('_', '.')])

        # 4. [Stage 3] Lagrangian Optimization
        if memory_budget is None:
            total_mem = sum(group_mem_costs)
            # target_ratio가 0.85라면 85%를 삭제하고 15%만 남기는 예산으로 설정
            target_pruning_ratio = self.config['strategy'].get('target_ratio', 0.85)
            memory_budget = total_mem * (1.0 - target_pruning_ratio)
            
            # 버그 방지: 예산이 너무 적으면 최소 1개 그룹은 살리도록 보정
            if memory_budget <= 0:
                memory_budget = total_mem * 0.05 

        print(f"[DEBUG] Total Mem: {total_mem/1e6:.2f}M | Target Budget: {memory_budget/1e6:.2f}M")

        # 최적의 생존 그룹 결정
        optimal_mask_flags = lagrangian_optimization(
            np.array(group_scores), 
            np.array(group_mem_costs), 
            memory_budget
        )

        # 안전장치: 모든 그룹이 죽었을 경우 점수가 가장 높은 그룹 하나는 강제로 살림
        if sum(optimal_mask_flags) == 0:
            print("[Warning] All groups killed by optimizer. Keeping top-1 group as fallback.")
            optimal_mask_flags[np.argmax(group_scores)] = True

        # 5. 마스크 실제 적용
        with torch.no_grad():
            for idx, is_alive in enumerate(optimal_mask_flags):
                target_group = group_names[idx]
                for layer_name in target_group:
                    m = all_modules.get(layer_name)
                    if isinstance(m, nn.Conv2d):
                        val = 1.0 if is_alive else 0.0
                        m.mask.fill_(val)
        
        print(f"[PDT] Optimization Complete. Groups Alive: {sum(optimal_mask_flags)}/{len(group_names)}")

    @torch.no_grad()
    def get_current_sparsity(self):
        total_params = sum(m.weight.nelement() for m in self.layers)
        active_params = sum(m.mask.sum().item() * (m.weight.nelement() / m.weight.shape[0]) for m in self.layers)
        if total_params == 0: return 0.0
        return (1 - active_params / total_params) * 100