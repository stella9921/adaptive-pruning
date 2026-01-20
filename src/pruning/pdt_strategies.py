import torch
import torch.nn as nn
from .base import BasePruner
# Hessian 엔진 및 최적화 도구 임포트
from .engine.hessian_free import SNOWSEngine
import numpy as np
from .optimizer import lagrangian_optimization  # Stage 3 최적화 로직

class PDTPruner(BasePruner):
    def __init__(self, model, config, topology_groups=None):
        super().__init__(model, config)
        self.ema_decay = config['strategy'].get('ema_decay', 0.9)
        self.lambda_h = config['strategy'].get('lambda_h', 0.5) 
        self.engine = SNOWSEngine(n_iter=config['strategy'].get('hessian_iter', 5))
        
        # [Stage 1] Topology Manager로부터 받은 그룹 정보 저장
        self.topology_groups = topology_groups
        
        # 프루닝 대상 레이어 리스트업
        self.layers_dict = nn.ModuleDict()
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                self.layers_dict[name.replace('.', '_')] = m
        
        self.layers = list(self.layers_dict.values())
        self._check_buffers()

    def _check_buffers(self):
        """각 레이어에 필요한 연산용 버퍼 등록"""
        for m in self.layers:
            n_f = m.weight.shape[0]
            if not hasattr(m, 'mask'):
                m.register_buffer("mask", torch.ones(n_f, device=self.device))
            if not hasattr(m, 'grad_ema'):
                m.register_buffer("grad_ema", torch.zeros(n_f, device=self.device))
            if not hasattr(m, 'hessian_score'):
                m.register_buffer("hessian_score", torch.zeros(n_f, device=self.device))

    def update_ema_and_mask_grad(self):
        """매 배치마다 Gradient EMA 업데이트"""
        with torch.no_grad():
            for m in self.layers:
                if hasattr(m, 'weight') and m.weight.grad is not None:
                    m.weight.grad.mul_(m.mask.view(-1, 1, 1, 1))
                    g = m.weight.grad.pow(2).reshape(m.weight.shape[0], -1).mean(1)
                    m.grad_ema.mul_(self.ema_decay).add_(g, alpha=1 - self.ema_decay)

    def apply_mask_to_weights(self, optimizer=None):
        """가중치와 옵티마이저 모멘텀에 마스크 적용"""
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
        """[Stage 2 & 3] Hessian 기반 그룹 스코어링 및 Lagrangian 최적화"""
        # 1. Hessian-Vector Product 추출 (SNOWS)
        _, hv_list = self.engine.get_smart_direction_p(loss, self.model)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # 2. [Stage 2] 레이어별 Hessian 점수 업데이트
        with torch.no_grad():
            for param, hv in zip(trainable_params, hv_list):
                for m in self.layers:
                    if m.weight is param:
                        h_energy = hv.pow(2).reshape(hv.shape[0], -1).mean(1)
                        m.hessian_score.copy_(h_energy)

        # 3. [Stage 1 & 2] 그룹 단위 스코어 합산 (Channel Grouping 반영)
        group_scores = []
        group_mem_costs = []
        group_names = []

        if self.topology_groups:
            # FX로 분석된 그룹이 있을 경우 그룹 단위 처리
            for group in self.topology_groups:
                g_score = 0
                g_mem = 0
                for layer_name in group:
                    # 레이어 이름 매칭 (topology_manager의 출력 형식에 맞춤)
                    m = dict(self.model.named_modules()).get(layer_name)
                    if isinstance(m, nn.Conv2d):
                        combined = m.grad_ema + (self.lambda_h * m.hessian_score)
                        g_score += combined.mean().item() # 그룹 평균 점수
                        g_mem += m.weight.nelement()     # 그룹 메모리 비용
                
                group_scores.append(g_score)
                group_mem_costs.append(g_mem)
                group_names.append(group)
        else:
            # 그룹 정보가 없을 경우 개별 레이어 처리 (Fallback)
            for name, m in self.layers_dict.items():
                combined = m.grad_ema + (self.lambda_h * m.hessian_score)
                group_scores.append(combined.mean().item())
                group_mem_costs.append(m.weight.nelement())
                group_names.append([name])

        # 4. [Stage 3] Lagrangian Optimization으로 최적 마스크 선별
        # budget이 없으면 설정파일의 target_ratio 기준으로 계산
        if memory_budget is None:
            total_mem = sum(group_mem_costs)
            target_ratio = self.config['strategy'].get('target_ratio', 0.4)
            memory_budget = total_mem * (1 - target_ratio)

        # 최적의 생존 그룹 결정
        optimal_mask_flags = lagrangian_optimization(
            np.array(group_scores), 
            np.array(group_mem_costs), 
            memory_budget
        )

        # 5. 마스크 실제 적용
        with torch.no_grad():
            for idx, is_alive in enumerate(optimal_mask_flags):
                target_group = group_names[idx]
                for layer_name in target_group:
                    m = dict(self.model.named_modules()).get(layer_name)
                    if isinstance(m, nn.Conv2d):
                        # 그룹이 죽으면 0, 살면 1 (여기서는 단순 그룹 전체 on/off 예시)
                        # 상세 채널별 제어가 필요하면 이 부분을 더 정교하게 다듬을 수 있음
                        val = 1.0 if is_alive else 0.0
                        m.mask.fill_(val)
        
        print(f"[PDT] Optimization Complete. Budget: {memory_budget/1e6:.2f}M, Groups Alive: {sum(optimal_mask_flags)}")

    @torch.no_grad()
    def get_current_sparsity(self):
        total_params = sum(m.weight.nelement() for m in self.layers)
        active_params = sum(m.mask.sum().item() * (m.weight.nelement() / m.weight.shape[0]) for m in self.layers)
        return (1 - active_params / (total_params + 1e-8)) * 100