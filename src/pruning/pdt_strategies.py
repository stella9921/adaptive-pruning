import torch
import torch.nn as nn
from .base import BasePruner
from src.models import find_prunable_blocks

class PDTPruner(BasePruner):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.ema_decay = config['strategy'].get('ema_decay', 0.9)
        
        # [수정] 공통 모듈을 사용하여 프루닝 대상 레이어들을 가져옴
        prunable_dict = find_prunable_blocks(model, config['model']['name'])
        self.layers = list(prunable_dict.values())
        
        # 모델 생성 시 이미 버퍼가 등록되어 있으므로, 초기화 확인만 수행
        self._check_buffers()

    def _check_buffers(self):
        """버퍼가 제대로 등록되어 있는지 확인하고, 만약 없으면 추가 (방어적 코드)"""
        for m in self.layers:
            if not hasattr(m, 'mask'):
                n_f = m.weight.shape[0]
                m.register_buffer("mask", torch.ones(n_f, device=self.device))
            if not hasattr(m, 'grad_ema'):
                n_f = m.weight.shape[0]
                m.register_buffer("grad_ema", torch.zeros(n_f, device=self.device))

    def update_ema_and_mask_grad(self):
        """논문 4.1 & 4.2: EMA 업데이트 및 프루닝된 필터의 그래디언트 차단"""
        with torch.no_grad():
            for m in self.layers:
                if m.weight.grad is not None:
                    # 1. 그래디언트 마스킹 (프루닝된 채널의 역전파 차단)
                    m.weight.grad.mul_(m.mask.view(-1, 1, 1, 1))
                    if hasattr(m, 'bias') and m.bias is not None and m.bias.grad is not None:
                        m.bias.grad.mul_(m.mask)
                    
                    # 2. 논문 수식 (6): Gradient Energy EMA 업데이트
                    # g_t = (1-alpha) * current_grad^2 + alpha * grad_ema
                    g = m.weight.grad.pow(2).view(m.weight.shape[0], -1).mean(1)
                    m.grad_ema.mul_(self.ema_decay).add_(g, alpha=1 - self.ema_decay)

    def apply_mask_to_weights(self, optimizer=None):
        """논문 4.2: 가중치와 옵티마이저 모멘텀에 마스크 강제 적용 (0으로 유지)"""
        with torch.no_grad():
            for m in self.layers:
                mask = m.mask
                # 가중치/편향 데이터 0으로 밀기
                m.weight.data.mul_(mask.view(-1, 1, 1, 1))
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.mul_(mask)

                # 옵티마이저 모멘텀 버퍼도 0으로 밀어야 나중에 다시 살아나지 않음
                if optimizer is not None:
                    for p in [m.weight, m.bias]:
                        if p is not None and p in optimizer.state:
                            state = optimizer.state[p]
                            if "momentum_buffer" in state:
                                state["momentum_buffer"].mul_(mask.view_as(p))

    def step_pruning(self, target_ratio=None):
        """논문 4.3: 전역 임계값을 계산하여 마스크 업데이트 (프루닝 이벤트 발생)"""
        if target_ratio is None:
            target_ratio = self.config['strategy'].get('target_ratio', 0.4)
            
        # 모든 레이어의 EMA 점수를 하나로 합쳐서 전역 임계값(Quantile) 산출
        all_scores = torch.cat([m.grad_ema for m in self.layers])
        threshold = torch.quantile(all_scores, target_ratio)

        with torch.no_grad():
            for m in self.layers:
                # EMA 점수가 임계값보다 높은 채널만 1, 나머지는 0
                m.mask.copy_((m.grad_ema > threshold).float())

    @torch.no_grad()
    def get_current_sparsity(self):
        """현재 마스킹된 모델의 실제 희소도(%) 계산"""
        total_conv_params = 0
        active_conv_params = 0
        
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nelem = m.weight.nelement()
                total_conv_params += nelem
                if hasattr(m, 'mask'):
                    # 마스크가 1인 필터의 파라미터 개수만 카운트
                    active_conv_params += int(m.mask.sum().item()) * (nelem // m.weight.shape[0])
                else:
                    active_conv_params += nelem
                    
        return (1 - active_conv_params / (total_conv_params + 1e-8)) * 100