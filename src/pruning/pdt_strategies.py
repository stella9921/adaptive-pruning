import torch
import torch.nn as nn
from .base import BasePruner
from src.models import find_prunable_blocks
# 앞서 작성한 Hessian 엔진 임포트
from .engine.hessian_free import SNOWSEngine

class PDTPruner(BasePruner):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.ema_decay = config['strategy'].get('ema_decay', 0.9)
        # [Ver.3 핵심] Hessian 반영 비중 (0.5면 반반 반영)
        self.lambda_h = config['strategy'].get('lambda_h', 0.5) 
        
        # SNOWS 엔진 초기화 (n_iter는 CG 반복 횟수)
        self.engine = SNOWSEngine(n_iter=config['strategy'].get('hessian_iter', 5))
        
        # 모델별 프루닝 대상 레이어 탐색
        prunable_dict = find_prunable_blocks(model, config['model']['name'])
        self.layers = list(prunable_dict.values())
        
        # 필요한 버퍼(mask, grad_ema, hessian_score) 등록 확인
        self._check_buffers()

    def _check_buffers(self):
        """각 레이어에 필요한 연산용 버퍼 등록"""
        for m in self.layers:
            n_f = m.weight.shape[0]
            if not hasattr(m, 'mask'):
                m.register_buffer("mask", torch.ones(n_f, device=self.device))
            if not hasattr(m, 'grad_ema'):
                m.register_buffer("grad_ema", torch.zeros(n_f, device=self.device))
            # [Ver.3 핵심] Hessian 에너지를 담을 버퍼 추가
            if not hasattr(m, 'hessian_score'):
                m.register_buffer("hessian_score", torch.zeros(n_f, device=self.device))

    def update_ema_and_mask_grad(self):
        """매 배치마다 실행: 1차 미분 에너지(Gradient Energy) 업데이트"""
        with torch.no_grad():
            for m in self.layers:
                if m.weight.grad is not None:
                    # 프루닝된 채널의 그래디언트 차단 (Zeroing)
                    m.weight.grad.mul_(m.mask.view(-1, 1, 1, 1))
                    if hasattr(m, 'bias') and m.bias is not None and m.bias.grad is not None:
                        m.bias.grad.mul_(m.mask)
                    
                    # Gradient Energy 계산 (제곱 평균)
                    # g = mean(grad^2) over (output, k, k) -> input channel dimension
                    g = m.weight.grad.pow(2).view(m.weight.shape[0], -1).mean(1)
                    # EMA 업데이트: g_ema = decay * g_ema + (1-decay) * current_g
                    m.grad_ema.mul_(self.ema_decay).add_(g, alpha=1 - self.ema_decay)

    def apply_mask_to_weights(self, optimizer=None):
        """가중치와 옵티마이저 모멘텀에 마스크 강제 적용"""
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
                                state["momentum_buffer"].mul_(mask.view_as(p))

    def step_pruning(self, loss, target_ratio=None):
        """
        프루닝 이벤트 발생 시 실행: Hessian(2차) + Gradient(1차) 결합 스코어링
        """
        if target_ratio is None:
            target_ratio = self.config['strategy'].get('target_ratio', 0.4)
            
        # --- [Step A & B] SNOWS 엔진을 통한 Hessian-Vector Product 추출 ---
        # p: 최적의 탐색 방향, hv_list: 각 파라미터별 Hessian 반응
        p, hv_list = self.engine.get_smart_direction_p(loss, self.model)
        
        # 모델의 모든 파라미터 중 학습 가능한 것만 필터링 (HV 리스트와 매칭용)
        trainable_params = [param for param in self.model.parameters() if param.requires_grad]

        # --- [Step C] Hessian 정보를 각 레이어의 버퍼로 변환 및 저장 ---
        with torch.no_grad():
            for param, hv in zip(trainable_params, hv_list):
                for m in self.layers:
                    if m.weight is param:
                        # Hv의 L2-Norm 제곱을 계산하여 '곡률 에너지' 산출
                        # 4D 텐서를 (채널, 나머지)로 펼쳐서 연산
                        h_energy = hv.pow(2).view(hv.shape[0], -1).mean(1)
                        m.hessian_score.copy_(h_energy)

        # --- 최종 결합 지표 산출 및 전역 임계값(Quantile) 프루닝 ---
        all_combined_scores = []
        for m in self.layers:
            # Score = 1차 미분(EMA) + lambda * 2차 미분(Hessian)
            combined = m.grad_ema + (self.lambda_h * m.hessian_score)
            all_combined_scores.append(combined)
            
        all_scores = torch.cat(all_combined_scores)
        threshold = torch.quantile(all_scores, target_ratio)

        # 마스크 업데이트
        with torch.no_grad():
            for m in self.layers:
                combined_m = m.grad_ema + (self.lambda_h * m.hessian_score)
                m.mask.copy_((combined_m > threshold).float())
        
        print(f"[PDT] Pruning Step: Target Ratio {target_ratio}, Threshold {threshold:.6f}")

    @torch.no_grad()
    def get_current_sparsity(self):
        """모델의 실제 희소도(Sparsity) 계산"""
        total_params = 0
        active_params = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nelem = m.weight.nelement()
                total_params += nelem
                if hasattr(m, 'mask'):
                    active_params += int(m.mask.sum().item()) * (nelem // m.weight.shape[0])
                else:
                    active_params += nelem
        return (1 - active_params / (total_params + 1e-8)) * 100