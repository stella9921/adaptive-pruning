import torch

class SNOWSEngine:
    """
    [Blackwell Optimized] K-Horizon Hessian-free Engine
    모델 전체가 아닌, 선별된(Target) 파라미터 리스트에 대해서만 
    정밀 Hessian-Vector Product(HVP)를 계산하여 연산 효율을 극대화합니다.
    """
    
    def __init__(self, n_iter=5, tolerance=1e-5):
        self.n_iter = n_iter        # Conjugate Gradient 반복 횟수
        self.tolerance = tolerance  # 수렴 임계값

    def compute_hvp(self, loss, params, p):
        """
        [수식 반영] Taylor 2차 근사의 핵심인 Hv (Hessian-Vector Product) 계산
        - ∇(∇Loss^T * p)를 통해 Hessian 행렬 직접 구성 없이 Hv 산출
        """
        # 1차 그래디언트 계산 (create_graph=True로 2차 미분 연산 그래프 유지)
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        
        # dot_product = g^T * p (그래디언트와 탐색 방향 p의 내적)
        dot_product = sum((g * p_i).sum() for g, p_i in zip(grads, p))
        
        # 2차 미분 수행: Blackwell의 48GB VRAM을 활용해 깊은 연산 그래프 보존
        hvp = torch.autograd.grad(dot_product, params, retain_graph=True)
        return hvp

    def get_k_step_hessian_selective(self, loss, target_params, K_horizon=10):
        """
        [Selective Targeting]
        pdt_strategies에서 위상(EMA) 분석을 통해 넘겨준 '하위 그룹 파라미터'들만 
        대상으로 CG(Conjugate Gradient)를 수행하여 정밀 Hessian 점수를 추출합니다.
        """
        if not target_params:
            return []

        # 1. 초기 잔차(r) 및 탐색 방향(p) 설정 (선별된 타겟 파라미터만 미분)
        grads = torch.autograd.grad(loss, target_params, create_graph=True, retain_graph=True)
        r = [g.detach().clone() for g in grads]
        p = [r_i.clone() for r_i in r]
        
        rdot_old = sum((r_i * r_i).sum() for r_i in r)
        
        # 2. CG Iteration: 선택된 타겟 내에서 곡률(Curvature) 탐색
        for i in range(self.n_iter):
            # 타겟 파라미터들에 대해서만 Hv 계산 (연산량 대폭 절감)
            hv = self.compute_hvp(loss, target_params, p)
            
            # 곡률 계산: p^T * H * p
            p_h_p = sum((p_i * hv_i).sum() for p_i, hv_i in zip(p, hv))
            
            # 스텝 사이즈 결정
            alpha = rdot_old / (p_h_p + 1e-10)
            
            # 잔차 업데이트
            for r_i, hv_i in zip(r, hv):
                r_i.sub_(alpha * hv_i)
            
            rdot_new = sum((r_i * r_i).sum() for r_i in r)
            
            # 임계값 도달 시 조기 종료
            if rdot_new < self.tolerance:
                break
                
            # 방향 켤레화(Conjugation)를 위한 beta 산출
            beta = rdot_new / rdot_old
            for p_i, r_i in zip(p, r):
                p_i.mul_(beta).add_(r_i)
                
            rdot_old = rdot_new
            
        # 3. 최적 방향으로 계산된 최종 Hessian 반응(hv) 리스트 반환
        return hv

    def get_channel_scores(self, hv_list, params_with_name):
        """
        [Score Conversion]
        HVP 결과(hv)를 채널별 에너지 점수(s_gc)로 변환합니다.
        """
        scores = {}
        for (name, param), hv in zip(params_with_name, hv_list):
            if hv.dim() >= 2: 
                # L2-Norm의 제곱을 통해 채널별 곡률 민감도 추출
                energy = torch.norm(hv, p=2, dim=0)**2 
                scores[name] = energy
        return scores