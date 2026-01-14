import torch

class SNOWSEngine:
    """
    SNOWS 논문의 Hessian-free Second-order Optimization 
    Hessian 행렬을 직접 구하지 않고 CG(Conjugate Gradient)를 통해 
    Hessian-Vector Product(HVP)를 계산하고 최적의 탐색 방향 p를 도출
    """
    
    def __init__(self, n_iter=5, tolerance=1e-5):
        self.n_iter = n_iter        # CG 반복 횟수 (논문 추천: 5~10)
        self.tolerance = tolerance  # 수렴 임계값

    def compute_hvp(self, loss, params, p):
        """
        Step B: Hessian-Vector Product (Hv) 계산
        - 논문의 핵심인 '미분의 미분' 트릭을 사용합니다.
        """
        # 1차 그래디언트 계산 (create_graph=True로 2차 미분 준비)
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        
        # Scalar = g^T * p (그래디언트와 방향 벡터 p의 내적)
        dot_product = sum((g * p_i).sum() for g, p_i in zip(grads, p))
        
        # 2차 미분: ∇(g^T * p) 결과가 바로 Hv (Hessian-Vector Product)
        hvp = torch.autograd.grad(dot_product, params, retain_graph=True)
        return hvp

    def get_smart_direction_p(self, loss, model):
        """
        Step A: Conjugate Gradient (CG) 알고리즘 구현
        - Hessian 정보를 반영하여 가장 효율적인 탐색 방향 p 찾기 
        """
        params = [p for p in model.parameters() if p.requires_grad]
        
        # 초기화: 첫 잔차(r)는 초기 그래디언트
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        r = [g.detach().clone() for g in grads]
        p = [r_i.clone() for r_i in r]  # 초기 탐색 방향 p = r
        
        # r^T * r 초기값 계산
        rdot_old = sum((r_i * r_i).sum() for r_i in r)
        
        for i in range(self.n_iter):
            # 1. Hv 계산 (Hessian-Vector Product)
            hv = self.compute_hvp(loss, params, p)
            
            # 2. 곡률(Curvature) 계산: p^T * H * p
            p_h_p = sum((p_i * hv_i).sum() for p_i, hv_i in zip(p, hv))
            
            # 3. 스텝 사이즈 alpha 계산 (alpha = r^T*r / p^T*H*p)
            alpha = rdot_old / (p_h_p + 1e-10)
            
            # 4. 잔차 r 업데이트: r = r - alpha * Hv
            for r_i, hv_i in zip(r, hv):
                r_i.sub_(alpha * hv_i)
            
            # 새로운 r^T * r 계산
            rdot_new = sum((r_i * r_i).sum() for r_i in r)
            
            # 수렴 여부 확인
            if rdot_new < self.tolerance:
                break
                
            # 5. 다음 방향 p 업데이트 (beta 계산 및 방향 켤레화)
            beta = rdot_new / rdot_old
            for p_i, r_i in zip(p, r):
                p_i.mul_(beta).add_(r_i)
                
            rdot_old = rdot_new
            
        return p, hv  # 최종 방향 p와 그때의 Hessian 반응(hv)을 반환

    def get_channel_scores(self, hv_list, params):
        """
        Step C: Hv 결과를 채널별 에너지 점수로 변환
        """
        scores = {}
        for (name, param), hv in zip(params, hv_list):
            if len(hv.shape) >= 2: # 가중치 행렬인 경우만 (Conv, Linear)
                # 입력 채널(dim=1) 기준으로 L2-Norm의 제곱을 계산하여 곡률 에너지 추출
                # SNOWS는 출력 차원(dim=0)을 기준으로 하지만, 프루닝은 입력 채널 기준이 일반적
                # 프로젝트 목적에 따라 dim=0 또는 dim=1 선택 가능
                energy = torch.norm(hv, p=2, dim=0)**2 
                scores[name] = energy
        return scores