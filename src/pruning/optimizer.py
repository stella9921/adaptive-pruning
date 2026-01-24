import numpy as np

def lagrangian_optimization(unit_scores, unit_costs, budget):
    """
    이 함수는 FX로 묶인 'Topology Unit(Group)' 단위로 동작
    
    unit_scores: 각 그룹(연계 채널 묶음)의 통합 Hessian + Gradient 에너지 합산값
    unit_costs: 각 그룹의 통합 파라미터/메모리 비용 합산값
    budget: 전체 네트워크에서 허용된 목표 메모리 양 (제약 조건)
    """
    
    # 1. 안전장치: 입력 데이터 확인
    if len(unit_scores) == 0:
        return np.array([], dtype=bool)

    # 2. [Topology-Aware] 가성비(Efficiency) 계산
    # 연계 관계가 있는 유닛의 단위 비용당 정확도 기여도
    # 목적함수 내의 수치화된 가치
    efficiencies = unit_scores / (unit_costs + 1e-8)
    
    # 3. 이진 탐색 범위 설정 (라그랑주 승수 λ 후보군)
    low = 0.0
    high = float(np.max(efficiencies)) if len(efficiencies) > 0 else 1.0
    
    # 초기 마스크 설정 (최소한의 유닛은 살리기)
    best_unit_mask = np.ones_like(unit_scores, dtype=bool)
    
    # 4. [Stage 3] 라그랑주 relaxation을 이용한 통합 최적화
    # 목적 함수: Maximize Σ(Score_g * m_g) s.t. Σ(Cost_g * m_g) <= Budget
    for _ in range(30):
        lambda_val = (low + high) / 2
        
        # [구조적 제약 조건 반영] 
        # Score_g - λ * Cost_g > 0 인 유닛(그룹)만 생존
        # 여기서 m_g (mask)는 그룹 전체의 운명을 결정하는 정성적/정량적 지표
        mask = (unit_scores - lambda_val * unit_costs) > 0
        current_total_mem = np.sum(unit_costs[mask])
        
        if current_total_mem <= budget:
            # 예산 내에 들어오면, 더 성능을 높일 수 있는지 확인
            best_unit_mask = mask
            high = lambda_val
        else:
            # 예산을 초과하면 벌금(λ)을 높여서 더 많은 유닛을 탈락
            low = lambda_val
            
    # 5. [안정성 보장] 구조 붕괴 방지용 최소 생존 로직
    # 만약 벌금이 너무 세서 모든 그룹이 다 죽었다면, 가성비 순으로 최소 5% 복구
    if np.sum(best_unit_mask) == 0:
        num_keep = max(1, int(len(unit_scores) * 0.05))
        top_indices = np.argsort(efficiencies)[-num_keep:]
        best_unit_mask[top_indices] = True
            
    return best_unit_mask