import numpy as np

def lagrangian_optimization(unit_scores, unit_costs, budget, unit_metadata=None):
    """
    이 함수는 각 'Topology Unit(Group)' 단위로 동작하며, 
    주어진 예산 내에서 최적의 생존 마스크를 결정합니다.
    """
    
    # 1. 안전장치: 입력 데이터 확인
    if len(unit_scores) == 0:
        return np.array([], dtype=bool)

    # 2. 수치적 안정화 (Scaling)
    # ViT와 같이 레이어 간 스코어 편차가 큰 경우를 위해 정규화 수행
    max_score = np.max(unit_scores) if np.max(unit_scores) > 0 else 1.0
    norm_scores = unit_scores / max_score
    
    # [Topology-Aware] 가성비(Efficiency) 계산
    efficiencies = norm_scores / (unit_costs + 1e-8)
    
    # 3. 이진 탐색 범위 설정 (라그랑주 승수 λ 후보군)
    low = 0.0
    high = float(np.max(efficiencies)) * 1.2 if len(efficiencies) > 0 else 1.0
    
    # 초기 마스크 설정
    best_unit_mask = np.ones_like(unit_scores, dtype=bool)
    
    # 4. [Stage 3] 라그랑주 relaxation을 이용한 통합 최적화
    # 목적 함수: Maximize Σ(Score_g * m_g) s.t. Σ(Cost_g * m_g) <= Budget
    for _ in range(30):
        lambda_val = (low + high) / 2
        
        # [구조적 제약 조건 반영] 
        # Score_g - λ * Cost_g > 0 인 유닛(그룹)만 생존
        mask = (norm_scores - lambda_val * unit_costs) > 0
        current_total_mem = np.sum(unit_costs[mask])
        
        if current_total_mem <= budget:
            # 예산 내에 들어오면, 더 성능을 높일 수 있는지 확인
            best_unit_mask = mask
            high = lambda_val
        else:
            # 예산을 초과하면 벌금을 높여서 더 많은 유닛을 탈락
            low = lambda_val
            
    # 5. [추가] 구조적 안전장치: 레이어별 최소 생존 보장
    # 특정 레이어가 완전히 날아가는 것을 방지하기 위해 상위 10% 유닛은 강제로 살립니다.
    min_keep_count = max(1, int(len(unit_scores) * 0.1))
    safe_indices = np.argsort(norm_scores)[-min_keep_count:]
    best_unit_mask[safe_indices] = True
            
    # 6. [최종 보루] 만약 모든 그룹이 죽었다면 가성비 순으로 5% 복구
    if np.sum(best_unit_mask) == 0:
        num_keep = max(1, int(len(unit_scores) * 0.05))
        top_indices = np.argsort(efficiencies)[-num_keep:]
        best_unit_mask[top_indices] = True
            
    return best_unit_mask