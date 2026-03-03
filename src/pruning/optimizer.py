# import numpy as np

# def lagrangian_optimization(unit_scores, unit_costs, budget):
#     """
#     이 함수는 FX로 묶인 'Topology Unit(Group)' 단위로 동작
    
#     unit_scores: 각 그룹(연계 채널 묶음)의 통합 Hessian + Gradient 에너지 합산값
#     unit_costs: 각 그룹의 통합 파라미터/메모리 비용 합산값
#     budget: 전체 네트워크에서 허용된 목표 메모리 양 (제약 조건)
#     """
    
#     # 1. 안전장치: 입력 데이터 확인
#     if len(unit_scores) == 0:
#         return np.array([], dtype=bool)

#     # 2. [Topology-Aware] 가성비(Efficiency) 계산
#     # 연계 관계가 있는 유닛의 단위 비용당 정확도 기여도
#     # 목적함수 내의 수치화된 가치
#     efficiencies = unit_scores / (unit_costs + 1e-8)
    
#     # 3. 이진 탐색 범위 설정 (라그랑주 승수 λ 후보군)
#     low = 0.0
#     high = float(np.max(efficiencies)) if len(efficiencies) > 0 else 1.0
    
#     # 초기 마스크 설정 (최소한의 유닛은 살리기)
#     best_unit_mask = np.ones_like(unit_scores, dtype=bool)
    
#     # 4. [Stage 3] 라그랑주 relaxation을 이용한 통합 최적화
#     # 목적 함수: Maximize Σ(Score_g * m_g) s.t. Σ(Cost_g * m_g) <= Budget
#     for _ in range(30):
#         lambda_val = (low + high) / 2
        
#         # [구조적 제약 조건 반영] 
#         # Score_g - λ * Cost_g > 0 인 유닛(그룹)만 생존
#         # 여기서 m_g (mask)는 그룹 전체의 운명을 결정하는 정성적/정량적 지표
#         mask = (unit_scores - lambda_val * unit_costs) > 0
#         current_total_mem = np.sum(unit_costs[mask])
        
#         if current_total_mem <= budget:
#             # 예산 내에 들어오면, 더 성능을 높일 수 있는지 확인
#             best_unit_mask = mask
#             high = lambda_val
#         else:
#             # 예산을 초과하면 벌금(λ)을 높여서 더 많은 유닛을 탈락
#             low = lambda_val
            
#     # 5. [안정성 보장] 구조 붕괴 방지용 최소 생존 로직
#     # 만약 벌금이 너무 세서 모든 그룹이 다 죽었다면, 가성비 순으로 최소 5% 복구
#     if np.sum(best_unit_mask) == 0:
#         num_keep = max(1, int(len(unit_scores) * 0.05))
#         top_indices = np.argsort(efficiencies)[-num_keep:]
#         best_unit_mask[top_indices] = True
            
#     return best_unit_mask


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


# import numpy as np

# def lagrangian_optimization(unit_scores, unit_costs, budget):
#     """
#     Algorithm 2: Resource Allocation via Efficiency Ranking 기반 구현
    
#     unit_scores: Saliency (Hessian + Grad EMA)
#     unit_costs: Resource Consumption (Cost)
#     budget: Target Budget (B)
#     """
    
#     # 1. 안전장치: 데이터가 없으면 빈 마스크 반환
#     num_units = len(unit_scores)
#     if num_units == 0:
#         return np.array([], dtype=bool)

#     # 2. [Step 1] 가성비(Efficiency) 계산: rho = S / Cost
#     # 정확도 기여도 대비 자원 소모 효율을 측정
#     efficiencies = unit_scores / (unit_costs + 1e-8)
    
#     # 3. [Step 1] 효율성 기준 내림차순 정렬 (Ranking)
#     # 가성비가 좋은 채널 묶음을 우선적으로 선택하기 위함
#     sorted_indices = np.argsort(efficiencies)[::-1]
    
#     # 4. [Step 2] Greedy Selection (예산 한도 내 선택)
#     best_unit_mask = np.zeros(num_units, dtype=bool)
#     current_total_cost = 0.0
    
#     for idx in sorted_indices:
#         # 이 유닛을 추가해도 예산(Budget)을 넘지 않는가?
#         if current_total_cost + unit_costs[idx] <= budget:
#             best_unit_mask[idx] = True
#             current_total_cost += unit_costs[idx]
#         else:
#             # 예산이 초과되면 더 이상 담지 않음 (Greedy 종료)
#             # [참고] 여기서 break를 하지 않으면 뒤에 아주 작은 cost 유닛이 들어올 수 있음
#             continue 

#     # 5. [Step 3] Topological Constraint Check (최소 생존 보장)
#     # 만약 예산이 너무 타이트해서 하나도 안 선택되었다면, 가성비 1등은 무조건 살림
#     if np.sum(best_unit_mask) == 0:
#         top_idx = sorted_indices[0]
#         best_unit_mask[top_idx] = True
            
#     return best_unit_mask