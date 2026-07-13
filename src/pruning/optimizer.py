# import numpy as np

# def lagrangian_optimization(unit_scores, unit_costs, budget, unit_metadata=None):
#     """
#     이 함수는 각 'Topology Unit(Group)' 단위로 동작하며, 
#     주어진 예산 내에서 최적의 생존 마스크를 결정합니다.
#     """
    
#     # 1. 안전장치: 입력 데이터 확인
#     if len(unit_scores) == 0:
#         return np.array([], dtype=bool)

#     # 2. 수치적 안정화 (Scaling)
#     # ViT와 같이 레이어 간 스코어 편차가 큰 경우를 위해 정규화 수행
#     max_score = np.max(unit_scores) if np.max(unit_scores) > 0 else 1.0
#     norm_scores = unit_scores / max_score
    
#     # [Topology-Aware] 가성비(Efficiency) 계산
#     efficiencies = norm_scores / (unit_costs + 1e-8)
    
#     # 3. 이진 탐색 범위 설정 (라그랑주 승수 λ 후보군)
#     low = 0.0
#     high = float(np.max(efficiencies)) * 1.2 if len(efficiencies) > 0 else 1.0
    
#     # 초기 마스크 설정
#     best_unit_mask = np.ones_like(unit_scores, dtype=bool)
    
#     # 4. [Stage 3] 라그랑주 relaxation을 이용한 통합 최적화
#     # 목적 함수: Maximize Σ(Score_g * m_g) s.t. Σ(Cost_g * m_g) <= Budget
#     for _ in range(30):
#         lambda_val = (low + high) / 2
        
#         # [구조적 제약 조건 반영] 
#         # Score_g - λ * Cost_g > 0 인 유닛(그룹)만 생존
#         mask = (norm_scores - lambda_val * unit_costs) > 0
#         current_total_mem = np.sum(unit_costs[mask])
        
#         if current_total_mem <= budget:
#             # 예산 내에 들어오면, 더 성능을 높일 수 있는지 확인
#             best_unit_mask = mask
#             high = lambda_val
#         else:
#             # 예산을 초과하면 벌금을 높여서 더 많은 유닛을 탈락
#             low = lambda_val
            
#     # 5. [추가] 구조적 안전장치: 레이어별 최소 생존 보장
#     # 특정 레이어가 완전히 날아가는 것을 방지하기 위해 상위 10% 유닛은 강제로 살립니다.
#     min_keep_count = max(1, int(len(unit_scores) * 0.1))
#     safe_indices = np.argsort(norm_scores)[-min_keep_count:]
#     best_unit_mask[safe_indices] = True
            
#     # 6. [최종 보루] 만약 모든 그룹이 죽었다면 가성비 순으로 5% 복구
#     if np.sum(best_unit_mask) == 0:
#         num_keep = max(1, int(len(unit_scores) * 0.05))
#         top_indices = np.argsort(efficiencies)[-num_keep:]
#         best_unit_mask[top_indices] = True
            
#     return best_unit_mask


import os

import numpy as np

def lagrangian_optimization(unit_scores, unit_costs, budget, unit_metadata=None):
    """
    ViT 최적화 버전: 
    - 수치적 안정성을 위한 로그 스케일링 적용
    - 레이어별 생존 균형을 위한 페널티 부여 가능
    """
    
    if len(unit_scores) == 0:
        return np.array([], dtype=bool)

    # 1. 수치적 안정화: 점수가 너무 작거나 큰 경우를 대비해 Log-space 느낌으로 정규화
    # 스코어가 0인 경우를 대비해 아주 작은 값을 더함
    scores = np.array(unit_scores)
    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
    
    # [ViT 특화] 가성비 계산 시 파라미터 절감 효과를 선형보다 조금 더 높게 평가할 수도 있음
    efficiencies = scores / (unit_costs + 1e-8)
    
    low = 0.0
    high = float(np.max(efficiencies)) * 2.0 if len(efficiencies) > 0 else 1.0
    
    best_unit_mask = np.ones_like(unit_scores, dtype=bool)
    
    # 2. 이진 탐색 (라그랑주 승수 찾기)
    for _ in range(40): # ViT의 복잡한 구조를 위해 반복 횟수 상향
        lambda_val = (low + high) / 2
        
        # 비용 대비 중요도가 벌금(lambda)보다 높은 것만 생존
        mask = (scores - lambda_val * unit_costs) > 0
        current_cost = np.sum(unit_costs[mask])
        
        if current_cost <= budget:
            best_unit_mask = mask
            high = lambda_val
        else:
            low = lambda_val
            
    # 3. [핵심] 구조적 붕괴 방지 (Per-Group Minimum Survival)
    # 특정 그룹(QKV, MLP 등)이 아예 사라지는 것을 막기 위해 
    # 각 레이어 그룹별로 최소 1개 이상의 유닛은 무조건 살림
    debug_resource_alloc = os.getenv('MCPRUNE_DEBUG_RESOURCE_ALLOC') == '1'
    if debug_resource_alloc:
        metadata_status = "enabled" if unit_metadata is not None else "missing"
        print(
            f"[Resource Allocation] metadata={metadata_status} "
            f"units={len(unit_scores)} budget={float(budget):.2f} "
            f"initial_keep={int(np.sum(best_unit_mask))}/{len(best_unit_mask)} "
            f"initial_cost={float(np.sum(unit_costs[best_unit_mask])):.2f}"
        )

    if unit_metadata is not None:
        # 그룹 ID별로 가장 스코어가 높은 녀석은 강제 생존
        unique_groups = sorted(set([m[0]['id'] for m in unit_metadata]))
        if debug_resource_alloc:
            print(
                f"[Resource Allocation] group-aware safety active: "
                f"{len(unique_groups)} groups"
            )
            for g_id in unique_groups:
                group_indices = [idx for idx, m in enumerate(unit_metadata) if m[0]['id'] == g_id]
                kept = int(np.sum(best_unit_mask[group_indices]))
                print(
                    f"  group {g_id:02d}: units={len(group_indices)} "
                    f"kept_before_safety={kept}"
                )

        restored_units = []
        for g_id in unique_groups:
            # 해당 그룹에 속한 유닛들의 인덱스 추출
            group_indices = [idx for idx, m in enumerate(unit_metadata) if m[0]['id'] == g_id]
            if not any(best_unit_mask[group_indices]):
                # 해당 그룹에서 가장 점수가 높은 놈 하나 복구
                best_idx = group_indices[np.argmax(scores[group_indices])]
                best_unit_mask[best_idx] = True
                restored_units.append((g_id, int(unit_metadata[best_idx][1])))

        if debug_resource_alloc:
            if restored_units:
                print("[Resource Allocation] restored one unit for empty groups:")
                for g_id, unit_idx in restored_units:
                    print(f"  group {g_id:02d}: restored_unit={unit_idx}")
            else:
                print("[Resource Allocation] no empty-group restore needed")

        # Keep the group-safety restore within the requested resource budget
        # whenever there is more than one surviving unit in a group.
        group_ids = [m[0]['id'] for m in unit_metadata]
        repair_removed = []
        while np.sum(unit_costs[best_unit_mask]) > budget:
            removable = []
            for idx, keep in enumerate(best_unit_mask):
                if not keep:
                    continue
                g_id = group_ids[idx]
                group_indices = [i for i, gid in enumerate(group_ids) if gid == g_id]
                if np.sum(best_unit_mask[group_indices]) > 1:
                    removable.append(idx)

            if not removable:
                break

            remove_idx = min(removable, key=lambda i: efficiencies[i])
            best_unit_mask[remove_idx] = False
            repair_removed.append((group_ids[remove_idx], int(unit_metadata[remove_idx][1])))

        if debug_resource_alloc:
            final_cost = float(np.sum(unit_costs[best_unit_mask]))
            if repair_removed:
                print("[Resource Allocation] budget repair removed restored/extra units:")
                for g_id, unit_idx in repair_removed[:20]:
                    print(f"  group {g_id:02d}: removed_unit={unit_idx}")
                if len(repair_removed) > 20:
                    print(f"  ... ({len(repair_removed) - 20} more)")
            else:
                print("[Resource Allocation] budget repair not needed")
            print(
                f"[Resource Allocation] final_keep={int(np.sum(best_unit_mask))}/"
                f"{len(best_unit_mask)} final_cost={final_cost:.2f} "
                f"budget={float(budget):.2f}"
            )

    # 4. 최종 예산 초과 여부 재확인 (강제 복구로 인해 예산이 넘칠 경우 미세 조정)
    # 이 부분은 필요 시 추가 (보통 1-2개 복구로는 크게 안 넘음)

    return best_unit_mask
