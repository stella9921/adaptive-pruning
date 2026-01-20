import numpy as np

def lagrangian_optimization(scores, memory_costs, budget):
    """
    scores: Hessian-based saliency (Stage 2 결과물)
    memory_costs: 각 채널/그룹의 실제 메모리 점유량
    budget: 목표 메모리 사용량 (Stage 3 제약 조건)
    """
    low, high = 0, 1e10
    best_mask = None
    
    # Lagrangian Multiplier를 이진 탐색으로 찾음
    for _ in range(20):
        lambda_val = (low + high) / 2
        # 목적 함수: Maximize(Scores * mask) - lambda * (Memory * mask)
        mask = (scores - lambda_val * memory_costs) > 0
        current_mem = (memory_costs * mask).sum()
        
        if current_mem <= budget:
            high = lambda_val
            best_mask = mask
        else:
            low = lambda_val
            
    return best_mask