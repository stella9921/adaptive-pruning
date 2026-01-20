import numpy as np

def lagrangian_optimization(scores, memory_costs, budget):
    """
    scores: Hessian + Gradient 결합 에너지
    memory_costs: 각 채널의 파라미터/메모리 비용
    budget: 남겨야 할 목표 메모리 양
    """
    # 1. 안전장치: 점수나 비용이 비어있으면 전부 살리는 마스크 반환
    if len(scores) == 0:
        return np.array([], dtype=bool)

    # 2. 이진 탐색 범위 설정 (점수/비용의 비율로 정밀하게 설정)
    # 람다(벌금)는 '단위 비용당 점수 가치'의 임계값입니다.
    efficiencies = scores / (memory_costs + 1e-8)
    low = 0.0
    high = float(np.max(efficiencies)) if len(efficiencies) > 0 else 1.0
    
    # 3. 초기 마스크 설정 (최소한 하나는 살려야 하므로 전체 생존으로 시작)
    best_mask = np.ones_like(scores, dtype=bool)
    
    # 4. 이진 탐색 (Lagrangian Multiplier 'lambda' 찾기)
    # 30번 정도 돌면 매우 정밀하게 수렴합니다.
    for _ in range(30):
        lambda_val = (low + high) / 2
        
        # [Stage 3 핵심] 비용 대비 가치가 lambda_val(벌금)보다 큰 놈들만 생존
        mask = (scores - lambda_val * memory_costs) > 0
        current_mem = np.sum(memory_costs[mask])
        
        if current_mem <= budget:
            # 예산 안쪽으로 들어오면, 더 많이 살릴 수 있는지 확인하기 위해 벌금을 낮춤
            best_mask = mask
            high = lambda_val
        else:
            # 예산을 초과하면 벌금을 높여서 더 많이 죽임
            low = lambda_val
            
    # 5. [최종 보루] 만약 너무 가혹해서 다 죽었다면, 가성비 순으로 5%는 강제로 살림
    if np.sum(best_mask) == 0:
        # 가성비(Efficiency) 상위 5% 선택
        num_keep = max(1, int(len(scores) * 0.05))
        top_indices = np.argsort(efficiencies)[-num_keep:]
        best_mask[top_indices] = True
            
    return best_mask