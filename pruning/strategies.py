# pruning/strategies.py

EPS = 1e-8


def compute_dynamic_ratios_vanilla(
    round_idx,
    n_rounds,
    sensitivity_si,
    filter_counts_Ni_original,
    global_target_ratio,
):
    """
    전략 1: p, beta 없이 원래 하던 방식
      - fi: 필터 개수 비율
      - wi: 민감도 역수 비율
    """
    current_target_ratio = (global_target_ratio / n_rounds) * round_idx

    total_original_filters = sum(filter_counts_Ni_original.values())
    dynamic_ratios = {}
    wi = {}

    # wi 계산 (p 없음)
    sum_of_inverse_si = sum(1.0 / (abs(s) + EPS) for s in sensitivity_si.values())
    for block_name, si in sensitivity_si.items():
        wi[block_name] = (1.0 / (abs(si) + EPS)) / (sum_of_inverse_si + EPS)

    # fi: 필터 개수 비율
    fi = {
        name: count / total_original_filters
        for name, count in filter_counts_Ni_original.items()
    }

    # pr_i 계산
    sum_of_fi_wi = sum(f * w for f, w in zip(fi.values(), wi.values()))
    pr_temp = {}
    for block_name in sorted(fi.keys()):
        pr_i = (
            current_target_ratio
            * fi[block_name]
            * (wi[block_name] / (sum_of_fi_wi + EPS))
        )
        pr_temp[block_name] = pr_i

    # 전체 합 보정
    actual_sum = sum(pr_temp.values())
    if abs(actual_sum - current_target_ratio) > EPS and actual_sum > 0:
        scale_factor = current_target_ratio / actual_sum
        for block_name, ratio in pr_temp.items():
            dynamic_ratios[block_name] = ratio * scale_factor
    else:
        dynamic_ratios = pr_temp

    return dynamic_ratios


def compute_dynamic_ratios_p(
    round_idx,
    n_rounds,
    sensitivity_si,
    param_counts_Ni,
    total_block_params,
    global_target_ratio,
    p=2.5,
):
    """
    전략 2: p 넣은 버전
      - fi: '파라미터 수' 비율
      - wi: (1/|si|)^p 기반
    """
    current_target_ratio = (global_target_ratio / n_rounds) * round_idx

    dynamic_ratios = {}
    wi = {}

    # wi 계산 (p 적용)
    sum_of_inverse_si = sum(
        (1.0 / (abs(s) + EPS)) ** p for s in sensitivity_si.values()
    )
    for block_name, si in sensitivity_si.items():
        wi[block_name] = ((1.0 / (abs(si) + EPS)) ** p) / (sum_of_inverse_si + EPS)

    # fi: 파라미터 수 비율
    fi = {
        name: count / total_block_params
        for name, count in param_counts_Ni.items()
    }

    sum_of_fi_wi = sum(f * w for f, w in zip(fi.values(), wi.values()))
    pr_temp = {}
    for block_name in sorted(fi.keys()):
        pr_i = (
            current_target_ratio
            * fi[block_name]
            * (wi[block_name] / (sum_of_fi_wi + EPS))
        )
        pr_temp[block_name] = pr_i

    actual_sum = sum(pr_temp.values())
    if abs(actual_sum - current_target_ratio) > EPS and actual_sum > 0:
        scale_factor = current_target_ratio / actual_sum
        for block_name, ratio in pr_temp.items():
            dynamic_ratios[block_name] = ratio * scale_factor
    else:
        dynamic_ratios = pr_temp

    return dynamic_ratios


def compute_dynamic_ratios_beta(
    round_idx,
    n_rounds,
    sensitivity_si,
    param_counts_Ni,
    total_block_params,
    global_target_ratio,
    beta=0.5,
):
    """
    전략 3: beta 가중합 버전
      - fi: 파라미터 수 비율
      - wi: 민감도 역수 비율
      - pr_i = target * (beta * fi + (1-beta) * wi)
    """
    current_target_ratio = (global_target_ratio / n_rounds) * round_idx

    dynamic_ratios = {}

    # wi 계산 (p 없이 역수)
    wi = {}
    sum_of_inverse_si = sum(1.0 / (abs(s) + EPS) for s in sensitivity_si.values())
    for block_name, si in sensitivity_si.items():
        wi[block_name] = (1.0 / (abs(si) + EPS)) / (sum_of_inverse_si + EPS)

    # fi: 파라미터 수 비율
    fi = {
        name: count / total_block_params
        for name, count in param_counts_Ni.items()
    }

    pr_temp = {}
    for block_name in sorted(fi.keys()):
        pr_i = current_target_ratio * (
            beta * fi[block_name] + (1 - beta) * wi[block_name]
        )
        pr_temp[block_name] = pr_i

    actual_sum = sum(pr_temp.values())
    if abs(actual_sum - current_target_ratio) > EPS and actual_sum > 0:
        scale_factor = current_target_ratio / actual_sum
        for block_name, ratio in pr_temp.items():
            dynamic_ratios[block_name] = ratio * scale_factor
    else:
        dynamic_ratios = pr_temp

    return dynamic_ratios
