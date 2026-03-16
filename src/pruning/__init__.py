from .pat_strategies import PATPruner
from .pdt_strategies import PDTPruner

def get_pruner(model, config, sensitivity_si=None):
    """
    설정(YAML)에 따라 적절한 프루너 객체를 생성하여 반환하는 팩토리 함수.
    """
    method = config['strategy']['method'].upper() # 'PAT' or 'PDT'
    
    if method == 'PAT':
        if sensitivity_si is None:
            raise ValueError("PAT 기법을 사용하려면 사전 계산된 sensitivity_si가 필요합니다.")
        return PATPruner(model, config, sensitivity_si)
    
    elif method == 'PDT':
        return PDTPruner(model, config)
    
    else:
        raise ValueError(f"지원하지 않는 프루닝 기법입니다: {method}")