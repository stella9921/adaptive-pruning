import argparse
import yaml
import os

def load_config():
    """
    configs 폴더 구조를 통합하여 하나의 딕셔너리로 반환.
    이름만 입력하거나 전체 경로를 입력해도 자동으로 처리합니다.
    """
    parser = argparse.ArgumentParser(description='Adaptive Pruning Experiment')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--strategy', type=str, required=True)
    args, _ = parser.parse_known_args()

    # --- 경로 전처리 함수 ---
    def get_real_path(input_path, base_dir):
        # 1. 이미 정확한 경로면 그대로 반환
        if os.path.exists(input_path):
            return input_path
        # 2. 확장자(.yaml)가 없으면 붙여보기
        p = input_path if input_path.endswith('.yaml') else f"{input_path}.yaml"
        if os.path.exists(p): return p
        # 3. base_dir(configs/model 등) 붙여보기
        p = os.path.join(base_dir, p)
        return p

    # --- 1. Base 설정 로드 ---
    base_path = 'configs/base.yaml'
    config = {}
    if os.path.exists(base_path):
        with open(base_path, 'r') as f:
            config = yaml.safe_load(f)

    # --- 2. Model 설정 병합 ---
    # 이제 configs/model/을 안 붙여도, 혹은 붙여도 알아서 찾습니다.
    model_config_path = get_real_path(args.model, 'configs/model')
    if os.path.exists(model_config_path):
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
            if 'model' not in config: config['model'] = {}
            config['model'].update(model_config)
            # 만약 모델 설정 안에 dataset 정보가 있다면 밖으로 빼줌
            if 'dataset' in model_config:
                config['dataset'] = model_config['dataset']
    else:
        print(f"Warning: Model config not found at {model_config_path}")

    # --- 3. Strategy 설정 병합 ---
    # 이제 이름만 써도, 경로를 다 써도 알아서 찾습니다.
    strategy_config_path = get_real_path(args.strategy, 'configs/strategy')
    if os.path.exists(strategy_config_path):
        with open(strategy_config_path, 'r') as f:
            config['strategy'] = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Strategy config file not found: {strategy_config_path}")

    # --- 4. 경로 자동 생성 ---
    config.setdefault('save_dir', './results/checkpoints')
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs('./results/logs', exist_ok=True)
    os.makedirs('./results/sensitivity', exist_ok=True)

    return config, args