import argparse
import yaml
import os

def load_config():
    """
    configs 폴더 구조를 통합하여 하나의 딕셔너리로 반환
    구조: configs/base.yaml + configs/model/{args.model}.yaml + configs/strategy/{args.strategy}.yaml
    """
    parser = argparse.ArgumentParser(description='Adaptive Pruning Experiment')
    
    # 1. 인자 설정
    parser.add_argument('--model', type=str, default='resnet18', 
                        help='Model name (resnet18, vgg16, efficientnet_b0 등)')
    parser.add_argument('--strategy', type=str, required=True, 
                        help='Strategy YAML file path (예: configs/strategy/pdt_hessian.yaml)')
    
    args, _ = parser.parse_known_args()

    # 2. Base 설정 로드 (공통 설정)
    base_path = 'configs/base.yaml'
    if os.path.exists(base_path):
        with open(base_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # 3. Model 설정 병합
    model_config_path = f'configs/model/{args.model}.yaml'
    if os.path.exists(model_config_path):
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
            # base 설정과 모델 설정을 병합 (모델 설정 우선)
            if 'model' not in config:
                config['model'] = {}
            config['model'].update(model_config)
    else:
        print(f"Warning: Model config not found at {model_config_path}")

    # 4. Strategy 설정 병합
    if os.path.exists(args.strategy):
        with open(args.strategy, 'r') as f:
            strategy_config = yaml.safe_load(f)
            config['strategy'] = strategy_config
    else:
        raise FileNotFoundError(f"Strategy config file not found: {args.strategy}")

    # 5. 결과 저장 경로 및 로그 경로 자동 설정
    # config['save_dir']이 없으면 기본값 설정
    if 'save_dir' not in config:
        config['save_dir'] = './results/checkpoints'
    
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs('./results/logs', exist_ok=True)
    os.makedirs('./results/sensitivity', exist_ok=True)

    return config, args