import argparse
import yaml
import os

def load_config():
    """
    configs 폴더 구조를 통합하여 하나의 딕셔너리로 반환.
    CLI 인자를 추가하여 터미널 명령어가 YAML 설정을 덮어쓸 수 있게 합니다.
    """
    parser = argparse.ArgumentParser(description='Adaptive Pruning Experiment')
    
    # --- [기존 인자] ---
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--strategy', type=str, required=True)
    
    # --- [현수님이 추가로 사용할 CLI 인자들 등록] ---
    # 이제 터미널에서 이 이름들을 사용할 수 있습니다.
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--start_epoch', type=int, default=None)
    parser.add_argument('--prune_every', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lambda_h', type=float, default=None)
    
    # PDTPruner 전용 핵심 인자들
    parser.add_argument('--group_selection_ratio', type=float, default=None)
    parser.add_argument('--channel_keep_ratio', type=float, default=None)
    parser.add_argument('--min_survival_ratio', type=float, default=None)

    # 알려지지 않은 인자가 와도 에러내지 않고 무시하도록 parse_known_args 사용
    args, _ = parser.parse_known_args()

    # --- 경로 전처리 함수 ---
    def get_real_path(input_path, base_dir):
        if os.path.exists(input_path):
            return input_path
        p = input_path if input_path.endswith('.yaml') else f"{input_path}.yaml"
        if os.path.exists(p): return p
        p = os.path.join(base_dir, p)
        return p

    # --- 1. Base 설정 로드 ---
    base_path = 'configs/base.yaml'
    config = {}
    if os.path.exists(base_path):
        with open(base_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

    # --- 2. Model 설정 병합 ---
    model_config_path = get_real_path(args.model, 'configs/model')
    if os.path.exists(model_config_path):
        with open(model_config_path, 'r', encoding='utf-8') as f:
            model_config = yaml.safe_load(f)
            if 'model' not in config: config['model'] = {}
            config['model'].update(model_config)
            if 'dataset' in model_config:
                config['dataset'] = model_config['dataset']
    else:
        print(f"Warning: Model config not found at {model_config_path}")

    # --- 3. Strategy 설정 병합 ---
    strategy_config_path = get_real_path(args.strategy, 'configs/strategy')
    if os.path.exists(strategy_config_path):
        with open(strategy_config_path, 'r', encoding='utf-8') as f:
            config['strategy'] = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Strategy config file not found: {strategy_config_path}")

    # --- 4. 경로 자동 생성 ---
    config.setdefault('save_dir', './results/checkpoints')
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs('./results/logs', exist_ok=True)
    os.makedirs('./results/sensitivity', exist_ok=True)

    return config, args