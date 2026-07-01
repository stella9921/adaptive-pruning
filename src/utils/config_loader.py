import argparse
import yaml
import os


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = yaml.load(f, Loader=UniqueKeyLoader) or {}
        except (yaml.YAMLError, ValueError) as error:
            raise ValueError(f"Invalid YAML config {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return data


def _normalize_strategy_config(data, path):
    """Validate the canonical `strategy: {...}` preset structure."""
    if 'strategy' not in data:
        raise ValueError(
            f"Strategy config must use the canonical 'strategy:' root key: {path}"
        )
    strategy = data['strategy']
    if not isinstance(strategy, dict):
        raise ValueError(f"Strategy config must contain a mapping: {path}")
    if 'method' not in strategy:
        raise ValueError(f"Strategy config is missing required key 'method': {path}")
    return strategy


def _apply_pruning_target(strategy, args, parser):
    """Normalize every method to one removal ratio in the [0, 1) range."""
    if args.pruning_ratio is not None:
        pruning_ratio = args.pruning_ratio
    elif 'pruning_ratio' in strategy:
        pruning_ratio = strategy['pruning_ratio']
    else:
        raise ValueError("Strategy config is missing required key 'pruning_ratio'")

    pruning_ratio = float(pruning_ratio)
    if not 0.0 <= pruning_ratio < 1.0:
        raise ValueError(
            f"pruning_ratio must satisfy 0 <= ratio < 1, got {pruning_ratio}"
        )

    strategy['pruning_ratio'] = pruning_ratio


def _validate_config(config, args):
    strategy = config['strategy']
    bounded_ratios = {
        'group_selection_ratio': (0.0, 1.0),
        'min_survival_ratio': (0.0, 1.0),
        'ema_decay': (0.0, 1.0),
    }
    for key, (lower, upper) in bounded_ratios.items():
        if key not in strategy:
            continue
        value = float(strategy[key])
        if not lower <= value <= upper:
            raise ValueError(f"{key} must be in [{lower}, {upper}], got {value}")

    positive_keys = ('start_epoch', 'prune_every', 'hessian_iter', 'k_horizon')
    for key in positive_keys:
        if key in strategy and int(strategy[key]) < 1:
            raise ValueError(f"{key} must be >= 1, got {strategy[key]}")

    if args.smoke_test:
        total_epochs = args.epochs if args.epochs is not None else 1
        start_epoch = args.start_epoch if args.start_epoch is not None else 1
        prune_every = args.prune_every if args.prune_every is not None else 1
    else:
        total_epochs = (
            args.epochs if args.epochs is not None
            else config.get('model', {}).get('epochs', config.get('epochs', 1))
        )
        start_epoch = (
            args.start_epoch if args.start_epoch is not None
            else strategy.get('start_epoch', 1)
        )
        prune_every = (
            args.prune_every if args.prune_every is not None
            else strategy.get('prune_every', 20)
        )
    if total_epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {total_epochs}")
    if start_epoch > total_epochs:
        raise ValueError(
            f"start_epoch ({start_epoch}) cannot exceed epochs ({total_epochs})"
        )
    if prune_every < 1:
        raise ValueError(f"prune_every must be >= 1, got {prune_every}")

def load_config():
    parser = argparse.ArgumentParser(description='Adaptive Pruning Experiment')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--strategy', type=str, required=True)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--start_epoch', type=int, default=None)
    parser.add_argument('--prune_every', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lambda_h', type=float, default=None)
    parser.add_argument('--group_selection_ratio', type=float, default=None)
    parser.add_argument('--pruning_ratio', type=float, default=None)
    parser.add_argument('--min_survival_ratio', type=float, default=None)
    parser.add_argument('--profile_pytorch', action='store_true')
    parser.add_argument('--profile_nvtx', action='store_true')
    parser.add_argument('--smoke_test', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()

    def get_real_path(input_path, base_dir):
        if os.path.exists(input_path):
            return input_path
        p = input_path if input_path.endswith('.yaml') else f"{input_path}.yaml"
        if os.path.exists(p): return p
        p = os.path.join(base_dir, p)
        return p

    base_path = 'configs/base.yaml'
    config = {}
    if os.path.exists(base_path):
        config = _load_yaml(base_path)

    model_config_path = get_real_path(args.model, 'configs/model')
    if os.path.exists(model_config_path):
        model_config = _load_yaml(model_config_path)
        if 'model' not in config: config['model'] = {}
        config['model'].update(model_config)
        if 'dataset' in model_config:
            config['dataset'] = model_config['dataset']
    else:
        print(f"Warning: Model config not found at {model_config_path}")

    strategy_config_path = get_real_path(args.strategy, 'configs/strategy')
    if os.path.exists(strategy_config_path):
        strategy_data = _load_yaml(strategy_config_path)
        config['strategy'] = _normalize_strategy_config(
            strategy_data, strategy_config_path
        )
    else:
        raise FileNotFoundError(f"Strategy config file not found: {strategy_config_path}")

    preset_name = os.path.splitext(os.path.basename(strategy_config_path))[0]
    config['strategy']['preset'] = preset_name
    _apply_pruning_target(config['strategy'], args, parser)
    _validate_config(config, args)
    config.setdefault('config_sources', {})
    config['config_sources'].update({
        'base': base_path,
        'model': model_config_path,
        'strategy': strategy_config_path,
    })

    config.setdefault('save_dir', './exp/checkpoints')
    os.makedirs(config['save_dir'], exist_ok=True)

    print(
        f"[Config] preset={preset_name} "
        f"method={config['strategy']['method']} "
        f"pruning_ratio={config['strategy']['pruning_ratio']:.4f} "
        f"file={strategy_config_path}"
    )

    return config, args
