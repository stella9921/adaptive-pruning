import atexit
import os
import re
import sys
from datetime import datetime


def _slug(value):
    return re.sub(r'[^a-zA-Z0-9._-]+', '-', str(value)).strip('-')


def prepare_experiment(config, args):
    strategy_preset = config['strategy'].get('preset', 'pdt')
    dataset = config.get('dataset', 'dataset')
    dataset_tag = dataset if isinstance(dataset, str) else dataset.get('name', 'dataset')
    if args.dataset is not None:
        dataset_tag = args.dataset
        config['dataset'] = args.dataset

    ratio_tag = f"{config['strategy']['pruning_ratio'] * 100:05.1f}".replace('.', 'p')
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    run_id = "__".join([
        _slug(config['model']['name']),
        _slug(dataset_tag),
        _slug(strategy_preset),
        f"prune-{ratio_tag}",
        timestamp,
    ])
    run_dir = os.path.join(config.get('exp_root', './exp'), 'runs', run_id)
    config.update({
        'run_id': run_id,
        'run_dir': run_dir,
        'checkpoint_dir': os.path.join(run_dir, 'checkpoints'),
        'profiling': {
            'pytorch': bool(args.profile_pytorch),
            'nvtx': bool(args.profile_nvtx),
        },
    })
    for dirname in ('logs', 'checkpoints', 'metrics', 'plots', 'profiles'):
        os.makedirs(os.path.join(run_dir, dirname), exist_ok=True)
    return f"{run_id}__run.log"


class TeeLogger:
    def __init__(self, path, terminal=None):
        self.terminal = terminal or sys.stdout
        self.log = open(path, 'a', encoding='utf-8')
        atexit.register(self.close)

    def write(self, message):
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            encoding = self.terminal.encoding or 'utf-8'
            safe = message.encode(encoding, errors='replace').decode(encoding)
            self.terminal.write(safe)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        if not self.log.closed:
            self.log.flush()

    def close(self):
        if not self.log.closed:
            self.flush()
            self.log.close()

    @property
    def encoding(self):
        return self.terminal.encoding
