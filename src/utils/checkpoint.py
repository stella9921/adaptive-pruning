import os
import random

import numpy as np
import torch


def experiment_signature(config):
    model = config.get('model', {})
    strategy = config.get('strategy', {})
    return {
        'model': model.get('name'),
        'strategy_method': strategy.get('method'),
        'strategy_preset': strategy.get('preset'),
        'pruning_ratio': float(strategy.get('pruning_ratio', 0.0)),
    }


def _random_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_random_state(state):
    if not state:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state.get('cuda') is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def save_training_checkpoint(path, model, optimizer, scheduler, epoch, history, **extra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'epoch': int(epoch),
        'history': history,
        'random_state': _random_state(),
        **extra,
    }
    temporary_path = f"{path}.tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)
    return path


def load_training_checkpoint(
    path, model, optimizer, scheduler, device, expected_signature=None
):
    checkpoint = torch.load(path, map_location=device)
    required = {'model_state_dict', 'optimizer_state_dict', 'epoch'}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(
            f"Checkpoint is not resumable; missing fields: {sorted(missing)}"
        )
    saved_signature = checkpoint.get('experiment_signature')
    if expected_signature is not None and saved_signature != expected_signature:
        raise ValueError(
            "Checkpoint config mismatch: "
            f"saved={saved_signature}, current={expected_signature}"
        )
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler_state = checkpoint.get('scheduler_state_dict')
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    _restore_random_state(checkpoint.get('random_state'))
    return checkpoint


def load_model_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint
