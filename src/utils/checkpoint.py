import os

import torch


def save_training_checkpoint(path, model, optimizer, scheduler, epoch, history, **extra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'epoch': int(epoch),
        'history': history,
        **extra,
    }, path)
    return path


def load_training_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device)
    required = {'model_state_dict', 'optimizer_state_dict', 'epoch'}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(
            f"Checkpoint is not resumable; missing fields: {sorted(missing)}"
        )
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler_state = checkpoint.get('scheduler_state_dict')
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    return checkpoint
