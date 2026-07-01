import torch.optim as optim


def build_scheduler(optimizer, model_config, total_epochs):
    scheduler_name = str(model_config.get('scheduler', 'none')).lower()
    if scheduler_name in ('none', '', 'null'):
        return None
    if scheduler_name == 'multistep':
        milestones = [int(epoch) for epoch in model_config.get('milestones', [])]
        if not milestones:
            raise ValueError("multistep scheduler requires non-empty milestones")
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=float(model_config.get('lr_gamma', 0.1)),
        )
    if scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(total_epochs),
            eta_min=float(model_config.get('min_lr', 0.0)),
        )
    raise ValueError(f"Unknown scheduler: {scheduler_name}")
