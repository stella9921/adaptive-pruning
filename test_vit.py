import torch
import timm
from src.pruning.topology_manager import get_model_topology
from src.pruning.pdt_strategies import ViTPDTPruner

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

model = timm.create_model('deit_small_patch16_224', pretrained=False).cuda()
topology = get_model_topology(model)

config = {'strategy': {'channel_keep_ratio': 0.5, 'ema_decay': 0.95,
                       'lambda_h': 0.005, 'k_horizon': 25, 'hessian_iter': 10}}
pruner = ViTPDTPruner(model, config, topology_groups=topology)

x = torch.randn(2, 3, 224, 224).cuda()
criterion = torch.nn.CrossEntropyLoss()

# ← EMA를 몇 스텝 먼저 쌓아줌 (실제 학습에서는 자동으로 쌓임)
print("EMA 워밍업 중...")
for _ in range(5):
    loss = criterion(model(x), torch.zeros(2, dtype=torch.long).cuda())
    loss.backward(retain_graph=True)
    pruner.update_ema_and_mask_grad()

# pruning 직전 EMA 상태 확인
import torch.nn as nn
for name, m in model.named_modules():
    if isinstance(m, nn.Linear) and hasattr(m, 'grad_ema') and 'blocks.0' in name:
        print(f"{name}: grad_ema mean={m.grad_ema.mean():.6f}, max={m.grad_ema.max():.6f}")

# pruning 실행
loss = criterion(model(x), torch.zeros(2, dtype=torch.long).cuda())
loss.backward(retain_graph=True)
pruner.step_pruning(loss, current_epoch=5, total_epochs=30)

with torch.no_grad():
    out = model(x)
    print(f"✅ Forward OK: {out.shape}")