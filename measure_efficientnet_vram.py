import torch
import torch.nn as nn
import torch_pruning as tp
from torchvision.models import efficientnet_b0

device = torch.device("cuda")

# ------------------------
# 1. 모델 로드
# ------------------------
model = efficientnet_b0(num_classes=100).to(device)

ckpt = torch.load("./exp/checkpoints/efficientnet_b0_pdt_final.pth", map_location=device)

if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
else:
    model.load_state_dict(ckpt, strict=False)

model.eval()

example_inputs = torch.randn(1, 3, 32, 32).to(device)
DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

# ------------------------
# 2. Physical pruning
# ------------------------
for module in model.modules():

    if not hasattr(module, "mask"):
        continue

    # depthwise conv는 제외
    if isinstance(module, nn.Conv2d):

        if module.groups == module.in_channels:
            continue  # depthwise skip

        prune_idx = torch.where(module.mask == 0)[0].tolist()

        if len(prune_idx) == 0:
            continue

        if len(prune_idx) >= module.out_channels:
            continue

        group = DG.get_pruning_group(
            module,
            tp.prune_conv_out_channels,
            prune_idx
        )
        group.prune()

# ------------------------
# 3. 저장 및 측정
# ------------------------
torch.save(model.state_dict(), "efficientnet_b0_physically_pruned.pth")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total Params After Physical Pruning: {total_params:,}")

import os
size_mb = os.path.getsize("efficientnet_b0_physically_pruned.pth") / (1024**2)
print(f"Compressed Model File Size: {size_mb:.2f} MB")

torch.cuda.reset_peak_memory_stats()
dummy = torch.randn(1, 3, 32, 32).to(device)
with torch.no_grad():
    _ = model(dummy)

peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
print(f"Peak VRAM After Physical Pruning: {peak_mem:.2f} MB")
