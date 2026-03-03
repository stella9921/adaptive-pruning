import torch
import torch.nn as nn
import torch_pruning as tp
from src.models import get_model
from src.utils.config_loader import load_config
import os

device = torch.device("cuda")

# ----------------------------
# 1️⃣ YAML 로드
# ----------------------------
config, _ = load_config()
config['model']['name'] = 'efficientnet_b0'

# ----------------------------
# 2️⃣ 모델 생성
# ----------------------------
model = get_model(config['model']).to(device)

# ----------------------------
# 3️⃣ 체크포인트 로드 (final 사용)
# ----------------------------
ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_final.pth"
checkpoint = torch.load(ckpt_path, map_location=device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict, strict=False)

model.eval()

# ----------------------------
# 4️⃣ Dependency Graph 생성
# ----------------------------
example_inputs = torch.randn(1, 3, 32, 32).to(device)
DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

# ----------------------------
# 5️⃣ Physical Pruning
# ----------------------------
for module in model.modules():

    if not hasattr(module, "mask"):
        continue

    if isinstance(module, nn.Conv2d):

        # depthwise conv skip
        if module.groups == module.in_channels:
            continue

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

print("✅ Physical pruning done.")

# ----------------------------
# 6️⃣ 파라미터 수 확인
# ----------------------------
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Params After Physical Pruning: {total_params:,}")

# ----------------------------
# 7️⃣ 파일 크기 측정
# ----------------------------
save_path = "./exp/checkpoints/efficientnet_b0_physically_pruned.pth"
torch.save(model.state_dict(), save_path)

size_mb = os.path.getsize(save_path) / (1024**2)
print(f"Compressed Model File Size: {size_mb:.2f} MB")

# ----------------------------
# 8️⃣ 실제 VRAM 사용량 측정
# ----------------------------
torch.cuda.reset_peak_memory_stats(device)

dummy = torch.randn(1, 3, 32, 32).to(device)
with torch.no_grad():
    _ = model(dummy)

peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
print(f"Peak VRAM After Physical Pruning: {peak_mem:.2f} MB")
