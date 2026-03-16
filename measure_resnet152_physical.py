

import torch
import torch.nn as nn
import torch_pruning as tp
import os
import sys
import gc

# ------------------------------
# 1️⃣ 설정
# ------------------------------
sys.path.append("./src")
from models.resnet import get_resnet

DEVICE = "cuda"

# 🔥 여기만 바꿔가며 사용
# CKPT_PATH = "./exp/checkpoints/resnet152_pdt_final.pth"
CKPT_PATH = "./exp/checkpoints/resnet152_DENSE_BASELINE.pth"

MODEL_NAME = "resnet152"

# ------------------------------
# 2️⃣ 모델 로드
# ------------------------------
print(f"\n🔄 Loading {MODEL_NAME} from: {os.path.basename(CKPT_PATH)}")

model = get_resnet(MODEL_NAME, num_classes=100)
state_dict = torch.load(CKPT_PATH, map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.to(DEVICE)
model.eval()

# ------------------------------
# 3️⃣ Physical Pruning (mask 전부 반영)
# ------------------------------
print("🔄 Building Dependency Graph...")
example_inputs = torch.randn(1,3,32,32).to(DEVICE)
DG = tp.DependencyGraph().build_dependency(model, example_inputs)

total_pruned = 0

for name, m in model.named_modules():

    if isinstance(m, nn.Conv2d) and hasattr(m, "mask"):

        prune_idx = torch.where(m.mask == 0)[0].tolist()

        # mask=0이면 전부 물리적 제거
        if len(prune_idx) > 0 and len(prune_idx) < m.out_channels:

            print(f"Pruning {name} | {len(prune_idx)} / {m.out_channels}")

            group = DG.get_pruning_group(
                m,
                tp.prune_conv_out_channels,
                prune_idx
            )

            if DG.check_pruning_group(group):
                group.prune()
                total_pruned += len(prune_idx)

if total_pruned > 0:
    print(f"✅ Physical pruning applied. Total pruned channels: {total_pruned}")
else:
    print("ℹ️ No pruning applied (Dense model or no masked channels).")

# ------------------------------
# 4️⃣ Memory Cleanup
# ------------------------------
del DG
del example_inputs
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print(f"✅ Physical pruning & Memory Cleanup done: {os.path.basename(CKPT_PATH)}")

# ------------------------------
# 5️⃣ 모델 통계
# ------------------------------
total_params = sum(p.numel() for p in model.parameters())

temp_path = "./temp_model.pth"
torch.save(model.state_dict(), temp_path)
size_mb = os.path.getsize(temp_path) / (1024**2)
os.remove(temp_path)

# ------------------------------
# 6️⃣ Inference Peak 측정
# ------------------------------
dummy = torch.randn(1,3,32,32).to(DEVICE)

with torch.inference_mode():

    # warmup
    for _ in range(10):
        _ = model(dummy)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    for _ in range(50):
        _ = model(dummy)
        torch.cuda.synchronize()

peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
peak_resv  = torch.cuda.max_memory_reserved() / (1024**2)

# ------------------------------
# 7️⃣ 출력
# ------------------------------
print("\n" + "="*55)
print(f"📊 Detailed Memory Analysis: {MODEL_NAME}")
print(f" Path: {os.path.basename(CKPT_PATH)}")
print("-"*55)
print(" [Model Stats]")
print(f"   - Total Params      : {total_params:,}")
print(f"   - Model File Size   : {size_mb:.2f} MB")
print("-"*55)
print(" [VRAM Stats - Inference Mode]")
print(f"   - Peak Allocated    : {peak_alloc:.2f} MB")
print(f"   - Peak Reserved     : {peak_resv:.2f} MB")
print("="*55 + "\n")