# import torch
# import os
# import sys

# # 프로젝트 루트 경로 추가
# sys.path.append(os.path.abspath("."))

# from src.models import get_model

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ----------------------------
# # 1️⃣ config 직접 만들어주기
# # ----------------------------
# config = {
#     "model": {
#         "name": "mobilenet_v2",
#         "num_classes": 100
#     }
# }

# # ----------------------------
# # 2️⃣ 모델 생성 (main.py와 동일 방식)
# # ----------------------------
# model = get_model(config["model"]).to(device)

# # ----------------------------
# # 3️⃣ 체크포인트 로드
# # ----------------------------
# ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_final.pth"
# state_dict = torch.load(ckpt_path, map_location=device)

# # 만약 저장할 때 dict 형태였다면:
# if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
#     state_dict = state_dict["model_state_dict"]

# model.load_state_dict(state_dict, strict=False)

# # ----------------------------
# # 4️⃣ 파라미터 수 계산
# # ----------------------------
# total_params = sum(p.numel() for p in model.parameters())

# # ----------------------------
# # 5️⃣ 파일 저장 후 크기 측정
# # ----------------------------
# save_path = "./exp/checkpoints/mobilenet_v2_physical.pth"
# torch.save(model.state_dict(), save_path)

# file_size_mb = os.path.getsize(save_path) / (1024**2)

# print("✅ Physical pruning done.")
# print(f"Total Params After Physical Pruning: {total_params:,}")
# print(f"Compressed Model File Size: {file_size_mb:.2f} MB")


# # Inference VRAM 측정
# model.eval()
# dummy = torch.randn(1, 3, 32, 32).to(device)

# torch.cuda.reset_peak_memory_stats(device)
# with torch.no_grad():
#     _ = model(dummy)

# peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)
# print(f"Peak VRAM After Physical Pruning (Inference): {peak_vram:.2f} MB")


# import torch
# import os
# import sys

# # -------------------------------------------------
# # 0️⃣ 프로젝트 루트 경로 추가
# # -------------------------------------------------
# sys.path.append(os.path.abspath("."))

# from src.models import get_model

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # -------------------------------------------------
# # 1️⃣ config 정의 (main.py 방식과 동일)
# # -------------------------------------------------
# config = {
#     "model": {
#         "name": "mobilenet_v2",
#         "num_classes": 100
#     }
# }

# # -------------------------------------------------
# # 2️⃣ 모델 생성
# # -------------------------------------------------
# model = get_model(config["model"]).to(device)

# # -------------------------------------------------
# # 3️⃣ 체크포인트 로드
# # -------------------------------------------------
# ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_final.pth"
# state_dict = torch.load(ckpt_path, map_location=device)

# if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
#     state_dict = state_dict["model_state_dict"]

# model.load_state_dict(state_dict, strict=False)

# print("✅ Checkpoint loaded.")

# # -------------------------------------------------
# # 4️⃣ 🔥 Auxiliary tensor 제거 (VERY IMPORTANT)
# # -------------------------------------------------
# removed = 0
# for name, module in model.named_modules():
#     for attr in ["mask", "grad_ema", "hessian_score"]:
#         if hasattr(module, attr):
#             delattr(module, attr)
#             removed += 1

# print(f"🧹 Removed auxiliary tensors: {removed}")

# # -------------------------------------------------
# # 5️⃣ 파라미터 수 계산
# # -------------------------------------------------
# total_params = sum(p.numel() for p in model.parameters())

# # -------------------------------------------------
# # 6️⃣ Clean 모델 저장 후 파일 크기 측정
# # -------------------------------------------------
# save_path = "./exp/checkpoints/mobilenet_v2_physical_clean.pth"
# torch.save(model.state_dict(), save_path)

# file_size_mb = os.path.getsize(save_path) / (1024**2)

# print("✅ Clean Physical Model Saved.")
# print(f"Total Params After Physical Pruning: {total_params:,}")
# print(f"Compressed Model File Size: {file_size_mb:.2f} MB")

# # -------------------------------------------------
# # 7️⃣ Inference VRAM 측정
# # -------------------------------------------------
# model.eval()
# dummy = torch.randn(1, 3, 32, 32).to(device)

# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats(device)

# with torch.no_grad():
#     _ = model(dummy)

# peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)

# print(f"Peak VRAM After Physical Pruning (Inference): {peak_vram:.2f} MB")


import torch
import torch.nn as nn
import torch_pruning as tp
import os
import sys

sys.path.append(os.path.abspath("."))

from src.models import get_model
from src.utils.config_loader import load_config

device = torch.device("cuda")

# -------------------------------------------------
# 1️⃣ YAML 로드
# -------------------------------------------------
config, _ = load_config()
config['model']['name'] = 'mobilenet_v2'

# -------------------------------------------------
# 2️⃣ 모델 생성
# -------------------------------------------------
model = get_model(config['model']).to(device)

# -------------------------------------------------
# 3️⃣ 체크포인트 로드
# -------------------------------------------------
# ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_final.pth"

ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep295_sp36.86.pth" # 예시 경로



# ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep200_sp91.68.pth"
# ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep40_sp10.10.pth" # 예시 경로
# ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep60_sp20.31.pth" # 예시 경로
# ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep80_sp28.55.pth" # 예시 경로
# ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep100_sp36.05.pth" # 예시 경로
# ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep120_sp47.38.pth" # 예시 경로
# ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep140_sp89.68.pth" # 예시 경로
# ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep160_sp89.68.pth" # 예시 경로
# ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep180_sp89.68.pth" # 예시 경로
# ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep200_sp91.68.pth" # 예시 경로
checkpoint = torch.load(ckpt_path, map_location=device)




# 만약 dict 안에 model_state_dict가 있으면 그걸 쓰고
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict, strict=False)


# -------------------------------------------------
# 4️⃣ Dependency Graph 생성
# -------------------------------------------------
example_inputs = torch.randn(1, 3, 32, 32).to(device)
DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

# -------------------------------------------------
# 5️⃣ 🔥 실제 채널 제거
# -------------------------------------------------
for module in model.modules():

    if not hasattr(module, "mask"):
        continue

    if isinstance(module, nn.Conv2d):

        prune_idx = torch.where(module.mask == 0)[0].tolist()

        if len(prune_idx) == 0:
            continue

        if len(prune_idx) >= module.out_channels:
            continue

        # 🔥 Depthwise 보호 로직
        if module.groups == module.in_channels and module.in_channels == module.out_channels:
            pruning_fn = tp.prune_conv_in_channels
        else:
            pruning_fn = tp.prune_conv_out_channels

        group = DG.get_pruning_group(
            module,
            pruning_fn,
            prune_idx
        )
        group.prune()

# -------------------------------------------------
# 6️⃣ Auxiliary tensor 제거
# -------------------------------------------------
for name, module in model.named_modules():
    for attr in ["mask", "grad_ema", "hessian_score"]:
        if hasattr(module, attr):
            delattr(module, attr)

# -------------------------------------------------
# 7️⃣ 저장
# -------------------------------------------------
save_path = "./exp/checkpoints/mobilenet_v2_physically_pruned.pth"
torch.save(model.state_dict(), save_path)

print("✅ Physical pruning done.")

# -------------------------------------------------
# 8️⃣ 파라미터 수
# -------------------------------------------------
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Params After Physical Pruning: {total_params:,}")

# -------------------------------------------------
# 9️⃣ 파일 크기
# -------------------------------------------------
size_mb = os.path.getsize(save_path) / (1024**2)
print(f"Compressed Model File Size: {size_mb:.2f} MB")

# -------------------------------------------------
# 🔟 실제 VRAM 측정
# -------------------------------------------------
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

dummy = torch.randn(1, 3, 32, 32).to(device)
with torch.no_grad():
    _ = model(dummy)

peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
print(f"Peak VRAM After Physical Pruning: {peak_mem:.2f} MB")


