# import torch
# import torch.nn as nn
# import torch_pruning as tp
# import os
# import sys

# sys.path.append(os.path.abspath("."))

# from src.models import get_model
# from src.utils.config_loader import load_config

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ================================
# # 1️⃣ 모델 설정
# # ================================
# config, _ = load_config()
# config['model']['name'] = 'resnet152'
# config['model']['num_classes'] = 100

# model = get_model(config['model']).to(device)

# # ================================
# # 2️⃣ 체크포인트 경로 (여기만 바꾸면 됨)
# # ================================
# # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep160_sp55.70.pth"
# # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep130_sp47.03.pth"
# # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep145_sp51.72.pth"
# # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep115_sp42.75.pth"
# # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep100_sp37.55.pth"
# # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep70_sp26.52.pth"
# # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep55_sp23.36.pth"
# # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep40_sp23.35.pth"
# ckpt_path = "./exp/checkpoints/resnet152_pdt_ep180_sp66.61.pth"

# # resnet152_DENSE_BASELINE.pth
# # resnet152_pdt_ep40_sp25.32.pth
# # resnet152_pdt_ep140_sp58.52.pth 정확도 끝
# # resnet152_pdt_ep180_sp66.61.pth


# checkpoint = torch.load(ckpt_path, map_location=device)

# if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
#     state_dict = checkpoint["model_state_dict"]
# else:
#     state_dict = checkpoint

# model.load_state_dict(state_dict, strict=False)
# model.eval()

# # ================================
# # 3️⃣ Physical Pruning
# # ================================
# example_inputs = torch.randn(1, 3, 32, 32).to(device)
# DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

# for module in model.modules():

#     if not hasattr(module, "mask"):
#         continue

#     if isinstance(module, nn.Conv2d):

#         prune_idx = torch.where(module.mask == 0)[0].tolist()

#         if len(prune_idx) == 0:
#             continue

#         if len(prune_idx) >= module.out_channels:
#             continue

#         group = DG.get_pruning_group(
#             module,
#             tp.prune_conv_out_channels,
#             prune_idx
#         )
#         group.prune()

# print("✅ Physical pruning done.")

# # ================================
# # 4️⃣ 파라미터 수
# # ================================
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total Params After Physical Pruning: {total_params:,}")

# # ================================
# # 5️⃣ 파일 크기
# # ================================
# save_path = "./exp/checkpoints/resnet152_physically_pruned.pth"
# torch.save(model.state_dict(), save_path)

# size_mb = os.path.getsize(save_path) / (1024**2)
# print(f"Compressed Model File Size: {size_mb:.2f} MB")

# # ================================
# # 6️⃣ Peak VRAM 측정 (Inference)
# # ================================
# torch.cuda.reset_peak_memory_stats(device)

# dummy = torch.randn(1, 3, 32, 32).to(device)
# with torch.no_grad():
#     _ = model(dummy)

# peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
# print(f"Peak VRAM After Physical Pruning: {peak_mem:.2f} MB")



# import torch
# import torch.nn as nn
# import torch_pruning as tp
# import os
# import sys
# import numpy as np

# sys.path.append(os.path.abspath("."))

# from src.models import get_model
# from src.utils.config_loader import load_config

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ================================
# # 1️⃣ 모델 설정 (ResNet-152로 변경)
# # ================================
# config, _ = load_config()
# config['model']['name'] = 'resnet152'  # ResNet-152 설정
# config['model']['num_classes'] = 100

# model = get_model(config['model']).to(device)

# # ================================
# # 2️⃣ 체크포인트 로드
# # ================================
# # ResNet-152 용 체크포인트 경로로 수정하여 사용하세요.
# ckpt_path = "./exp/checkpoints/resnet152_pdt_final.pth" 
# # ckpt_path = "./exp/checkpoints/resnet152_DENSE_BASELINE.pth" 

# ######## 실험 다시 함
# # resnet152_DENSE_BASELINE.pth
# # resnet152_pdt_ep40_sp25.66.pth
# # resnet152_pdt_ep60_sp28.19.pth
# # resnet152_pdt_ep80_sp38.36.pth
# # resnet152_pdt_ep100_sp45.40.pth
# # resnet152_pdt_ep120_sp51.34.pth
# # resnet152_pdt_ep140_sp56.39.pth
# # resnet152_pdt_ep160_sp58.57.pth
# # resnet152_pdt_ep180_sp65.22.pth
# # resnet152_pdt_ep200_sp67.24.pth
# # resnet152_pdt_final.pth







# if not os.path.exists(ckpt_path):
#     print(f"❌ File not found: {ckpt_path}")
#     # 파일이 없더라도 구조 확인을 위해 중단하지 않으려면 아래 sys.exit()을 주석 처리하세요.
#     sys.exit()

# checkpoint = torch.load(ckpt_path, map_location=device)
# state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

# model.load_state_dict(state_dict, strict=False)
# model.eval()

# # ================================
# # 3️⃣ Physical Pruning (실제 필터 삭제)
# # ================================
# # ResNet-152는 입력 사이즈에 따라 메모리 점유가 크므로 예시 입력 유지
# example_inputs = torch.randn(1, 3, 32, 32).to(device)
# DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

# print(f"🔄 Analyzing Dependency Graph for ResNet-152...")

# for module in model.modules():
#     if not hasattr(module, "mask"):
#         continue

#     if isinstance(module, nn.Conv2d):
#         prune_idx = torch.where(module.mask == 0)[0].tolist()
#         if len(prune_idx) == 0: continue
        
#         # 실제 채널 수보다 많이 잘리는 오류 방지
#         if len(prune_idx) >= module.out_channels:
#             # 안전을 위해 최대 90%까지만 프루닝하거나 스킵
#             prune_idx = prune_idx[:int(module.out_channels * 0.9)]
            
#         if len(prune_idx) == 0: continue

#         group = DG.get_pruning_group(module, tp.prune_conv_out_channels, prune_idx)
#         group.prune()

# print(f"✅ Physical pruning done: {os.path.basename(ckpt_path)}")

# # 파라미터 수 및 실제 파일 크기 계산
# total_params = sum(p.numel() for p in model.parameters())
# save_path = "./exp/checkpoints/resnet152_temp_phys.pth"


# # # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep40_sp23.35.pth"
# # # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep70_sp26.52.pth"
# # # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep100_sp37.55.pth"
# # # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep130_sp47.03.pth"
# # # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep160_sp55.70.pth"


# # # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep160_sp55.70.pth"
# # # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep130_sp47.03.pth"
# # # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep145_sp51.72.pth"
# # # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep115_sp42.75.pth"
# # # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep100_sp37.55.pth"
# # # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep70_sp26.52.pth"
# # # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep55_sp23.36.pth"
# # # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep40_sp23.35.pth"
# # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep180_sp66.61.pth"

# torch.save(model.state_dict(), save_path)
# size_mb = os.path.getsize(save_path) / (1024**2)
# if os.path.exists(save_path): os.remove(save_path)

# # ================================
# # 4️⃣ 상세 VRAM 측정 (Inference Mode)
# # ================================

# # 측정 전 메모리 완전 초기화 및 동기화 (슬랙 피드백 반영)
# torch.cuda.empty_cache()
# torch.cuda.synchronize() 
# torch.cuda.reset_peak_memory_stats(device) 

# dummy = torch.randn(1, 3, 32, 32).to(device)

# # torch.inference_mode() 사용 (슬랙 피드백 반영)
# with torch.inference_mode():
#     # Warm-up
#     for _ in range(10):
#         _ = model(dummy)
    
#     torch.cuda.synchronize()
    
#     # 실제 추론 수행 (메모리 트레이스)
#     for _ in range(50):
#         _ = model(dummy)
    
#     torch.cuda.synchronize() 

# # 핵심 메모리 지표 추출
# curr_alloc = torch.cuda.memory_allocated(device) / (1024**2)
# curr_resv  = torch.cuda.memory_reserved(device) / (1024**2)
# peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**2)
# peak_resv  = torch.cuda.max_memory_reserved(device) / (1024**2)

# # ================================
# # 5️⃣ 결과 출력
# # ================================
# print("\n" + "="*50)
# print(f"📊 Detailed Memory Analysis: ResNet-152")
# print(f" Path: {os.path.basename(ckpt_path)}")
# print("-" * 50)
# print(f" [Model Stats]")
# print(f"   - Total Params      : {total_params:,}")
# print(f"   - Model File Size   : {size_mb:.2f} MB")
# print("-" * 50)
# print(f" [VRAM Stats - Inference Mode]")
# print(f"   - Current Allocated : {curr_alloc:.2f} MB")
# print(f"   - Current Reserved  : {curr_resv:.2f} MB")
# print(f"   - Peak Allocated    : {peak_alloc:.2f} MB")
# print(f"   - Peak Reserved     : {peak_resv:.2f} MB")
# print("="*50 + "\n")



# import torch
# import torch.nn as nn
# import torch_pruning as tp
# import sys
# import gc

# sys.path.append("./src")
# from models.resnet import get_resnet

# DEVICE = "cuda"
# MODEL_NAME = "resnet152"
# CKPT_PATH = "./exp/checkpoints/resnet152_DENSE_BASELINE.pth"

# model = get_resnet(MODEL_NAME, num_classes=100)
# model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"), strict=False)
# model.to(DEVICE)
# model.eval()

# example_inputs = torch.randn(1,3,32,32).to(DEVICE)
# DG = tp.DependencyGraph().build_dependency(model, example_inputs)

# print("🔥 Aggressive Width Pruning (50%)")

# for name, m in model.named_modules():

#     # Bottleneck의 conv3만 공격적으로 prune
#     if isinstance(m, nn.Conv2d) and "conv3" in name:

#         out_channels = m.out_channels
#         prune_idx = list(range(out_channels // 2))  # 절반 제거

#         print(f"Pruning {name} | {len(prune_idx)} / {out_channels}")

#         group = DG.get_pruning_group(
#             m,
#             tp.prune_conv_out_channels,
#             prune_idx
#         )

#         if DG.check_pruning_group(group):
#             group.prune()

# print("✅ Width pruning complete.")

# # VRAM 측정
# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats()

# model.train()

# dummy = torch.randn(32,3,32,32).to(DEVICE)
# out = model(dummy)
# loss = out.sum()
# loss.backward()

# torch.cuda.synchronize()

# print("\n🔥 Training Peak After Width Pruning")
# print("Peak Allocated:",
#       torch.cuda.max_memory_allocated()/1024**2, "MB")
# print("Peak Reserved:",
#       torch.cuda.max_memory_reserved()/1024**2, "MB")

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