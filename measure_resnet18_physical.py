# # import torch
# # import torch.nn as nn
# # import torch_pruning as tp
# # import os
# # import sys

# # sys.path.append(os.path.abspath("."))

# # from src.models import get_model
# # from src.utils.config_loader import load_config

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # ================================
# # # 1️⃣ 모델 설정 (ResNet-18)
# # # ================================
# # config, _ = load_config()
# # config['model']['name'] = 'resnet18'  # ResNet-18로 변경
# # config['model']['num_classes'] = 100

# # model = get_model(config['model']).to(device)

# # # ================================
# # # 2️⃣ 체크포인트 경로 (ResNet-18용 체크포인트 지정)
# # # ================================
# # # ResNet-18용 체크포인트 파일명을 여기에 입력하세요.
# # # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep40_sp10.80.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep60_sp19.96.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep80_sp26.82.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep100_sp32.69.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep120_sp41.31.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep140_sp58.10.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep160_sp58.10.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep180_sp58.10.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep200_sp58.10.pth" # 예시 경로






# # if not os.path.exists(ckpt_path):
# #     print(f"❌ File not found: {ckpt_path}")
# #     sys.exit()

# # checkpoint = torch.load(ckpt_path, map_location=device)

# # if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
# #     state_dict = checkpoint["model_state_dict"]
# # else:
# #     state_dict = checkpoint

# # model.load_state_dict(state_dict, strict=False)
# # model.eval()

# # # ================================
# # # 3️⃣ Physical Pruning
# # # ================================
# # # ResNet-18 구조에 맞는 의존성 그래프 생성
# # example_inputs = torch.randn(1, 3, 32, 32).to(device)
# # DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

# # for module in model.modules():
# #     if not hasattr(module, "mask"):
# #         continue

# #     if isinstance(module, nn.Conv2d):
# #         # 마스크가 0인 인덱스(제거 대상) 추출
# #         prune_idx = torch.where(module.mask == 0)[0].tolist()

# #         if len(prune_idx) == 0:
# #             continue

# #         # 모든 채널이 잘리는 것 방지
# #         if len(prune_idx) >= module.out_channels:
# #             continue

# #         # 물리적 제거 실행
# #         group = DG.get_pruning_group(
# #             module,
# #             tp.prune_conv_out_channels,
# #             prune_idx
# #         )
# #         group.prune()

# # print("✅ ResNet-18 Physical pruning done.")

# # # ================================
# # # 4️⃣ 파라미터 수 확인
# # # ================================
# # total_params = sum(p.numel() for p in model.parameters())
# # print(f"Total Params After Physical Pruning (ResNet-18): {total_params:,}")

# # # ================================
# # # 5️⃣ 파일 크기 확인
# # # ================================
# # save_path = "./exp/checkpoints/resnet18_physically_pruned.pth"
# # torch.save(model.state_dict(), save_path)

# # size_mb = os.path.getsize(save_path) / (1024**2)
# # print(f"Compressed Model File Size: {size_mb:.2f} MB")

# # # ================================
# # # 6️⃣ Peak VRAM 측정 (Inference)
# # # ================================
# # torch.cuda.reset_peak_memory_stats(device)

# # dummy = torch.randn(1, 3, 32, 32).to(device)
# # with torch.no_grad():
# #     _ = model(dummy)

# # peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
# # print(f"Peak VRAM After Physical Pruning: {peak_mem:.2f} MB")


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
# # 1️⃣ 모델 설정
# # ================================
# config, _ = load_config()
# config['model']['name'] = 'resnet18'
# config['model']['num_classes'] = 100

# model = get_model(config['model']).to(device)

# # ================================
# # 2️⃣ 체크포인트 경로 설정
# # ================================
# # 분석할 체크포인트 경로를 선택하세요.

# # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep40_sp10.80.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep60_sp19.96.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep80_sp26.82.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep100_sp32.69.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep120_sp41.31.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep140_sp58.10.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep160_sp58.10.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep180_sp58.10.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep200_sp58.10.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/resnet18_DENSE_BASELINE.pth" # 예시 경로
# ckpt_path = "./exp/checkpoints/resnet18_pdt_ep40_sp11.06.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep80_sp26.20.pth" # 예시 경로






# if not os.path.exists(ckpt_path):
#     print(f"❌ File not found: {ckpt_path}")
#     sys.exit()

# checkpoint = torch.load(ckpt_path, map_location=device)
# state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

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
#         if len(prune_idx) == 0: continue
#         if len(prune_idx) >= module.out_channels: continue

#         group = DG.get_pruning_group(module, tp.prune_conv_out_channels, prune_idx)
#         group.prune()

# print(f"✅ Physical pruning done for: {os.path.basename(ckpt_path)}")

# # ================================
# # 4️⃣ 파라미터 수 & 파일 크기
# # ================================
# total_params = sum(p.numel() for p in model.parameters())
# save_path = "./exp/checkpoints/resnet18_temp_phys.pth"
# torch.save(model.state_dict(), save_path)
# size_mb = os.path.getsize(save_path) / (1024**2)
# if os.path.exists(save_path): os.remove(save_path) # 측정 후 임시파일 삭제

# # ================================
# # 5️⃣ Peak & Average VRAM 측정 (Inference)
# # ================================
# # 메모리 통계 초기화
# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats(device)

# dummy = torch.randn(1, 3, 32, 32).to(device)
# mems = []

# with torch.no_grad():
#     # Warm-up (GPU 캐시 및 초기화를 위해 10번 선행 실행)
#     for _ in range(10):
#         _ = model(dummy)
    
#     # 실제 샘플링 루프 (50번 반복하며 평균 측정)
#     for _ in range(50):
#         _ = model(dummy)
#         # memory_allocated(): 현재 GPU에 실제로 '할당되어 사용 중인' 메모리
#         mems.append(torch.cuda.memory_allocated(device) / (1024**2))

# peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)
# avg_vram = np.mean(mems)

# # ================================
# # 6️⃣ 결과 출력
# # ================================
# print("\n" + "="*40)
# print(f"📊 Physical Pruning Analysis: {os.path.basename(ckpt_path)}")
# print("-" * 40)
# print(f" - Total Params  : {total_params:,}")
# print(f" - Model Size   : {size_mb:.2f} MB")
# print(f" - Peak VRAM    : {peak_vram:.2f} MB")
# print(f" - Average VRAM : {avg_vram:.2f} MB")
# print("="*40 + "\n")


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
# # 1️⃣ 모델 설정
# # ================================
# config, _ = load_config()
# config['model']['name'] = 'resnet18'
# config['model']['num_classes'] = 100

# model = get_model(config['model']).to(device)

# # ================================
# # 2️⃣ 체크포인트 로드
# # ================================
# # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep40_sp11.06.pth" 



# # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep40_sp10.80.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep60_sp19.96.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep80_sp26.82.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep100_sp32.69.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep120_sp41.31.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep140_sp58.10.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep160_sp58.10.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep180_sp58.10.pth" # 예시 경로
# ckpt_path = "./exp/checkpoints/resnet18_pdt_ep200_sp58.10.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/resnet18_DENSE_BASELINE.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep40_sp11.06.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/resnet18_pdt_ep80_sp26.20.pth" # 예시 경로

# if not os.path.exists(ckpt_path):
#     print(f"❌ File not found: {ckpt_path}")
#     sys.exit()

# checkpoint = torch.load(ckpt_path, map_location=device)
# state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

# model.load_state_dict(state_dict, strict=False)
# model.eval()

# # ================================
# # 3️⃣ Physical Pruning (실제 필터 삭제)
# # ================================
# example_inputs = torch.randn(1, 3, 32, 32).to(device)
# DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

# for module in model.modules():
#     if not hasattr(module, "mask"):
#         continue

#     if isinstance(module, nn.Conv2d):
#         prune_idx = torch.where(module.mask == 0)[0].tolist()
#         if len(prune_idx) == 0: continue
#         if len(prune_idx) >= module.out_channels: continue

#         group = DG.get_pruning_group(module, tp.prune_conv_out_channels, prune_idx)
#         group.prune()

# print(f"✅ Physical pruning done: {os.path.basename(ckpt_path)}")

# # 파라미터 수 및 실제 파일 크기 계산
# total_params = sum(p.numel() for p in model.parameters())
# save_path = "./exp/checkpoints/resnet18_temp_phys.pth"



# torch.save(model.state_dict(), save_path)
# size_mb = os.path.getsize(save_path) / (1024**2)
# if os.path.exists(save_path): os.remove(save_path)

# # ================================
# # 4️⃣ 상세 VRAM 측정 (Inference Mode)
# # ================================

# # [슬랙 피드백 반영] 측정 전 메모리 완전 초기화 및 동기화
# torch.cuda.empty_cache()
# torch.cuda.synchronize() # GPU 작업 완료 대기
# torch.cuda.reset_peak_memory_stats(device) # 피크 통계 초기화

# dummy = torch.randn(1, 3, 32, 32).to(device)

# # [슬랙 피드백 반영] torch.inference_mode() 사용
# with torch.inference_mode():
#     # Warm-up (초기화 오버헤드 제거)
#     for _ in range(10):
#         _ = model(dummy)
    
#     torch.cuda.synchronize()
    
#     # 실제 추론 수행 (메모리 트레이스)
#     for _ in range(50):
#         _ = model(dummy)
    
#     torch.cuda.synchronize() # 모든 연산이 끝날 때까지 대기

# # [슬랙 피드백 반영] 4가지 핵심 메모리 지표 추출
# # 1. Allocated: 실제 텐서가 점유 중인 메모리
# # 2. Reserved: PyTorch 캐싱 할당자가 확보한 전체 메모리 (Pool)
# curr_alloc = torch.cuda.memory_allocated(device) / (1024**2)
# curr_resv  = torch.cuda.memory_reserved(device) / (1024**2)
# peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**2)
# peak_resv  = torch.cuda.max_memory_reserved(device) / (1024**2)

# # ================================
# # 5️⃣ 결과 출력 (Table 2 보완용)
# # ================================
# print("\n" + "="*50)
# print(f"📊 Detailed Memory Analysis: {os.path.basename(ckpt_path)}")
# print("-" * 50)
# print(f" [Model Stats]")
# print(f"  - Total Params      : {total_params:,}")
# print(f"  - Model File Size   : {size_mb:.2f} MB")
# print("-" * 50)
# print(f" [VRAM Stats - Inference Mode]")
# print(f"  - Current Allocated : {curr_alloc:.2f} MB (Actual Tensors)")
# print(f"  - Current Reserved  : {curr_resv:.2f} MB (Caching Pool)")
# print(f"  - Peak Allocated    : {peak_alloc:.2f} MB (Max Actual Use)")
# print(f"  - Peak Reserved     : {peak_resv:.2f} MB (Max Pool Size)")
# print("="*50 + "\n")




# import torch
# import torch.nn as nn
# import torch_pruning as tp
# import os
# import sys
# import numpy as np
# import gc

# sys.path.append(os.path.abspath("."))

# from src.models import get_model
# from src.utils.config_loader import load_config

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ================================
# # 1️⃣ 모델 설정
# # ================================
# config, _ = load_config()
# config['model']['name'] = 'resnet18'
# config['model']['num_classes'] = 100

# model = get_model(config['model']).to(device)

# # ================================
# # 2️⃣ 체크포인트 로드
# # ================================
# ckpt_path = "./exp/checkpoints/resnet18_pdt_final.pth" 
# # ckpt_path = "./exp/checkpoints/resnet18_DENSE_BASELINE.pth" 

# if not os.path.exists(ckpt_path):
#     print(f"❌ File not found: {ckpt_path}")
#     sys.exit()

# checkpoint = torch.load(ckpt_path, map_location=device)
# state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

# model.load_state_dict(state_dict, strict=False)
# model.eval()

# # ================================
# # 3️⃣ Physical Pruning (Revised)
# # ================================

# import torch_pruning as tp

# example_inputs = torch.randn(1, 3, 32, 32).to(device)
# DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

# for name, m in model.named_modules():
#     if isinstance(m, nn.Conv2d) and hasattr(m, "mask"):

#         prune_idx = torch.where(m.mask == 0)[0].tolist()

#         if len(prune_idx) > 0 and len(prune_idx) < m.out_channels:

#             print(f"Pruning {name} | {len(prune_idx)} / {m.out_channels}")

#             group = DG.get_pruning_group(
#                 m,
#                 tp.prune_conv_out_channels,
#                 prune_idx
#             )

#             if DG.check_pruning_group(group):
#                 group.prune()

# torch.cuda.empty_cache()
# torch.cuda.synchronize()

# print(f"✅ Physical pruning & Memory Cleanup done: {os.path.basename(ckpt_path)}")

# # 파라미터 수 확인
# total_params = sum(p.numel() for p in model.parameters())
# save_path = "./exp/checkpoints/resnet18_temp_phys.pth"
# torch.save(model.state_dict(), save_path)
# size_mb = os.path.getsize(save_path) / (1024**2)
# if os.path.exists(save_path): os.remove(save_path)

# # ================================
# # 4️⃣ 상세 VRAM 재측정 (Inference Mode)
# # ================================

# # [슬랙 피드백 반영] 측정 직전 완전 동기화 및 통계 리셋
# torch.cuda.synchronize()
# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats(device) 

# dummy = torch.randn(1, 3, 32, 32).to(device)

# with torch.inference_mode():
#     # Warm-up (모델 및 CUDA 컨텍스트 안정화)
#     for _ in range(10):
#         _ = model(dummy)
    
#     torch.cuda.synchronize()
#     # Warm-up 이후의 피크를 측정하기 위해 통계 다시 리셋
#     torch.cuda.reset_peak_memory_stats(device) 
    
#     # 실제 추론 수행
#     for _ in range(50):
#         _ = model(dummy)
#         torch.cuda.synchronize()

# # 최종 지표 추출
# curr_alloc = torch.cuda.memory_allocated(device) / (1024**2)
# curr_resv  = torch.cuda.memory_reserved(device) / (1024**2)
# peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**2)
# peak_resv  = torch.cuda.max_memory_reserved(device) / (1024**2)

# # ================================
# # 5️⃣ 결과 출력 (Table 2/Fig 3 보완용)
# # ================================
# print("\n" + "="*50)
# print(f"📊 Optimized Memory Analysis: {os.path.basename(ckpt_path)}")
# print("-" * 50)
# print(f" [Model Stats]")
# print(f"  - Total Params      : {total_params:,}")
# print(f"  - Model File Size   : {size_mb:.2f} MB")
# print("-" * 50)
# print(f" [VRAM Stats - Clean State]")
# print(f"  - Peak Allocated    : {peak_alloc:.2f} MB (Actual Tensors)")
# print(f"  - Peak Reserved     : {peak_resv:.2f} MB (Caching Pool)")
# print("="*50 + "\n")

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
CKPT_PATH = "./exp/checkpoints/resnet18_pdt_final.pth"  # 바꿔가며 사용
CKPT_PATH = "./exp/checkpoints/resnet18_DENSE_BASELINE.pth"  # 바꿔가며 사용


# ------------------------------
# 2️⃣ 모델 로드
# ------------------------------
print(f"Loading model from: {os.path.basename(CKPT_PATH)}")

model = get_resnet("resnet18", num_classes=100)
state_dict = torch.load(CKPT_PATH, map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.to(DEVICE)
model.eval()

# ------------------------------
# 3️⃣ Physical Pruning (mask 존재 시)
# ------------------------------
example_inputs = torch.randn(1,3,32,32).to(DEVICE)
DG = tp.DependencyGraph().build_dependency(model, example_inputs)

pruned_any = False

for name, m in model.named_modules():
    if isinstance(m, nn.Conv2d) and hasattr(m, "mask"):

        prune_idx = torch.where(m.mask == 0)[0].tolist()

        if len(prune_idx) > 0 and len(prune_idx) < m.out_channels:

            print(f"Pruning {name} | {len(prune_idx)} / {m.out_channels}")

            group = DG.get_pruning_group(
                m,
                tp.prune_conv_out_channels,
                prune_idx
            )

            if DG.check_pruning_group(group):
                group.prune()
                pruned_any = True

if pruned_any:
    print("✅ Physical pruning applied.")
else:
    print("ℹ️ No pruning applied (Dense model).")

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
print("\n" + "="*50)
print(f"📊 Optimized Memory Analysis: {os.path.basename(CKPT_PATH)}")
print("-"*50)
print(" [Model Stats]")
print(f"  - Total Params      : {total_params:,}")
print(f"  - Model File Size   : {size_mb:.2f} MB")
print("-"*50)
print(" [VRAM Stats - Clean State]")
print(f"  - Peak Allocated    : {peak_alloc:.2f} MB (Actual Tensors)")
print(f"  - Peak Reserved     : {peak_resv:.2f} MB (Caching Pool)")
print("="*50 + "\n")