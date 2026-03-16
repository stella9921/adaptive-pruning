# # import torch
# # import os
# # import sys

# # sys.path.append(os.path.abspath("."))

# # from src.models import get_model
# # from src.utils.config_loader import load_config

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # ----------------------------
# # # CUDA 초기화
# # # ----------------------------
# # torch.cuda.empty_cache()
# # torch.cuda.reset_peak_memory_stats()

# # # ----------------------------
# # # 모델 설정
# # # ----------------------------
# # config, _ = load_config()
# # config['model']['name'] = 'resnet152'
# # config['model']['num_classes'] = 100

# # # ----------------------------
# # # 모델 생성 (Dense)
# # # ----------------------------
# # model = get_model(config['model']).to(device)
# # model.eval()

# # # ----------------------------
# # # Params
# # # ----------------------------
# # total_params = sum(p.numel() for p in model.parameters())
# # print(f"Total Params (Dense): {total_params:,}")

# # # ----------------------------
# # # File Size
# # # ----------------------------
# # save_path = "./resnet152_dense_temp.pth"
# # torch.save(model.state_dict(), save_path)
# # file_size_mb = os.path.getsize(save_path) / (1024**2)
# # print(f"Model File Size (Dense): {file_size_mb:.2f} MB")

# # # ----------------------------
# # # Inference VRAM
# # # ----------------------------
# # torch.cuda.empty_cache()
# # torch.cuda.reset_peak_memory_stats()

# # dummy = torch.randn(1, 3, 32, 32).to(device)

# # with torch.no_grad():
# #     _ = model(dummy)

# # peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
# # print(f"Peak VRAM (Dense, Inference): {peak_vram:.2f} MB")

# # print("✅ Dense measurement finished cleanly.")


# import torch
# import torch.nn as nn
# import torch_pruning as tp
# import os
# import sys
# import numpy as np

# # 프로젝트 루트 경로 추가
# sys.path.append(os.path.abspath("."))

# from src.models import get_model
# from src.utils.config_loader import load_config

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ================================
# # 1️⃣ 모델 설정 (ResNet-152)
# # ================================
# config, _ = load_config()
# config['model']['name'] = 'resnet152' # ResNet-152로 변경
# config['model']['num_classes'] = 100

# model = get_model(config['model']).to(device)

# # ================================
# # 2️⃣ 체크포인트 경로 설정 (ResNet-152용)
# # ================================
# # 분석할 ResNet-152 체크포인트 경로를 순차적으로 입력하세요.
# # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep40_sp23.35.pth"
# # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep70_sp26.52.pth"
# # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep100_sp37.55.pth"
# # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep130_sp47.03.pth"
# # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep160_sp55.70.pth"
# ckpt_path = "./exp/checkpoints/resnet152_pdt_ep160_sp55.70.pth" # 예시 경로

# if not os.path.exists(ckpt_path):
#     print(f"❌ File not found: {ckpt_path}")
#     sys.exit()

# checkpoint = torch.load(ckpt_path, map_location=device)
# state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

# model.load_state_dict(state_dict, strict=False)
# model.eval()

# # ================================
# # 3️⃣ Physical Pruning (물리적 구조 변경)
# # ================================
# # ResNet-152는 층이 깊어 의존성 그래프 구축에 시간이 조금 더 소요될 수 있습니다.
# example_inputs = torch.randn(1, 3, 32, 32).to(device)
# DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

# for module in model.modules():
#     if not hasattr(module, "mask"):
#         continue

#     if isinstance(module, nn.Conv2d):
#         # 마스크가 0인 인덱스(제거 대상) 추출
#         prune_idx = torch.where(module.mask == 0)[0].tolist()
        
#         if len(prune_idx) == 0: continue
#         if len(prune_idx) >= module.out_channels: continue

#         # 물리적 채널 제거 및 의존성 레이어(BN 등) 자동 조정
#         group = DG.get_pruning_group(module, tp.prune_conv_out_channels, prune_idx)
#         group.prune()

# print(f"✅ Physical pruning done for: {os.path.basename(ckpt_path)}")

# # ================================
# # 4️⃣ 파라미터 수 & 파일 크기 측정
# # ================================
# total_params = sum(p.numel() for p in model.parameters())
# save_path = "./exp/checkpoints/resnet152_temp_phys.pth"
# torch.save(model.state_dict(), save_path)
# size_mb = os.path.getsize(save_path) / (1024**2)
# if os.path.exists(save_path): os.remove(save_path) # 측정 후 즉시 삭제

# # ================================
# # 5️⃣ Peak & Average VRAM 측정 (Inference)
# # ================================
# # 이전 메모리 기록 초기화
# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats(device)

# dummy = torch.randn(1, 3, 32, 32).to(device)
# mems = []

# with torch.no_grad():
#     # Warm-up (GPU 캐시 최적화 및 안정화)
#     for _ in range(15): # 152 모델은 조금 더 충분히 실행
#         _ = model(dummy)
    
#     # 실제 샘플링 루프 (50번 반복하여 평균값 도출)
#     for _ in range(50):
#         _ = model(dummy)
#         # memory_allocated(): 연산 과정에서 실제로 점유 중인 메모리 양
#         mems.append(torch.cuda.memory_allocated(device) / (1024**2))

# # 최대치 및 평균치 계산
# peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)
# avg_vram = np.mean(mems)

# # ================================
# # 6️⃣ 결과 출력
# # ================================
# print("\n" + "="*45)
# print(f"📊 Physical Pruning Analysis: {os.path.basename(ckpt_path)}")
# print("-" * 45)
# print(f" - Total Params  : {total_params:,}")
# print(f" - Model Size   : {size_mb:.2f} MB")
# print(f" - Peak VRAM    : {peak_vram:.2f} MB")
# print(f" - Average VRAM : {avg_vram:.2f} MB")
# print("="*45 + "\n")


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
# ckpt_path = "./exp/checkpoints/resnet152_DENSE_BASELINE.pth" 

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
# save_path = "./exp/checkpoints/resnet152_temps_phys.pth"


# # # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep40_sp23.35.pth"
# # # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep70_sp26.52.pth"
# # # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep100_sp37.55.pth"
# # # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep130_sp47.03.pth"
# # # ckpt_path = "./exp/checkpoints/resnet152_pdt_ep160_sp55.70.pth"





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
# import os
# import sys
# import gc

# # ------------------------------
# # 1️⃣ 설정
# # ------------------------------
# sys.path.append("./src")
# from models.resnet import get_resnet

# DEVICE = "cuda"

# # 🔥 여기만 바꾸면 됨
# CKPT_PATH = "./exp/checkpoints/resnet152_pdt_final.pth"
# # CKPT_PATH = "./exp/checkpoints/resnet152_DENSE_BASELINE.pth"

# MODEL_NAME = "resnet152"

# # ------------------------------
# # 2️⃣ 모델 로드
# # ------------------------------
# print(f"Loading {MODEL_NAME} from: {os.path.basename(CKPT_PATH)}")

# model = get_resnet(MODEL_NAME, num_classes=100)
# state_dict = torch.load(CKPT_PATH, map_location="cpu")
# model.load_state_dict(state_dict, strict=False)
# model.to(DEVICE)
# model.eval()

# # ------------------------------
# # 3️⃣ Physical Pruning (mask 존재 시)
# # ------------------------------
# example_inputs = torch.randn(1,3,32,32).to(DEVICE)
# DG = tp.DependencyGraph().build_dependency(model, example_inputs)

# pruned_any = False

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
#                 pruned_any = True

# if pruned_any:
#     print("✅ Physical pruning applied.")
# else:
#     print("ℹ️ No pruning applied (Dense model).")

# # ------------------------------
# # 4️⃣ Memory Cleanup
# # ------------------------------
# del DG
# del example_inputs
# gc.collect()
# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats()

# print(f"✅ Physical pruning & Memory Cleanup done: {os.path.basename(CKPT_PATH)}")

# # ------------------------------
# # 5️⃣ 모델 통계
# # ------------------------------
# total_params = sum(p.numel() for p in model.parameters())

# temp_path = "./temp_model.pth"
# torch.save(model.state_dict(), temp_path)
# size_mb = os.path.getsize(temp_path) / (1024**2)
# os.remove(temp_path)

# # ------------------------------
# # 6️⃣ Inference Peak 측정
# # ------------------------------
# dummy = torch.randn(1,3,32,32).to(DEVICE)

# with torch.inference_mode():
#     for _ in range(10):
#         _ = model(dummy)

#     torch.cuda.synchronize()
#     torch.cuda.reset_peak_memory_stats()

#     for _ in range(50):
#         _ = model(dummy)
#         torch.cuda.synchronize()

# peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
# peak_resv  = torch.cuda.max_memory_reserved() / (1024**2)

# # ------------------------------
# # 7️⃣ 출력
# # ------------------------------
# print("\n" + "="*50)
# print(f"📊 Optimized Memory Analysis: {os.path.basename(CKPT_PATH)}")
# print("-"*50)
# print(" [Model Stats]")
# print(f"  - Total Params      : {total_params:,}")
# print(f"  - Model File Size   : {size_mb:.2f} MB")
# print("-"*50)
# print(" [VRAM Stats - Clean State]")
# print(f"  - Peak Allocated    : {peak_alloc:.2f} MB (Actual Tensors)")
# print(f"  - Peak Reserved     : {peak_resv:.2f} MB (Caching Pool)")
# print("="*50 + "\n")


import torch
import torch.nn as nn
import torch_pruning as tp
import sys
import gc

sys.path.append("./src")
from models.resnet import get_resnet

DEVICE = "cuda"
MODEL_NAME = "resnet152"
CKPT_PATH = "./exp/checkpoints/resnet152_DENSE_BASELINE.pth"

model = get_resnet(MODEL_NAME, num_classes=100)
model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"), strict=False)
model.to(DEVICE)
model.eval()

example_inputs = torch.randn(1,3,32,32).to(DEVICE)
DG = tp.DependencyGraph().build_dependency(model, example_inputs)

print("🔥 Aggressive Width Pruning (50%)")

for name, m in model.named_modules():

    # Bottleneck의 conv3만 공격적으로 prune
    if isinstance(m, nn.Conv2d) and "conv3" in name:

        out_channels = m.out_channels
        prune_idx = list(range(out_channels // 2))  # 절반 제거

        print(f"Pruning {name} | {len(prune_idx)} / {out_channels}")

        group = DG.get_pruning_group(
            m,
            tp.prune_conv_out_channels,
            prune_idx
        )

        if DG.check_pruning_group(group):
            group.prune()

print("✅ Width pruning complete.")

# VRAM 측정
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

model.train()

dummy = torch.randn(32,3,32,32).to(DEVICE)
out = model(dummy)
loss = out.sum()
loss.backward()

torch.cuda.synchronize()

print("\n🔥 Training Peak After Width Pruning")
print("Peak Allocated:",
      torch.cuda.max_memory_allocated()/1024**2, "MB")
print("Peak Reserved:",
      torch.cuda.max_memory_reserved()/1024**2, "MB")

print("Layer4 conv3 out:", model.layer4[0].conv3.out_channels)