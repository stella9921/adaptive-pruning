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
# # 1️⃣ 모델 설정 (EfficientNet-B0)
# # ================================
# config, _ = load_config()
# config['model']['name'] = 'efficientnet_b0' # 모델명 변경
# config['model']['num_classes'] = 100

# model = get_model(config['model']).to(device)

# # ================================
# # 2️⃣ 체크포인트 경로 설정
# # ================================
# # 분석할 EfficientNet-B0 체크포인트 경로를 순차적으로 입력하세요.
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep40_sp10.00.pth"



# ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep40_sp12.29.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep60_sp22.28.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep80_sp32.61.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep100_sp43.06.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep120_sp51.15.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep140_sp60.23.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep160_sp70.73.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep180_sp94.16.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep200_sp95.31.pth" # 예시 경로







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
# # EfficientNet은 Depthwise/Group Conv가 포함되어 있어 의존성 그래프의 역할이 매우 중요합니다.
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

#         # 물리적 채널 제거
#         group = DG.get_pruning_group(module, tp.prune_conv_out_channels, prune_idx)
#         group.prune()

# print(f"✅ Physical pruning done for: {os.path.basename(ckpt_path)}")

# # ================================
# # 4️⃣ 파라미터 수 & 파일 크기 측정
# # ================================
# total_params = sum(p.numel() for p in model.parameters())
# save_path = "./exp/checkpoints/efficientnet_temp_phys.pth"
# torch.save(model.state_dict(), save_path)
# size_mb = os.path.getsize(save_path) / (1024**2)
# if os.path.exists(save_path): os.remove(save_path) 

# # ================================
# # 5️⃣ Peak & Average VRAM 측정 (Inference)
# # ================================
# # 메모리 캐시 초기화
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()
#     torch.cuda.reset_peak_memory_stats(device)

# dummy = torch.randn(1, 3, 32, 32).to(device)
# mems = []

# with torch.no_grad():
#     # Warm-up (초기화 및 최적화를 위해 10번 실행)
#     for _ in range(10):
#         _ = model(dummy)
    
#     # 실제 샘플링 루프 (50번 반복하며 평균 측정)
#     for _ in range(50):
#         _ = model(dummy)
#         # memory_allocated(): 현재 GPU에 실제로 할당되어 사용 중인 메모리
#         mems.append(torch.cuda.memory_allocated(device) / (1024**2))

# # 결과 산출
# peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)
# avg_vram = np.mean(mems)

# # ================================
# # 6️⃣ 결과 출력
# # ================================
# print("\n" + "="*45)
# print(f"📊 EfficientNet-B0 Physical Pruning Analysis: {os.path.basename(ckpt_path)}")
# print("-" * 45)
# print(f" - Total Params  : {total_params:,}")
# print(f" - Model Size    : {size_mb:.2f} MB")
# print(f" - Peak VRAM     : {peak_vram:.2f} MB")
# print(f" - Average VRAM  : {avg_vram:.2f} MB")
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

# # 1. 모델 로드
# config, _ = load_config()
# config['model']['name'] = 'efficientnet_b0' 
# config['model']['num_classes'] = 100
# model = get_model(config['model']).to(device)

# # 2. 체크포인트 로드
 
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep40_sp12.29.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep60_sp22.28.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep80_sp32.61.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep100_sp43.06.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep120_sp51.15.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep140_sp60.23.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep160_sp70.73.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep180_sp94.16.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep260_sp48.7.pth" # 예시 경로



# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep120_sp15.5.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep140_sp23.7.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep160_sp28.5.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep180_sp32.3.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep200_sp35.8.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep220_sp39.7.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep240_sp44.7.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep260_sp48.7.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep280_sp52.9.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep300_sp56.5.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep320_sp60.3.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep340_sp63.3.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep360_sp65.9.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep380_sp68.3.pth" # 예시 경로
# ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep140_sp57.55.pth" # 예시 경로

# # efficientnet_b0_DENSE_BASELINE.pth
# # efficientnet_b0_pdt_ep40_sp11.65.pth
# # efficientnet_b0_pdt_ep140_sp57.55.pth
# # efficientnet_b0_pdt_ep180_sp94.01.pth

# checkpoint = torch.load(ckpt_path, map_location=device)
# state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
# model.load_state_dict(state_dict, strict=False)
# model.eval()

# # 3. Physical Pruning
# example_inputs = torch.randn(1, 3, 32, 32).to(device)
# DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

# for module in model.modules():
#     if hasattr(module, "mask") and isinstance(module, nn.Conv2d):
#         prune_idx = torch.where(module.mask == 0)[0].tolist()
#         if len(prune_idx) == 0 or len(prune_idx) >= module.out_channels:
#             continue
#         try:
#             group = DG.get_pruning_group(module, tp.prune_conv_out_channels, prune_idx)
#             group.prune()
#         except: continue

# # 🔥 [핵심 수정] 모든 Conv2d 레이어의 무결성 검사 및 강제 수정
# for name, m in model.named_modules():
#     if isinstance(m, nn.Conv2d):
#         # Weight의 첫 번째 차원이 실제 출력 채널 수입니다.
#         actual_out_channels = m.weight.data.shape[0]
#         actual_in_channels = m.weight.data.shape[1]
        
#         # 1. 채널 수 설정 동기화
#         m.out_channels = actual_out_channels
        
#         # 2. Depthwise Conv(groups > 1)인 경우 groups 값을 강제로 맞춤
#         if m.groups > 1:
#             # Depthwise Conv는 out_channels와 groups가 항상 같아야 함
#             # 또한 weight.shape[1]은 항상 1이어야 함 (Depthwise 특성)
#             if actual_in_channels == 1:
#                 m.groups = actual_out_channels
#                 m.in_channels = actual_out_channels
#             else:
#                 # 일반 Group Conv인 경우 (거의 없지만 방어용)
#                 # weight.shape[0]이 groups로 나누어 떨어져야 함
#                 if actual_out_channels % m.groups != 0:
#                      m.groups = 1 # 에러 방지를 위해 일반 Conv로 강제 전환

# print(f"✅ Physical pruning & Group synchronization complete.")

# # 4. 결과 측정
# total_params = sum(p.numel() for p in model.parameters())
# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats(device)
# mems = []

# with torch.no_grad():
#     dummy = torch.randn(1, 3, 32, 32).to(device)
#     # Warm-up (여기가 터지던 구간)
#     try:
#         for _ in range(10): _ = model(dummy)
        
#         for _ in range(50):
#             _ = model(dummy)
#             mems.append(torch.cuda.memory_allocated(device) / (1024**2))

#         peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)
#         avg_vram = np.mean(mems)

#         print("\n" + "="*45)
#         print(f"📊 Results: {os.path.basename(ckpt_path)}")
#         print(f" - Total Params  : {total_params:,}")
#         print(f" - Peak VRAM     : {peak_vram:.2f} MB")
#         print(f" - Average VRAM  : {avg_vram:.2f} MB")
#         print("="*45 + "\n")
#     except Exception as e:
#         print(f"❌ 여전히 에러 발생: {e}")
#         # 에러가 난 레이어의 정보를 출력해서 원인 파악
#         for name, m in model.named_modules():
#             if isinstance(m, nn.Conv2d):
#                 if m.weight.shape[0] % m.groups != 0:
#                     print(f"⚠️ 문제 레이어: {name} | Weight: {m.weight.shape} | Groups: {m.groups}")


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
# # 1️⃣ 모델 설정 (EfficientNet-B0로 변경)
# # ================================
# config, _ = load_config()
# config['model']['name'] = 'efficientnet_b0'  # EfficientNet-B0 설정
# config['model']['num_classes'] = 100

# model = get_model(config['model']).to(device)

# # ================================
# # 2️⃣ 체크포인트 로드
# # ================================
# # EfficientNet-B0 용 체크포인트 경로로 수정하여 사용하세요.
# ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep200_sp95.30.pth" 
# # 실험 다시 함
# # efficientnet_b0_DENSE_BASELINE.pth
# # efficientnet_b0_pdt_ep40_sp11.64.pth
# # efficientnet_b0_pdt_ep60_sp21.79.pth
# # efficientnet_b0_pdt_ep80_sp30.38.pth
# # efficientnet_b0_pdt_ep100_sp38.69.pth
# # efficientnet_b0_pdt_ep120_sp46.62.pth
# # efficientnet_b0_pdt_ep140_sp57.37.pth
# # efficientnet_b0_pdt_ep160_sp67.11.pth
# # efficientnet_b0_pdt_ep180_sp93.87.pth
# # efficientnet_b0_pdt_ep200_sp95.30.pth
# # efficientnet_b0_pdt_final.pth
# ./exp/checkpoints/efficientnet_b0_pdt_ep240_sp44.7.pth

# # # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep40_sp12.29.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep60_sp22.28.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep80_sp32.61.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep100_sp43.06.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep120_sp51.15.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep140_sp60.23.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep160_sp70.73.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep180_sp94.16.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep260_sp48.7.pth" # 예시 경로



# # # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep120_sp15.5.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep140_sp23.7.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep160_sp28.5.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep180_sp32.3.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep200_sp35.8.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep220_sp39.7.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep240_sp44.7.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep260_sp48.7.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep280_sp52.9.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep300_sp56.5.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep320_sp60.3.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep340_sp63.3.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep360_sp65.9.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep380_sp68.3.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/efficientnet_b0_pdt_ep140_sp57.55.pth" # 예시 경로


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
# # EfficientNet은 채널 수가 작으므로 예시 입력 (1, 3, 224, 224) 또는 (1, 3, 32, 32) 유지
# example_inputs = torch.randn(1, 3, 32, 32).to(device)
# DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

# print(f"🔄 Analyzing Dependency Graph for EfficientNet-B0...")

# for name, module in model.named_modules():

#     if isinstance(module, nn.Conv2d):

#         # 🔴 depthwise conv는 건너뛰기
#         if module.groups == module.in_channels and module.in_channels == module.out_channels:
#             continue

#         weight = module.weight.data
#         out_channels = weight.shape[0]

#         channel_norm = weight.view(out_channels, -1).abs().sum(dim=1)
#         prune_idx = torch.where(channel_norm == 0)[0].tolist()

#         if len(prune_idx) == 0:
#             continue

#         if len(prune_idx) >= out_channels:
#             prune_idx = prune_idx[:-1]

#         print(f"   -> Pruning Conv: {name} | {len(prune_idx)} channels removed")

#         group = DG.get_pruning_group(
#             module,
#             tp.prune_conv_out_channels,
#             prune_idx
#         )
#         group.prune()

# print(f"✅ Physical pruning done: {os.path.basename(ckpt_path)}")

# # 파라미터 수 및 실제 파일 크기 계산
# total_params = sum(p.numel() for p in model.parameters())
# save_path = "./exp/checkpoints/eff_b0_temp_phys.pth"

# torch.save(model.state_dict(), save_path)
# size_mb = os.path.getsize(save_path) / (1024**2)
# if os.path.exists(save_path): os.remove(save_path)

# # ================================
# # 4️⃣ 상세 VRAM 측정 (Inference Mode)
# # ================================

# # [슬랙 피드백 반영] 측정 전 메모리 완전 초기화 및 동기화
# torch.cuda.empty_cache()
# torch.cuda.synchronize() 
# torch.cuda.reset_peak_memory_stats(device) 

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
# print(f"📊 Detailed Memory Analysis: EfficientNet-B0")
# print(f" Path: {os.path.basename(ckpt_path)}")
# print("-" * 50)
# print(f" [Model Stats]")
# print(f"   - Total Params      : {total_params:,}")
# print(f"   - Model File Size   : {size_mb:.2f} MB")
# print("-" * 50)
# print(f" [VRAM Stats - Inference Mode]")
# print(f"   - Current Allocated : {curr_alloc:.2f} MB")
# print(f"   - Current Reserved  : {curr_resv:.2f} MB")
# print(f"   - Peak Allocated    : {peak_alloc:.2f} MB (Actual Tensors)")
# print(f"   - Peak Reserved     : {peak_resv:.2f} MB (Caching Pool)")
# print("="*50 + "\n")





import torch
import torch.nn as nn
import torch_pruning as tp
import torchvision.models as models
import os
import gc

DEVICE = "cuda"

# 🔥 체크포인트 바꿔가며 사용
CKPT_PATH = "./exp/checkpoints/efficientnet_b0_pdt_final.pth"


# CKPT_PATH = "./exp/checkpoints/efficientnet_b0_DENSE_BASELINE.pth"

MODEL_NAME = "efficientnet_b0"

print(f"\n🔄 Loading {MODEL_NAME} from: {os.path.basename(CKPT_PATH)}")

# ------------------------------
# 1️⃣ 모델 생성
# ------------------------------
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 100)

state_dict = torch.load(CKPT_PATH, map_location="cpu")
model.load_state_dict(state_dict, strict=False)

model.to(DEVICE)
model.eval()

# ------------------------------
# 2️⃣ Dependency Graph 구성
# ------------------------------
print("🔄 Building Dependency Graph...")
example_inputs = torch.randn(1,3,32,32).to(DEVICE)
DG = tp.DependencyGraph().build_dependency(model, example_inputs)

total_pruned = 0

# ------------------------------
# 3️⃣ mask=0 채널 전부 물리 제거
# ------------------------------
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
                total_pruned += len(prune_idx)

if total_pruned > 0:
    print(f"✅ Physical pruning applied. Total pruned channels: {total_pruned}")
else:
    print("ℹ️ No pruning applied (Dense model or no masked channels).")

# ------------------------------
# 4️⃣ 메모리 정리
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



