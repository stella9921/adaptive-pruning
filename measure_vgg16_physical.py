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
# # 1️⃣ 모델 설정 (VGG-16)
# # ================================
# config, _ = load_config()
# config['model']['name'] = 'vgg16'  # VGG-16으로 변경
# config['model']['num_classes'] = 100

# model = get_model(config['model']).to(device)

# # ================================
# # 2️⃣ 체크포인트 경로 (VGG-16용 체크포인트 지정)
# # ================================
# # VGG-16용 체크포인트 파일명을 여기에 입력하세요.
# # 예: ckpt_path = "./exp/checkpoints/vgg16_pdt_ep160_sp60.00.pth"

# # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep40_sp16.99.pth" 
# # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep60_sp24.63.pth" 
# # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep80_sp33.42.pth" 
# # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep100_sp57.05.pth" 
# # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep120_sp97.93.pth" 
# # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep140_sp97.93.pth" 
# # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep160_sp97.93.pth" 
# # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep180_sp97.93.pth" 
# ckpt_path = "./exp/checkpoints/vgg16_pdt_ep200_sp99.15.pth" 

# if not os.path.exists(ckpt_path):
#     print(f"❌ File not found: {ckpt_path}")
#     sys.exit()

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
# # VGG-16 구조에 맞는 의존성 그래프 생성
# example_inputs = torch.randn(1, 3, 32, 32).to(device)
# DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

# for module in model.modules():
#     if not hasattr(module, "mask"):
#         continue

#     if isinstance(module, nn.Conv2d):
#         # 마스크가 0인 인덱스(제거 대상) 추출
#         prune_idx = torch.where(module.mask == 0)[0].tolist()

#         if len(prune_idx) == 0:
#             continue

#         # 모든 채널이 잘리는 것 방지 (최소 1개 채널은 유지)
#         if len(prune_idx) >= module.out_channels:
#             continue

#         # 물리적 제거 실행
#         # VGG는 단순 직렬 구조이므로 tp.prune_conv_out_channels가 후속 레이어에 즉각 반영됨
#         group = DG.get_pruning_group(
#             module,
#             tp.prune_conv_out_channels,
#             prune_idx
#         )
#         group.prune()

# print("✅ VGG-16 Physical pruning done.")

# # ================================
# # 4️⃣ 파라미터 수 확인
# # ================================
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total Params After Physical Pruning (VGG-16): {total_params:,}")

# # ================================
# # 5️⃣ 파일 크기 확인
# # ================================
# save_path = "./exp/checkpoints/vgg16_physically_pruned.pth"
# torch.save(model.state_dict(), save_path)

# size_mb = os.path.getsize(save_path) / (1024**2)
# print(f"Compressed Model File Size: {size_mb:.2f} MB")

# # ================================
# # 6️⃣ Peak VRAM 측정 (Inference)
# # ================================
# # 측정 전 메모리 통계 초기화
# if torch.cuda.is_available():
#     torch.cuda.reset_peak_memory_stats(device)

# dummy = torch.randn(1, 3, 32, 32).to(device)
# with torch.no_grad():
#     _ = model(dummy)

# if torch.cuda.is_available():
#     peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
#     print(f"Peak VRAM After Physical Pruning: {peak_mem:.2f} MB")
# else:
#     print("CUDA is not available. VRAM measurement skipped.")

# # 임시 파일 삭제 (용량 관리)
# if os.path.exists(save_path):
#     os.remove(save_path)

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
# # 1️⃣ 모델 설정 (VGG-16)
# # ================================
# config, _ = load_config()
# config['model']['name'] = 'vgg16'
# config['model']['num_classes'] = 100

# model = get_model(config['model']).to(device)

# # ================================
# # 2️⃣ 체크포인트 경로 설정
# # ================================
# # 분석할 VGG-16 체크포인트를 선택하세요.
# # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep40_sp16.99.pth" 
# # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep60_sp24.63.pth" 
# # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep80_sp33.42.pth" 
# # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep100_sp57.05.pth" 
# # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep120_sp97.93.pth" 
# # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep140_sp97.93.pth" 
# # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep160_sp97.93.pth" 
# # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep180_sp97.93.pth" 
# # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep200_p99.15.pth" 
# ckpt_path = "./exp/checkpoints/vgg16_DENSE_BASELINE.pth" 
# # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep40_sp15.89.pth" 
# # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep80_sp31.25.pth" 


# # vgg16_DENSE_BASELINE.pth
# # vgg16_pdt_ep40_sp15.89.pth
# # vgg16_pdt_ep80_sp31.25.pth

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

# print(f"✅ VGG-16 Physical pruning done for: {os.path.basename(ckpt_path)}")

# # ================================
# # 4️⃣ 파라미터 수 & 파일 크기
# # ================================
# total_params = sum(p.numel() for p in model.parameters())
# save_path = "./exp/checkpoints/vgg16_temp_phys.pth"
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
#     # Warm-up (GPU 초기 연산 최적화를 위해 10번 실행)
#     for _ in range(10):
#         _ = model(dummy)
    
#     # 실제 샘플링 루프 (50번 반복하며 평균 측정)
#     for _ in range(50):
#         _ = model(dummy)
#         # 현재 GPU에 실제로 할당되어 사용 중인 메모리 양 기록
#         mems.append(torch.cuda.memory_allocated(device) / (1024**2))

# # 결과 산출
# peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)
# avg_vram = np.mean(mems)

# # ================================
# # 6️⃣ 결과 출력
# # ================================
# print("\n" + "="*45)
# print(f"📊 VGG-16 Physical Pruning Analysis: {os.path.basename(ckpt_path)}")
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

# # 1️⃣ 모델 설정 및 체크포인트 로드
# config, _ = load_config()
# config['model']['name'] = 'vgg16'
# config['model']['num_classes'] = 100
# model = get_model(config['model']).to(device)

# # ckpt_path = "./exp/checkpoints/vgg16_DENSE_BASELINE.pth" # 분석 타겟 설정
# ckpt_path = "./exp/checkpoints/vgg16_pdt_ep40_sp15.89.pth" 
# # # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep80_sp31.25.pth" 


# # # vgg16_DENSE_BASELINE.pth
# # # vgg16_pdt_ep40_sp15.89.pth
# # # vgg16_pdt_ep80_sp31.25.pth


# if not os.path.exists(ckpt_path):
#     print(f"❌ File not found: {ckpt_path}"); sys.exit()

# checkpoint = torch.load(ckpt_path, map_location=device)
# state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
# model.load_state_dict(state_dict, strict=False)
# model.eval()

# # 2️⃣ 의존성 그래프 구축 (Pruning의 핵심)
# # VGG16은 구조가 단순하지만 BN층 등이 엮여있어 DependencyGraph가 필수입니다.
# example_inputs = torch.randn(1, 3, 32, 32).to(device)
# DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

# # 3️⃣ 물리적 프루닝 수행 (수정된 로직)
# print(f"🚀 Physical pruning starting for: {os.path.basename(ckpt_path)}")

# # 모델의 모든 레이어를 돌며 마스크 기반으로 채널 삭제 지점 탐색
# for m in model.modules():
#     # 'mask' 속성이 있고, Conv2d이면서 마스크에 0(삭제)이 포함된 경우
#     if isinstance(m, nn.Conv2d) and hasattr(m, "mask"):
#         # 채널별 마스크 합산 등을 통해 삭제할 인덱스 추출
#         # 보통 mask는 [out_channels, 1, 1, 1] 형태이거나 [out_channels] 형태입니다.
#         mask_sum = m.mask.view(m.out_channels, -1).sum(1)
#         prune_idx = torch.where(mask_sum == 0)[0].tolist()
        
#         if len(prune_idx) > 0:
#             # 안전장치: 모든 채널이 잘리는 것 방지 (최소 1개는 남김)
#             if len(prune_idx) >= m.out_channels:
#                 prune_idx = prune_idx[:-1] 
            
#             # 의존성 그룹 추출 및 물리적 삭제 실행
#             group = DG.get_pruning_group(m, tp.prune_conv_out_channels, idxs=prune_idx)
#             if DG.check_pruning_group(group): # 프루닝 가능 여부 확인
#                 group.prune()

# print(f"✅ Physical pruning finished.")

# # 4️⃣ 지표 측정
# # 실제 파라미터 수 확인
# total_params = sum(p.numel() for p in model.parameters())

# # 실제 파일 크기 확인 (물리적으로 변한 모델의 용량)
# save_path = "./exp/checkpoints/temp_vgg_phys.pth"
# torch.save(model, save_path) # state_dict가 아니라 모델 전체를 저장해 구조 변경 반영
# size_mb = os.path.getsize(save_path) / (1024**2)
# if os.path.exists(save_path): os.remove(save_path)

# # 5️⃣ VRAM 측정
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()
#     torch.cuda.reset_peak_memory_stats(device)

# with torch.no_grad():
#     dummy = torch.randn(1, 3, 32, 32).to(device)
#     # Warm-up
#     for _ in range(10): _ = model(dummy)
    
#     # Sampling
#     mems = []
#     for _ in range(50):
#         _ = model(dummy)
#         mems.append(torch.cuda.memory_allocated(device) / (1024**2))

# peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)
# avg_vram = np.mean(mems)

# # 6️⃣ 결과 출력
# print("\n" + "="*45)
# print(f"📊 VGG-16 Physical Analysis Result")
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

# # 1️⃣ 모델 설정 및 체크포인트 로드
# config, _ = load_config()
# config['model']['name'] = 'vgg16'
# config['model']['num_classes'] = 100
# model = get_model(config['model']).to(device)

# # 분석할 체크포인트 경로 (실행 시 수동 변경 또는 인자로 받기)
# # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep120_sp97.93.pth" 

# # ckpt_path = "./exp/checkpoints/vgg16_DENSE_BASELINE.pth" # 분석 타겟 설정
# # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep40_sp15.89.pth" 
# # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep80_sp31.25.pth" 
# ckpt_path = "./exp/checkpoints/vgg16_pdt_ep200_sp98.19.pth" 

# if not os.path.exists(ckpt_path):
#     print(f"❌ File not found: {ckpt_path}"); sys.exit()

# checkpoint = torch.load(ckpt_path, map_location=device)
# state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
# model.load_state_dict(state_dict, strict=False)
# model.eval()

# # 2️⃣ 물리적 프루닝 수행 (개선된 인덱스 로직)
# print(f"🚀 Physical pruning starting for: {os.path.basename(ckpt_path)}")

# # Dependency Graph 구축 (VGG-16 구조 전파용)
# example_inputs = torch.randn(1, 3, 32, 32).to(device)
# DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

# # 프루닝 대상 레이어 탐색 (Conv2d만 타겟팅)
# pruning_plan = []

# for name, m in model.named_modules():
#     if isinstance(m, nn.Conv2d) and hasattr(m, "mask"):
#         # 마스크에서 값이 0인 인덱스 추출 (물리적으로 삭제할 채널)
#         # mask가 [C, 1, 1, 1]일 수도 있으므로 flatten 처리
#         m_mask = m.mask.view(-1)
#         prune_idx = torch.where(m_mask == 0)[0].tolist()
        
#         if len(prune_idx) > 0:
#             # 모든 채널이 삭제되는 것 방지
#             if len(prune_idx) >= m.out_channels:
#                 prune_idx = prune_idx[:-1]
            
#             # 프루닝 계획에 추가 (순차적으로 실행하기 위함)
#             pruning_plan.append((m, prune_idx))

# # 실제로 모델 구조를 변경 (물리적 삭제)
# for module, idxs in pruning_plan:
#     group = DG.get_pruning_group(module, tp.prune_conv_out_channels, idxs=idxs)
#     if DG.check_pruning_group(group):
#         group.prune()

# print(f"✅ Physical pruning finished. Structure modified.")

# # 3️⃣ 지표 측정 (구조 변경 후 파라미터 수 확인)
# total_params = sum(p.numel() for p in model.parameters())

# # 가상 파일 저장을 통한 실제 모델 크기 측정
# save_path = "./exp/checkpoints/temp_phys_vgg.pth"
# torch.save(model, save_path) 
# size_mb = os.path.getsize(save_path) / (1024**2)
# if os.path.exists(save_path): os.remove(save_path)

# # 4️⃣ VRAM 측정 (Inference)
# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats(device)

# with torch.no_grad():
#     dummy = torch.randn(1, 3, 32, 32).to(device)
#     for _ in range(15): _ = model(dummy) # Warm-up
    
#     mems = []
#     for _ in range(50):
#         _ = model(dummy)
#         mems.append(torch.cuda.memory_allocated(device) / (1024**2))

# peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)
# avg_vram = np.mean(mems)

# # 5️⃣ 결과 출력
# print("\n" + "="*45)
# print(f"📊 VGG-16 Physical Analysis: {os.path.basename(ckpt_path)}")
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
# import gc

# sys.path.append(os.path.abspath("."))

# from src.models import get_model
# from src.utils.config_loader import load_config

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.enabled = False

# # ================================
# # 1️⃣ 모델 설정 (VGG16으로 설정)
# # ================================
# config, _ = load_config()
# config['model']['name'] = 'vgg16'  # 모델명을 vgg16으로 변경
# config['model']['num_classes'] = 100

# model = get_model(config['model']).to(device)

# # ================================
# # 2️⃣ 체크포인트 로드
# # ================================
# ckpt_path = "./exp/checkpoints/vgg16_pdt_ep120_sp97.93.pth" 


# # # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep40_sp16.99.pth" 
# # # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep60_sp24.63.pth" 
# # # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep80_sp33.42.pth" 
# # # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep100_sp57.05.pth" 
# # # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep120_sp97.93.pth" 
# # # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep140_sp97.93.pth" 
# # # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep160_sp97.93.pth" 
# # # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep180_sp97.93.pth" 
# # # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep200_p99.15.pth" 
# # ckpt_path = "./exp/checkpoints/vgg16_DENSE_BASELINE.pth" 
# # # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep40_sp15.89.pth" 
# # # ckpt_path = "./exp/checkpoints/vgg16_pdt_ep80_sp31.25.pth" 


# if not os.path.exists(ckpt_path):
#     print(f"❌ File not found: {ckpt_path}")
#     sys.exit()

# checkpoint = torch.load(ckpt_path, map_location=device)
# state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

# model.load_state_dict(state_dict, strict=False)
# model.eval()

# # ================================
# # 3️⃣ Physical Pruning 수행 및 정리
# # ================================
# # ================================
# # 3️⃣ Physical Pruning (Weight 기반)
# # ================================
# example_inputs = torch.randn(1, 3, 32, 32).to(device)
# DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

# print(f"🔄 Analyzing Dependency Graph for VGG16...")

# for name, module in model.named_modules():

#     # Conv2d 처리
#     if isinstance(module, nn.Conv2d):
#         weight = module.weight.data
#         out_channels = weight.shape[0]

#         # 채널별 L1 norm 계산
#         channel_norm = weight.view(out_channels, -1).abs().sum(dim=1)

#         # 완전히 0인 채널 찾기
#         prune_idx = torch.where(channel_norm == 0)[0].tolist()

#         if len(prune_idx) == 0:
#             continue

#         print(f"   -> Pruning Conv: {name} | {len(prune_idx)} channels removed")

#         group = DG.get_pruning_group(
#             module,
#             tp.prune_conv_out_channels,
#             prune_idx
#         )
#         group.prune()

#     # Linear 처리 (VGG classifier)
#     elif isinstance(module, nn.Linear):
#         weight = module.weight.data
#         out_features = weight.shape[0]

#         feature_norm = weight.abs().sum(dim=1)
#         prune_idx = torch.where(feature_norm == 0)[0].tolist()

#         if len(prune_idx) == 0:
#             continue

#         print(f"   -> Pruning Linear: {name} | {len(prune_idx)} features removed")

#         group = DG.get_pruning_group(
#             module,
#             tp.prune_linear_out_channels,
#             prune_idx
#         )
#         group.prune()

# # 정리
# del DG
# del example_inputs
# gc.collect()
# torch.cuda.empty_cache()

# print(f"✅ Physical pruning done.")
# print(f"✅ Physical pruning & Memory Cleanup done: {os.path.basename(ckpt_path)}")

# # 파라미터 수 확인
# total_params = sum(p.numel() for p in model.parameters())
# save_path = "./exp/checkpoints/vgg16_temp_phys.pth"
# torch.save(model.state_dict(), save_path)
# size_mb = os.path.getsize(save_path) / (1024**2)
# if os.path.exists(save_path): os.remove(save_path)

# # ================================
# # 4️⃣ 상세 VRAM 측정 (Inference Mode)
# # ================================
# # [슬랙 피드백 반영] 측정 전 완전 초기화 및 동기화
# torch.cuda.synchronize()
# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats(device) 

# # dummy = torch.randn(1, 3, 32, 32).to(device)
# dummy = torch.randn(64, 3, 32, 32).to(device)

# # [슬랙 피드백 반영] torch.inference_mode() 사용
# with torch.inference_mode():
#     # Warm-up
#     for _ in range(10):
#         _ = model(dummy)
    
#     torch.cuda.synchronize()
#     # Warm-up 오버헤드 제외를 위해 피크 통계 리셋
#     torch.cuda.reset_peak_memory_stats(device) 
    
#     # 실제 추론 수행 (50회)
#     for _ in range(50):
#         _ = model(dummy)
#         torch.cuda.synchronize()

# # 4가지 핵심 메모리 지표 추출
# curr_alloc = torch.cuda.memory_allocated(device) / (1024**2)
# curr_resv  = torch.cuda.memory_reserved(device) / (1024**2)
# peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**2)
# peak_resv  = torch.cuda.max_memory_reserved(device) / (1024**2)

# # ================================
# # 5️⃣ 결과 출력 (Table 2/Fig 3 보완용)
# # ================================
# print("\n" + "="*50)
# print(f"📊 Detailed Memory Analysis: VGG16")
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


# import torch
# import torch.nn as nn
# import torch_pruning as tp
# import torchvision.models as models
# import os
# import gc

# # ------------------------------
# # 1️⃣ 설정
# # ------------------------------
# DEVICE = "cuda"

# # 🔥 체크포인트 바꿔가며 사용
# CKPT_PATH = "./exp/checkpoints/vgg16_pdt_final.pth"
# # CKPT_PATH = "./exp/checkpoints/vgg16_DENSE_BASELINE.pth"

# MODEL_NAME = "vgg16"

# # ------------------------------
# # 2️⃣ 모델 로드
# # ------------------------------
# print(f"Loading model from: {os.path.basename(CKPT_PATH)}")

# model = models.vgg16(weights=None)

# # CIFAR-100용 classifier 수정
# model.classifier[6] = nn.Linear(model.classifier[6].in_features, 100)

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
import os
import sys
import gc

# ------------------------------
# 1️⃣ 설정
# ------------------------------
DEVICE = "cuda"
CKPT_PATH = "./exp/checkpoints/vgg16_pdt_ep61.pth"
# CKPT_PATH = "./exp/checkpoints/vgg16_DENSE_BASELINE.pth"

# vgg16_pdt_ep57.pth
# vgg16_pdt_ep58.pth
# vgg16_pdt_ep59.pth
# vgg16_pdt_ep60_sp23.24.pth
# vgg16_pdt_ep60.pth
# vgg16_pdt_ep61.pth
# vgg16_pdt_ep62.pth
# vgg16_pdt_ep63.pth


sys.path.append("./src")
from models.vgg import get_vgg16   # 🔥 네 프로젝트 함수 사용

print(f"Loading model from: {os.path.basename(CKPT_PATH)}")

# ------------------------------
# 2️⃣ 모델 생성 (CIFAR 구조)
# ------------------------------
model = get_vgg16(num_classes=100)

state_dict = torch.load(CKPT_PATH, map_location="cpu")
model.load_state_dict(state_dict, strict=False)

model.to(DEVICE)
model.eval()

# ------------------------------
# 3️⃣ Physical Pruning
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