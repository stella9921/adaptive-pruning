# # import torch
# # import torch.nn as nn
# # import torch_pruning as tp
# # import os
# # import sys
# # import numpy as np

# # # 프로젝트 루트 경로 추가
# # sys.path.append(os.path.abspath("."))

# # from src.models import get_model
# # from src.utils.config_loader import load_config

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # ================================
# # # 1️⃣ 모델 설정 (MobileNet-V2)
# # # ================================
# # config, _ = load_config()
# # config['model']['name'] = 'mobilenet_v2' # 모델명 변경
# # config['model']['num_classes'] = 100

# # model = get_model(config['model']).to(device)

# # # ================================
# # # 2️⃣ 체크포인트 경로 설정
# # # ================================
# # # 분석할 MobileNet-V2 체크포인트 경로를 순차적으로 입력하세요.
# # # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep40_sp15.00.pth"
# # # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep200_sp55.00.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep200_sp55.00.pth" # 예시 경로

# # # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep40_sp10.10.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep60_sp20.31.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep80_sp28.55.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep100_sp36.05.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep120_sp47.38.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep140_sp89.68.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep160_sp89.68.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep180_sp89.68.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep200_sp91.68.pth" # 예시 경로

# # if not os.path.exists(ckpt_path):
# #     print(f"❌ File not found: {ckpt_path}")
# #     sys.exit()

# # checkpoint = torch.load(ckpt_path, map_location=device)
# # state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

# # model.load_state_dict(state_dict, strict=False)
# # model.eval()

# # # ================================
# # # 3️⃣ Physical Pruning (물리적 구조 변경)
# # # ================================
# # # MobileNet-V2는 Inverted Residual 구조이므로 의존성 그래프 분석이 필수적입니다.
# # example_inputs = torch.randn(1, 3, 32, 32).to(device)
# # DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

# # for module in model.modules():
# #     if not hasattr(module, "mask"):
# #         continue

# #     if isinstance(module, nn.Conv2d):
# #         # 마스크가 0인 인덱스(제거 대상) 추출
# #         prune_idx = torch.where(module.mask == 0)[0].tolist()
        
# #         if len(prune_idx) == 0: continue
# #         if len(prune_idx) >= module.out_channels: continue

# #         # 물리적 채널 제거 및 연결된 BN/Depthwise 레이어 동시 조정
# #         group = DG.get_pruning_group(module, tp.prune_conv_out_channels, prune_idx)
# #         group.prune()

# # print(f"✅ MobileNet-V2 Physical pruning done for: {os.path.basename(ckpt_path)}")

# # # ================================
# # # 4️⃣ 파라미터 수 & 파일 크기 측정
# # # ================================
# # total_params = sum(p.numel() for p in model.parameters())
# # save_path = "./exp/checkpoints/mobilenet_temp_phys.pth"
# # torch.save(model.state_dict(), save_path)
# # size_mb = os.path.getsize(save_path) / (1024**2)
# # if os.path.exists(save_path): os.remove(save_path) 

# # # ================================
# # # 5️⃣ Peak & Average VRAM 측정 (Inference)
# # # ================================
# # if torch.cuda.is_available():
# #     torch.cuda.empty_cache()
# #     torch.cuda.reset_peak_memory_stats(device)

# # dummy = torch.randn(1, 3, 32, 32).to(device)
# # mems = []

# # with torch.no_grad():
# #     # Warm-up (GPU 안정화를 위해 10번 실행)
# #     for _ in range(10):
# #         _ = model(dummy)
    
# #     # 실제 샘플링 루프 (50번 반복하며 평균 측정)
# #     for _ in range(50):
# #         _ = model(dummy)
# #         # memory_allocated(): 현재 GPU에 실제로 할당된 메모리 양 기록
# #         mems.append(torch.cuda.memory_allocated(device) / (1024**2))

# # # 결과 산출
# # peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)
# # avg_vram = np.mean(mems)

# # # ================================
# # # 6️⃣ 결과 출력
# # # ================================
# # print("\n" + "="*45)
# # print(f"📊 MobileNet-V2 Physical Pruning Analysis: {os.path.basename(ckpt_path)}")
# # print("-" * 45)
# # print(f" - Total Params  : {total_params:,}")
# # print(f" - Model Size    : {size_mb:.2f} MB")
# # print(f" - Peak VRAM     : {peak_vram:.2f} MB")
# # print(f" - Average VRAM  : {avg_vram:.2f} MB")
# # print("="*45 + "\n")
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

# # 1. 모델 및 체크포인트 설정
# config, _ = load_config()
# config['model']['name'] = 'mobilenet_v2'
# config['model']['num_classes'] = 100
# model = get_model(config['model']).to(device)

# # 측정할 체크포인트 (파일명에 따라 결과가 확실히 달라져야 함)


# # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep40_sp10.10.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep60_sp20.31.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep80_sp28.55.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep100_sp36.05.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep120_sp47.38.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep140_sp89.68.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep160_sp89.68.pth" # 예시 경로
# # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep180_sp89.68.pth" # 예시 경로
# ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep180_sp89.61.pth" # 예시 경로

# # mobilenet_v2_DENSE_BASELINE.pth
# # mobilenet_v2_pdt_ep40_sp9.96.pth
# # mobilenet_v2_pdt_ep100_sp36.23.pth
# # mobilenet_v2_pdt_ep180_sp89.61.pth


# checkpoint = torch.load(ckpt_path, map_location=device)
# state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
# model.load_state_dict(state_dict, strict=False)
# model.eval()

# # 프루닝 전 파라미터 수 기록 (비교용)
# params_before = sum(p.numel() for p in model.parameters())

# # 2. Dependency Graph 구축
# example_inputs = torch.randn(1, 3, 32, 32).to(device)
# DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

# # 3. 모든 레이어를 전수 조사하여 물리적 삭제 수행
# pruning_count = 0
# for name, module in model.named_modules():
#     # 마스크가 존재하는 모든 Conv2d 레이어 탐색 (중첩 구조 포함)
#     if hasattr(module, "mask") and isinstance(module, nn.Conv2d):
#         mask = module.mask
#         prune_idx = torch.where(mask == 0)[0].tolist()
        
#         if len(prune_idx) > 0 and len(prune_idx) < module.out_channels:
#             group = DG.get_pruning_group(module, tp.prune_conv_out_channels, prune_idx)
#             group.prune()
#             pruning_count += 1

# # 4. MobileNet-V2 특화 Group/Channel 동기화 (이게 빠지면 VRAM이 안 줄어듦)
# for m in model.modules():
#     if isinstance(m, nn.Conv2d):
#         m.out_channels = m.weight.data.shape[0]
#         m.in_channels = m.weight.data.shape[1]
#         if m.groups > 1: # Depthwise
#             m.in_channels = m.out_channels
#             m.groups = m.out_channels

# params_after = sum(p.numel() for p in model.parameters())
# print(f"✅ Pruning Done: {pruning_count} layers modified.")
# print(f"✅ Params Change: {params_before:,} -> {params_after:,}")

# # 5. VRAM 측정
# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats(device)
# mems = []
# with torch.no_grad():
#     dummy = torch.randn(1, 3, 32, 32).to(device)
#     for _ in range(10): _ = model(dummy)
#     for _ in range(50):
#         _ = model(dummy)
#         mems.append(torch.cuda.memory_allocated(device) / (1024**2))

# print("\n" + "="*45)
# print(f"📊 Final Results: {os.path.basename(ckpt_path)}")
# print(f" - Peak VRAM: {torch.cuda.max_memory_allocated(device)/(1024**2):.2f} MB")
# print(f" - Avg VRAM : {np.mean(mems):.2f} MB")
# print("="*45)

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

# # 1️⃣ 모델 및 체크포인트 로드
# config, _ = load_config()
# config['model']['name'] = 'mobilenet_v2'
# config['model']['num_classes'] = 100
# model = get_model(config['model']).to(device)

# # 분석할 체크포인트 (사용자님의 파일명으로 변경)
# ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep200_sp91.68.pth" 

# # # mobilenet_v2_DENSE_BASELINE.pth
# # # mobilenet_v2_pdt_ep40_sp9.96.pth
# # # mobilenet_v2_pdt_ep100_sp36.23.pth
# # # mobilenet_v2_pdt_ep180_sp89.61.pth


# if not os.path.exists(ckpt_path):
#     print(f"❌ File not found: {ckpt_path}"); sys.exit()

# checkpoint = torch.load(ckpt_path, map_location=device)
# state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
# model.load_state_dict(state_dict, strict=False)
# model.eval()

# params_before = sum(p.numel() for p in model.parameters())
# print(f"🚀 Physical pruning starting for: {os.path.basename(ckpt_path)}")

# # 2️⃣ 의존성 그래프 구축
# example_inputs = torch.randn(1, 3, 32, 32).to(device)
# DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

# # 3️⃣ 물리적 프루닝 수행 (조건 강화)
# pruning_plan = []
# for name, m in model.named_modules():
#     # 마스크 속성이 있고 Conv2d인 경우
#     if isinstance(m, nn.Conv2d) and hasattr(m, "mask"):
#         # [핵심 수정] 마스크의 모든 차원을 평균내어 채널별 생존 여부 확인
#         # mask가 [C, 1, 1, 1] 혹은 [C]일 때 모두 대응 가능하도록 처리
#         mask_flat = m.mask.view(m.out_channels, -1).sum(1)
#         prune_idx = torch.where(mask_flat == 0)[0].tolist()
        
#         if len(prune_idx) > 0:
#             # 전체가 잘리는 것 방지
#             if len(prune_idx) >= m.out_channels:
#                 prune_idx = prune_idx[:-1]
            
#             pruning_plan.append((m, prune_idx))

# # 실제로 구조 변경 실행
# for module, idxs in pruning_plan:
#     # tp.prune_conv_out_channels를 사용하여 연쇄적인 삭제 유도
#     group = DG.get_pruning_group(module, tp.prune_conv_out_channels, idxs=idxs)
#     if DG.check_pruning_group(group):
#         group.prune()

# # 4️⃣ MobileNet-V2 Depthwise 구조 강제 동기화 (VRAM 감소 필수 단계)
# # 이 부분이 제대로 안 돌면 가중치만 변하고 실제 채널 수가 안 줄어듭니다.
# for m in model.modules():
#     if isinstance(m, nn.Conv2d):
#         m.out_channels = m.weight.shape[0]
#         m.in_channels = m.weight.shape[1]
#         if m.groups > 1: # Depthwise
#             m.groups = m.out_channels
#             m.in_channels = m.out_channels

# # 5️⃣ 지표 측정
# params_after = sum(p.numel() for p in model.parameters())

# # 실제 파일 크기 확인 (구조가 변경된 모델 저장)
# save_path = "./exp/checkpoints/temp_phys_final.pth"
# torch.save(model, save_path) 
# size_mb = os.path.getsize(save_path) / (1024**2)
# if os.path.exists(save_path): os.remove(save_path)

# # VRAM 측정
# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats(device)

# with torch.no_grad():
#     dummy = torch.randn(1, 3, 32, 32).to(device)
#     for _ in range(15): _ = model(dummy) # Warm-up
    
#     mems = []
#     for _ in range(50):
#         _ = model(dummy)
#         mems.append(torch.cuda.memory_allocated(device) / (1024**2))

# print("\n" + "="*45)
# print(f"📊 MobileNet-V2 Physical Analysis: {os.path.basename(ckpt_path)}")
# print("-" * 45)
# print(f" - Params Change  : {params_before:,} -> {params_after:,}")
# print(f" - Compression    : {(1 - params_after/params_before)*100:.2f}% reduced")
# print(f" - Model Size     : {size_mb:.2f} MB")
# print(f" - Peak VRAM      : {torch.cuda.max_memory_allocated(device)/(1024**2):.2f} MB")
# print(f" - Average VRAM   : {np.mean(mems):.2f} MB")
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

# # ------------------------------
# # 설정
# # ------------------------------
# CKPT_PATH = "./exp/checkpoints/mobilenet_v2_pdt_final.pth"
# INPUT_SIZE = (1, 3, 32, 32)


# # # 분석할 체크포인트 (사용자님의 파일명으로 변경)
# # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep200_sp91.68.pth" 

# # # # mobilenet_v2_DENSE_BASELINE.pth
# # # # mobilenet_v2_pdt_ep40_sp9.96.pth
# # # # mobilenet_v2_pdt_ep100_sp36.23.pth
# # # # mobilenet_v2_pdt_ep180_sp89.61.
# # mobilenet_v2_pdt_final.pth


# if not os.path.exists(CKPT_PATH):
#     print(f"❌ Checkpoint not found: {CKPT_PATH}")
#     sys.exit()

# print(f"🚀 Loading checkpoint: {os.path.basename(CKPT_PATH)}")

# # ------------------------------
# # 모델 생성
# # ------------------------------
# config, _ = load_config()
# config['model']['name'] = 'mobilenet_v2'
# config['model']['num_classes'] = 100

# model = get_model(config['model']).to(device)

# checkpoint = torch.load(CKPT_PATH, map_location=device)
# state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

# # ------------------------------
# # mask 복원 시도
# # ------------------------------
# mask_keys = [k for k in state_dict.keys() if "mask" in k]

# if len(mask_keys) == 0:
#     print("❌ No mask found in checkpoint. Structural pruning cannot proceed.")
#     sys.exit()

# model.load_state_dict(state_dict, strict=False)
# model.eval()

# params_before = sum(p.numel() for p in model.parameters())

# print(f"Params BEFORE physical pruning: {params_before:,}")

# # ------------------------------
# # Dependency Graph 생성
# # ------------------------------
# example_inputs = torch.randn(*INPUT_SIZE).to(device)
# DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

# # ------------------------------
# # Pruning Plan 생성
# # ------------------------------
# pruning_groups = []

# for module in model.modules():
#     if isinstance(module, nn.Conv2d) and hasattr(module, "mask"):

#         mask_flat = module.mask.view(module.out_channels, -1).sum(1)
#         prune_idx = torch.where(mask_flat == 0)[0].tolist()

#         if len(prune_idx) > 0 and len(prune_idx) < module.out_channels:
#             group = DG.get_pruning_group(
#                 module,
#                 tp.prune_conv_out_channels,
#                 prune_idx
#             )
#             if DG.check_pruning_group(group):
#                 pruning_groups.append(group)

# if len(pruning_groups) == 0:
#     print("⚠ No structural channels to prune.")
# else:
#     print(f"🔥 Applying {len(pruning_groups)} structural pruning groups...")
#     for g in pruning_groups:
#         g.prune()

# # ------------------------------
# # Depthwise 동기화 (MobileNet 필수)
# # ------------------------------
# for m in model.modules():
#     if isinstance(m, nn.Conv2d):
#         m.out_channels = m.weight.shape[0]
#         m.in_channels = m.weight.shape[1]
#         if m.groups > 1:
#             m.groups = m.out_channels
#             m.in_channels = m.out_channels

# params_after = sum(p.numel() for p in model.parameters())

# print(f"Params AFTER physical pruning: {params_after:,}")
# print(f"Compression Ratio: {(1 - params_after/params_before)*100:.2f}%")

# # ------------------------------
# # 모델 크기 측정
# # ------------------------------
# temp_path = "temp_structural_model.pth"
# torch.save(model.state_dict(), temp_path)
# size_mb = os.path.getsize(temp_path) / (1024**2)
# os.remove(temp_path)

# # ------------------------------
# # Inference VRAM 측정
# # ------------------------------
# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats()

# dummy = torch.randn(*INPUT_SIZE).to(device)

# with torch.no_grad():
#     # warmup
#     for _ in range(20):
#         _ = model(dummy)

#     torch.cuda.reset_peak_memory_stats()

#     mems = []
#     for _ in range(100):
#         _ = model(dummy)
#         mems.append(torch.cuda.memory_allocated() / (1024**2))

# peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
# avg_vram = np.mean(mems)

# # ------------------------------
# # 결과 출력
# # ------------------------------
# print("\n" + "="*55)
# print(f"📊 MobileNetV2 Structural Compression Result")
# print("-"*55)
# print(f"Params      : {params_before:,} → {params_after:,}")
# print(f"Model Size  : {size_mb:.2f} MB")
# print(f"Peak VRAM   : {peak_vram:.2f} MB")
# print(f"Average VRAM: {avg_vram:.2f} MB")
# print("="*55)



# import torch
# import os
# import numpy as np

# device = torch.device("cuda")

# CKPT_PATH = "./exp/checkpoints/mobilenet_v2_pdt_final.pth"
# import torch
# import os
# import numpy as np
# import sys

# sys.path.append(os.path.abspath("."))

# from src.models import get_model
# from src.utils.config_loader import load_config

# device = torch.device("cuda")

# CKPT_PATH = "./exp/checkpoints/mobilenet_v2_pdt_ep40_sp9.96.pth"

# # 모델 생성
# config, _ = load_config()
# config['model']['name'] = 'mobilenet_v2'
# config['model']['num_classes'] = 100

# model = get_model(config['model'])

# # state_dict 로드
# state_dict = torch.load(CKPT_PATH, map_location=device)
# model.load_state_dict(state_dict,strict=False)

# model.to(device)
# model.eval()

# # ---------------- Params ----------------
# params = sum(p.numel() for p in model.parameters())

# # ---------------- File Size ----------------
# temp_path = "temp_model.pth"
# torch.save(model.state_dict(), temp_path)
# size_mb = os.path.getsize(temp_path) / (1024**2)
# os.remove(temp_path)

# # ---------------- VRAM ----------------
# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats()

# dummy = torch.randn(1,3,32,32).to(device)

# with torch.no_grad():
#     for _ in range(20):
#         _ = model(dummy)

#     torch.cuda.reset_peak_memory_stats()

#     mems = []
#     for _ in range(100):
#         _ = model(dummy)
#         mems.append(torch.cuda.memory_allocated()/(1024**2))

# peak = torch.cuda.max_memory_allocated()/(1024**2)

# print("\n==== FINAL COMPRESSED RESULT ====")
# print("Params:", params)
# print("Size (MB):", size_mb)
# print("Peak VRAM (MB):", peak)
# print("Avg VRAM (MB):", np.mean(mems))
# print("=================================")


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
# # 1️⃣ 모델 설정 (MobileNetV2로 설정)
# # ================================
# config, _ = load_config()
# config['model']['name'] = 'mobilenet_v2'  # 모델명을 mobilenet_v2로 변경
# config['model']['num_classes'] = 100

# model = get_model(config['model']).to(device)

# # ================================
# # 2️⃣ 체크포인트 로드
# # ================================
# ckpt_path = "./exp/checkpoints//mobilenet_v2_pdt_ep200_sp91.68.pth" 


# # # # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep40_sp15.00.pth"
# # # # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep200_sp55.00.pth" # 예시 경로
# # # # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep200_sp55.00.pth" # 예시 경로

# # # # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep40_sp10.10.pth" # 예시 경로
# # # # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep60_sp20.31.pth" # 예시 경로
# # # # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep80_sp28.55.pth" # 예시 경로
# # # # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep100_sp36.05.pth" # 예시 경로
# # # # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep120_sp47.38.pth" # 예시 경로
# # # # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep140_sp89.68.pth" # 예시 경로
# # # # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep160_sp89.68.pth" # 예시 경로
# # # # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep180_sp89.68.pth" # 예시 경로
# # # ckpt_path = "./exp/checkpoints/mobilenet_v2_pdt_ep200_sp91.68.pth" # 예시 경로



# if not os.path.exists(ckpt_path):
#     print(f"❌ File not found: {ckpt_path}")
#     sys.exit()

# checkpoint = torch.load(ckpt_path, map_location=device)
# state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

# model.load_state_dict(state_dict, strict=False)
# model.eval()

# # ================================
# # 3️⃣ Physical Pruning (Weight 기반)
# # ================================
# example_inputs = torch.randn(1, 3, 32, 32).to(device)
# DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

# print("🔄 Analyzing Dependency Graph for MobileNetV2...")

# for name, module in model.named_modules():

#     if isinstance(module, nn.Conv2d):

#         weight = module.weight.data
#         out_channels = weight.shape[0]

#         # 채널별 L1 norm 계산
#         channel_norm = weight.view(out_channels, -1).abs().sum(dim=1)

#         # 완전히 0인 채널 찾기
#         prune_idx = torch.where(channel_norm == 0)[0].tolist()

#         if len(prune_idx) == 0:
#             continue

#         # 최소 1개 채널 보장
#         if len(prune_idx) >= out_channels:
#             prune_idx = prune_idx[:-1]

#         print(f"   -> Pruning Conv: {name} | {len(prune_idx)} channels removed")

#         group = DG.get_pruning_group(
#             module,
#             tp.prune_conv_out_channels,
#             prune_idx
#         )
#         group.prune()

# # 정리
# del DG
# del example_inputs
# gc.collect()
# torch.cuda.empty_cache()

# print("✅ Physical pruning done.")

# # 파라미터 수 확인
# total_params = sum(p.numel() for p in model.parameters())
# save_path = "./exp/checkpoints/mobilenet_temp_phys.pth"
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

# dummy = torch.randn(1, 3, 32, 32).to(device)

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
# print(f"📊 Detailed Memory Analysis: MobileNetV2")
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

# ------------------------------
# 1️⃣ 설정
# ------------------------------
DEVICE = "cuda"

# 🔥 체크포인트 바꿔가며 사용
CKPT_PATH = "./exp/checkpoints/mobilenet_v2_pdt_final.pth"
# CKPT_PATH = "./exp/checkpoints/mobilenetv2_DENSE_BASELINE.pth"

MODEL_NAME = "mobilenet_v2"

# ------------------------------
# 2️⃣ 모델 로드
# ------------------------------
print(f"Loading model from: {os.path.basename(CKPT_PATH)}")

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 100)

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