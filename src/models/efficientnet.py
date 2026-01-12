import copy
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# --- [0] 필터 카운트 (PAT 전략에서 원본 채널 참조용) ---
# EfficientNet-B0의 주요 MBConv 블록 내 conv_pw(Pointwise) 채널 수 정의
def get_efficientnet_filter_counts():
    # 주요 타겟이 되는 레이어들의 원본 채널 수를 반환
    # 예: "features.1.block.0.0" (첫 번째 MBConv의 pw conv)
    # 실제 레이어 이름은 모델 구조에 따라 달라질 수 있으므로 find_prunable_blocks와 연동
    return {
        # 필요 시 구체적인 레이어 이름을 key로, 채널 수를 value로 채움 
    }

# --- [1] 설계도: 모델 생성 및 마스크 등록 ---
def get_efficientnet(num_classes=100):
    # ImageNet 가중치를 사용하여 로드 (논문 설정 반영 가능)
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # 마지막 분류기(classifier) 클래스 수 조정
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    # [PDT] 각 Conv 레이어에 mask와 grad_ema 등록
    for m in model.modules():
        # 입력 채널이 3보다 큰 레이어(주로 MBConv 내부 레이어) 타겟
        if isinstance(m, nn.Conv2d) and m.in_channels > 3:
            m.register_buffer('mask', torch.ones(m.weight.shape[0]))
            m.register_buffer('grad_ema', torch.zeros(m.weight.shape[0]))
            
    return model

# --- [2] 물리적 프루닝 유틸리티 ---
def prune_conv_efficientnet(conv: nn.Conv2d, keep_idx, in_keep_idx=None):
    """EfficientNet 전용 Conv 프루닝 (Depthwise/Pointwise 구분)"""
    if not keep_idx:
        raise ValueError("Pruning produced zero output channels.")

    W = conv.weight.data.clone()
    B = conv.bias.data.clone() if conv.bias is not None else None

    # 1. 입력 채널 줄이기 (Pointwise 또는 일반 Conv이면서 그룹이 1인 경우)
    if in_keep_idx is not None and conv.groups == 1:
        W = W[:, in_keep_idx, :, :]
    
    # 2. 출력 채널 줄이기
    W = W[keep_idx, :, :, :]
    if B is not None:
        B = B[keep_idx]

    # 3. 새로운 레이어 생성 (Depthwise인 경우 groups를 출력 채널수와 동일하게 설정)
    new_conv = nn.Conv2d(
        in_channels=W.shape[1],
        out_channels=W.shape[0],
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=W.shape[0] if conv.groups > 1 else 1,
        bias=(conv.bias is not None),
    )
    new_conv.weight.data = W.clone()
    if B is not None:
        new_conv.bias.data = B.clone()

    return new_conv

def prune_efficientnet_blockwise(model, block_keep_indices, device):
    """EfficientNet-B0 물리적 프루닝 통합 함수"""
    model = copy.deepcopy(model).cpu()
    all_modules = dict(model.named_modules())
    
    # 설정된 블록 이름(예: features.2.block.0)을 순회하며 프루닝
    for name, module in model.named_modules():
        if name in block_keep_indices:
            keep_idx = block_keep_indices[name]
            
            if isinstance(module, nn.Conv2d):
                # 1. Conv 레이어 교체
                new_conv = prune_conv_efficientnet(module, keep_idx)
                parent_parts = name.rsplit('.', 1)
                parent = all_modules[parent_parts[0]]
                setattr(parent, parent_parts[1], new_conv)
                
                # 2. 관련 BN(BatchNorm) 레이어 업데이트
                # EfficientNet 구조상 conv 뒤에 바로 bn이 오는 규칙을 활용
                bn_name = name.replace('conv', 'bn').replace('pw', 'pw_bn').replace('dw', 'dw_bn')
                if bn_name in all_modules and isinstance(all_modules[bn_name], nn.BatchNorm2d):
                    bn = all_modules[bn_name]
                    new_bn = nn.BatchNorm2d(len(keep_idx))
                    new_bn.weight.data = bn.weight.data[keep_idx].clone()
                    new_bn.bias.data = bn.bias.data[keep_idx].clone()
                    new_bn.running_mean.data = bn.running_mean.data[keep_idx].clone()
                    new_bn.running_var.data = bn.running_var.data[keep_idx].clone()
                    
                    bn_parent_parts = bn_name.rsplit('.', 1)
                    bn_parent = all_modules[bn_parts[0]] if 'bn_parts' in locals() else all_modules[bn_parent_parts[0]]
                    setattr(bn_parent, bn_parent_parts[1], new_bn)

    return model.to(device)