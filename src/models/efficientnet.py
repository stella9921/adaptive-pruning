import copy
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# --- [1] 설계도: 모델 생성 및 마스크 등록 ---
def get_efficientnet(num_classes=100):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # 마지막 분류기 조정
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    # [PDT] 입력 채널이 3보다 큰 모든 Conv에 마스크 등록
    # MBConv 내부의 확장(Pointwise), Depthwise, 축소 레이어를 모두 포함함
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and m.in_channels > 3:
            m.register_buffer('mask', torch.ones(m.weight.shape[0]))
            m.register_buffer('grad_ema', torch.zeros(m.weight.shape[0]))
            
    return model

# --- [2] 물리적 프루닝 유틸리티 ---
def prune_conv_efficientnet(conv: nn.Conv2d, keep_idx, in_keep_idx=None):
    """EfficientNet 전용 Conv 프루닝 (Depthwise 대응)"""
    W = conv.weight.data.clone()
    B = conv.bias.data.clone() if conv.bias is not None else None

    # Depthwise Conv (groups > 1)인 경우: 입력과 출력이 동시에 똑같이 잘려야 함
    if conv.groups > 1:
        W = W[keep_idx, :, :, :] # DW는 in_channels가 항상 1이므로 0번 차원만 슬라이싱
    else:
        # 일반 Conv 혹은 Pointwise Conv
        if in_keep_idx is not None:
            W = W[:, in_keep_idx, :, :]
        W = W[keep_idx, :, :, :]

    if B is not None:
        B = B[keep_idx]

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
    """
    16개 MBConv 그룹 전략을 물리적으로 반영하는 함수
    block_keep_indices: { 'features.2.0': [indices...], ... } 형태
    """
    model = copy.deepcopy(model).cpu()
    all_modules = dict(model.named_modules())
    
    for block_base_name, keep_idx in block_keep_indices.items():
        if block_base_name not in all_modules: continue
        
        block = all_modules[block_base_name] # MBConv 객체
        
        # 1. 확장(Expansion) 또는 첫 번째 Conv (block[0])
        if hasattr(block, '0') and isinstance(block[0], nn.Sequential):
            # block[0]은 보통 [Conv, BN, Act] 세트
            target_conv = block[0][0]
            new_conv = prune_conv_efficientnet(target_conv, keep_idx)
            block[0][0] = new_conv
            
            # BN 업데이트
            bn = block[0][1]
            new_bn = nn.BatchNorm2d(len(keep_idx))
            # ... (BN 데이터 복사 로직 - 아래 유틸리티로 대체 가능)
            block[0][1] = new_bn

        # 2. Depthwise Conv (block[1])
        if hasattr(block, '1') and isinstance(block[1], nn.Sequential):
            target_dw = block[1][0]
            # DW는 입력과 출력을 동일한 keep_idx로 잘라야 함
            new_dw = prune_conv_efficientnet(target_dw, keep_idx)
            block[1][0] = new_dw
            block[1][1] = nn.BatchNorm2d(len(keep_idx)) # BN 업데이트

        # 3. SE Layer (Squeeze-and-Excitation) (block[2])
        # SE 레이어는 채널 중요도를 계산하므로 채널 수 변화에 아주 민감함
        if hasattr(block, '2') and hasattr(block[2], 'fc1'):
            se = block[2]
            # fc1: in_channels -> squeezed_channels (보통 1/4)
            # Rebuild fc1 for the pruned input channels
            old_fc1 = se.fc1
            new_fc1 = nn.Conv2d(len(keep_idx), old_fc1.out_channels, 1)
            se.fc1 = new_fc1
            
            # fc2: squeezed_channels -> out_channels (다시 원래대로)
            old_fc2 = se.fc2
            new_fc2 = nn.Conv2d(old_fc2.in_channels, len(keep_idx), 1)
            se.fc2 = new_fc2

        # 4. 축소(Projection) 레이어 (block[3])
        if hasattr(block, '3') and isinstance(block[3], nn.Sequential):
            target_proj = block[3][0]
            # Projection 레이어는 '입력'이 keep_idx만큼 줄어든 상태임
            # 출력(out_channels)은 다음 Residual을 위해 유지하거나 별도 처리
            new_proj = prune_conv_efficientnet(target_proj, list(range(target_proj.out_channels)), in_keep_idx=keep_idx)
            block[3][0] = new_proj

    return model.to(device)
