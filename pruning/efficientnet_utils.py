import copy
import torch
import torch.nn as nn

def prune_conv_efficientnet(conv: nn.Conv2d, keep_idx, in_keep_idx=None):

    if not keep_idx:
        raise ValueError("Pruning produced zero output channels.")

    W = conv.weight.data.clone()
    B = conv.bias.data.clone() if conv.bias is not None else None

    # 1. 입력 채널 줄이기 (Pointwise 혹은 일반 Conv인 경우)
    if in_keep_idx is not None and conv.groups == 1:
        W = W[:, in_keep_idx, :, :]
    
    # 2. 출력 채널 줄이기
    W = W[keep_idx, :, :, :]
    if B is not None:
        B = B[keep_idx]

    # 3. 새로운 레이어 구성
    new_conv = nn.Conv2d(
        in_channels=W.shape[1],
        out_channels=W.shape[0],
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        # Depthwise Conv인 경우 groups를 출력 채널 수와 동일하게 설정
        groups=W.shape[0] if conv.groups > 1 else 1,
        bias=(conv.bias is not None),
    )
    new_conv.weight.data = W.clone()
    if B is not None:
        new_conv.bias.data = B.clone()

    return new_conv

def prune_efficientnet_blockwise(model, block_keep_indices, device):

    model = copy.deepcopy(model).cpu()
    
    # EfficientNet-B0의 주요 프루닝 대상 레이어 (MBConv 내의 conv_pw 등)

    for name, module in model.named_modules():
        if name in block_keep_indices:
            keep_idx = block_keep_indices[name]
            
            # 1. 현재 Conv 레이어 프루닝
            if isinstance(module, nn.Conv2d):
                new_conv = prune_conv_efficientnet(module, keep_idx)
                
                # 상위 모듈에서 교체
                parent_parts = name.rsplit('.', 1)
                if len(parent_parts) > 1:
                    parent = dict(model.named_modules())[parent_parts[0]]
                    setattr(parent, parent_parts[1], new_conv)
                else:
                    setattr(model, parent_parts[0], new_conv)

                # 2. 관련 Batch Normalization 레이어 업데이트
                # 보통 Conv 바로 다음에 BN이 오는 구조 (예: name이 '...conv' 이면 '...bn' 찾기)
                bn_name = name.replace('conv', 'bn').replace('pw', 'pw_bn').replace('dw', 'dw_bn')
                all_modules = dict(model.named_modules())
                if bn_name in all_modules and isinstance(all_modules[bn_name], nn.BatchNorm2d):
                    bn = all_modules[bn_name]
                    new_bn = nn.BatchNorm2d(len(keep_idx)).to(device)
                    
                    # 가중치 복사
                    new_bn.weight.data = bn.weight.data[keep_idx].clone()
                    new_bn.bias.data = bn.bias.data[keep_idx].clone()
                    new_bn.running_mean.data = bn.running_mean.data[keep_idx].clone()
                    new_bn.running_var.data = bn.running_var.data[keep_idx].clone()
                    
                    # BN 교체
                    bn_parent_parts = bn_name.rsplit('.', 1)
                    bn_parent = dict(model.named_modules())[bn_parent_parts[0]]
                    setattr(bn_parent, bn_parent_parts[1], new_bn)

    return model.to(device)