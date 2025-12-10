# pruning/efficientnet_utils.py
import copy
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


###############################################################################
# 1) Conv + BN prune (ResNet과 동일한 구조)
###############################################################################
def prune_conv_and_bn(conv, bn, keep_idx, in_keep_idx=None):
    """
    EfficientNet에서도 Conv2d + BatchNorm2d pruning 방식은 동일하게 쓸 수 있음.
    out_channels: keep_idx 기준으로 선택
    in_channels: in_keep_idx 기준으로 선택
    """
    if not keep_idx:
        raise ValueError("Pruning produced zero output channels in EfficientNet.")

    W = conv.weight.data.clone()
    B = conv.bias.data.clone() if conv.bias is not None else None

    # out-ch prune
    W = W[keep_idx, :, :, :]
    if B is not None:
        B = B[keep_idx]

    # in-ch prune
    if in_keep_idx is not None:
        W = W[:, in_keep_idx, :, :]

    new_conv = nn.Conv2d(
        in_channels=W.shape[1],
        out_channels=W.shape[0],
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=1 if conv.groups == conv.in_channels else conv.groups,
        bias=(conv.bias is not None),
    )
    new_conv.weight.data = W.clone()
    if B is not None:
        new_conv.bias.data = B.clone()

    # BN 재생성
    new_bn = nn.BatchNorm2d(len(keep_idx))
    new_bn.weight.data = bn.weight.data[keep_idx].clone()
    new_bn.bias.data = bn.bias.data[keep_idx].clone()
    new_bn.running_mean = bn.running_mean[keep_idx].clone()
    new_bn.running_var = bn.running_var[keep_idx].clone()

    return new_conv, new_bn


###############################################################################
# 2) Conv input-channel만 prune (projection conv 용)
###############################################################################
def prune_conv_in_channels(conv, in_keep_idx):
    if in_keep_idx is None:
        return conv

    W = conv.weight.data.clone()
    if conv.bias is not None:
        B = conv.bias.data.clone()
    else:
        B = None

    W = W[:, in_keep_idx, :, :]

    new_conv = nn.Conv2d(
        in_channels=W.shape[1],
        out_channels=W.shape[0],
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=1 if conv.groups == conv.in_channels else conv.groups,
        bias=(conv.bias is not None),
    )
    new_conv.weight.data = W.clone()
    if B is not None:
        new_conv.bias.data = B.clone()

    return new_conv


###############################################################################
# 3) EfficientNet-B0 모델 로더
###############################################################################
def build_efficientnet(num_classes=100):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # classifier 변경
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


###############################################################################
# 4) EfficientNet 내 Prunable blocks 찾기
###############################################################################
def find_prunable_blocks_efficientnet(model):
    """
    EfficientNet의 모든 Conv2d를 pruning 대상으로 사용.
    block-wise pruning까지는 아직 확장 전이므로 Conv 단위만 제공.
    """
    blocks = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            blocks[name] = module
    return blocks


###############################################################################
# 5) EfficientNet block-wise pruning (확장 가능하게 기본 틀만 제공)
###############################################################################
def prune_efficientnet_blockwise(model, block_keep_indices, device):
    """
    현재는 Conv-level pruning을 기반으로 동작하도록 설계.
    MBConvBlock 내부 구조(확장 conv -> depthwise conv -> projection conv)를
    정교하게 pruning하려면 이후 확장 필요.
    """
    model = copy.deepcopy(model).cpu()

    # Conv2d만 찾아서 prune 수행 (VGG 방식과 같음)
    for name, module in model.named_modules():
        if name in block_keep_indices and isinstance(module, nn.Conv2d):
            keep_idx = block_keep_indices[name]
            # Conv에 이어지는 BN 찾기
            parent = name.rsplit(".", 1)[0]
            bn_name = parent + ".bn"
            bn = dict(model.named_modules()).get(bn_name, None)

            if bn is not None:
                new_conv, new_bn = prune_conv_and_bn(module, bn, keep_idx)
                setattr_module(model, name, new_conv)
                setattr_module(model, bn_name, new_bn)

    return model.to(device)


###############################################################################
# 6) Utility: module 교체 함수
###############################################################################
def setattr_module(model, name, new_module):
    """'features.3.0' 처럼 nested module을 안전하게 교체하는 함수"""
    parts = name.split(".")
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], new_module)

