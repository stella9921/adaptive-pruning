import copy
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    resnet,
)
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)


_RESNET_BUILDERS = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
}

_RESNET_WEIGHTS = {
    "resnet18": ResNet18_Weights.DEFAULT,
    "resnet34": ResNet34_Weights.DEFAULT,
    "resnet50": ResNet50_Weights.DEFAULT,
    "resnet101": ResNet101_Weights.DEFAULT,
    "resnet152": ResNet152_Weights.DEFAULT,
}

# --- [0] 필터 카운트 (PAT) ---
# BasicBlock용 (18, 34)
RESNET_BASIC_FILTER_COUNTS = {
    "layer1.0": 64, "layer1.1": 64, 
    "layer2.0": 128, "layer2.1": 128,
    "layer3.0": 256, "layer3.1": 256,
    "layer4.0": 512, "layer4.1": 512,
}

# Bottleneck용 (50, 101, 152) - 각 블록의 중간 conv2(3x3) 채널 수
def get_bottleneck_filter_counts(name):
    counts = {}
    layers = [3, 4, 6, 3] if name == "resnet50" else ([3, 4, 23, 3] if name == "resnet101" else [3, 8, 36, 3])
    channels = [64, 128, 256, 512]
    for i, num_blocks in enumerate(layers):
        for j in range(num_blocks):
            counts[f"layer{i+1}.{j}"] = channels[i]
    return counts

# --- [1] 설계도: 모델 생성 및 마스크 등록 ---
def _get_resnet_legacy(name="resnet18", num_classes=100):
    # 모델 로드
    if name == "resnet18": model = resnet18(weights=None)
    elif name == "resnet34": model = resnet34(weights=None)
    elif name == "resnet50": model = resnet50(weights=None)
    elif name == "resnet101": model = resnet101(weights=None)
    elif name == "resnet152": 
        from torchvision.models import resnet152
        model = resnet152(weights=None)
    
    #  CIFAR-100용 구조 수정
    # 1. 초기 7x7 conv를 3x3으로 변경 (이미지 사이즈가 작으므로)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # 2. Maxpool 제거 (Identity로 대체하여 해상도 유지)
    model.maxpool = nn.Identity()
    # 3. FC 레이어 클래스 수 조정
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # [PDT] 각 Conv 레이어에 mask와 grad_ema 등록
    for m in model.modules():
        if isinstance(m, nn.Conv2d): # 모든 Conv 레이어에 마스크와 EMA 등록
            m.register_buffer('mask', torch.ones(m.weight.shape[0], device=m.weight.device))
            m.register_buffer('grad_ema', torch.zeros(m.weight.shape[0], device=m.weight.device))
        # if isinstance(m, nn.Conv2d) and m.in_channels > 3:
        #     # 출력 채널 크기에 맞춘 버퍼 등록
        #     m.register_buffer('mask', torch.ones(m.weight.shape[0]))
        #     m.register_buffer('grad_ema', torch.zeros(m.weight.shape[0]))

        
            
    return model

# --- [2] 물리적 프루닝 유틸리티 ---
def get_resnet(name="resnet18", num_classes=100, input_size=32, pretrained=False):
    if name not in _RESNET_BUILDERS:
        raise ValueError(f"Unsupported ResNet variant: {name}")

    weights = _RESNET_WEIGHTS[name] if pretrained else None
    model = _RESNET_BUILDERS[name](weights=weights)

    if int(input_size) < 128:
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module.register_buffer(
                'mask',
                torch.ones(module.weight.shape[0], device=module.weight.device)
            )
            module.register_buffer(
                'grad_ema',
                torch.zeros(module.weight.shape[0], device=module.weight.device)
            )

    return model


def prune_conv_and_bn(conv, bn, keep_idx, in_keep_idx=None):
    """Conv2d와 BatchNorm2d 쌍의 채널을 물리적으로 깎아냄"""
    W = conv.weight.data.clone()
    B = conv.bias.data.clone() if conv.bias is not None else None

    # 출력 채널 슬라이싱
    W = W[keep_idx, :, :, :]
    if B is not None: B = B[keep_idx]

    # 입력 채널 슬라이싱 (이전 레이어의 keep_idx 영향)
    if in_keep_idx is not None:
        W = W[:, in_keep_idx, :, :]

    new_conv = nn.Conv2d(W.shape[1], W.shape[0], conv.kernel_size, 
                         conv.stride, conv.padding, bias=(conv.bias is not None))
    new_conv.weight.data = W.clone()
    if B is not None: new_conv.bias.data = B.clone()

    new_bn = nn.BatchNorm2d(len(keep_idx))
    new_bn.weight.data = bn.weight.data[keep_idx].clone()
    new_bn.bias.data = bn.bias.data[keep_idx].clone()
    new_bn.running_mean = bn.running_mean[keep_idx].clone()
    new_bn.running_var = bn.running_var[keep_idx].clone()

    return new_conv, new_bn

def prune_resnet_blockwise(model, block_keep_indices, device):
    """ResNet 전 계열(18~152) 물리적 프루닝 통합 함수"""
    model = copy.deepcopy(model).cpu()
    
    # 모든 블록(BasicBlock, Bottleneck) 추출
    block_items = [
        (n, m) for n, m in model.named_modules() 
        if isinstance(m, (resnet.BasicBlock, resnet.Bottleneck))
    ]
    
    # 초기 입력 채널 인덱스 (model.conv1의 출력)
    prev_out_indices = list(range(model.conv1.out_channels))

    for name, block in block_items:
        # --- CASE 1: BasicBlock (18, 34) ---
        if isinstance(block, resnet.BasicBlock):
            keep_idx = block_keep_indices.get(name, list(range(block.conv1.out_channels)))
            
            # conv1(in:이전, out:keep) / conv2(in:keep, out:keep)
            block.conv1, block.bn1 = prune_conv_and_bn(block.conv1, block.bn1, keep_idx, prev_out_indices)
            block.conv2, block.bn2 = prune_conv_and_bn(block.conv2, block.bn2, keep_idx, keep_idx)
            
            # Shortcut(downsample) 처리
            if block.downsample is not None:
                ds_conv, ds_bn = block.downsample[0], block.downsample[1]
                new_ds_conv, new_ds_bn = prune_conv_and_bn(ds_conv, ds_bn, keep_idx, prev_out_indices)
                block.downsample = nn.Sequential(new_ds_conv, new_ds_bn)
            
            # BasicBlock은 블록 출력이 keep_idx로 바뀌므로 업데이트
            prev_out_indices = keep_idx

        # --- CASE 2: Bottleneck (50, 101, 152) ---
        elif isinstance(block, resnet.Bottleneck):
            # 논문 및 기존 코드 전략: 중간 3x3 conv(conv2)의 채널을 줄임
            keep_idx = block_keep_indices.get(name, list(range(block.conv2.out_channels)))
            
            # conv1: 1x1 (out:keep) / conv2: 3x3 (in:keep, out:keep)
            block.conv1, block.bn1 = prune_conv_and_bn(block.conv1, block.bn1, keep_idx)
            block.conv2, block.bn2 = prune_conv_and_bn(block.conv2, block.bn2, keep_idx, keep_idx)
            
            # conv3: 1x1 (in:keep, out:고정) - 블록 최종 출력 채널은 유지
            W_c3 = block.conv3.weight.data[:, keep_idx, :, :]
            new_c3 = nn.Conv2d(W_c3.shape[1], block.conv3.out_channels, 1, stride=1, bias=False)
            new_c3.weight.data = W_c3.clone()
            block.conv3 = new_c3
            
            # Bottleneck은 최종 출력 채널(conv3)을 원본대로 유지하므로 prev_out_indices 업데이트 불필요

    # 마지막 FC 레이어 입력 차원 재조정
    last_block = block_items[-1][1]
    if isinstance(last_block, resnet.BasicBlock):
        final_in = last_block.bn2.num_features
    else: # Bottleneck
        final_in = last_block.bn3.num_features
    model.fc = nn.Linear(final_in, model.fc.out_features)

    return model.to(device)
