import copy
import torch
import torch.nn as nn
from torchvision.models import vgg16_bn

# PAT 전략에서 참조할 원본 필터 개수
VGG16_FILTER_COUNTS = {
    "features.0": 64, "features.3": 64, "features.7": 128, "features.10": 128,
    "features.14": 256, "features.17": 256, "features.20": 256,
    "features.24": 512, "features.27": 512, "features.30": 512,
    "features.34": 512, "features.37": 512, "features.40": 512,
}

# --- [1] 설계도: 모델 생성 및 마스크 등록 ---
def get_vgg16(num_classes=100):
    # vgg16_bn을 사용하여 BatchNorm 포함 (학습 안정성)
    model = vgg16_bn(weights=None)
    
    # CIFAR-100: 마지막을 1x1로 줄이도록 avgpool 설정
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    # 분류기 재구성
    model.classifier = nn.Sequential(
        nn.Linear(512, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )
    
    # PDT용 버퍼 등록 (중요: 입력 채널 3 초과하는 모든 Conv에 등록)
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.in_channels > 3:
            m.register_buffer('mask', torch.ones(m.weight.shape[0]))
            m.register_buffer('grad_ema', torch.zeros(m.weight.shape[0]))
            
    return model

# --- [2] 물리적 프루닝 함수 ---
def prune_conv_vgg(conv: nn.Conv2d, keep_idx, in_keep_idx=None):
    if not keep_idx: raise ValueError("Zero channels.")
    
    W = conv.weight.data.clone()
    B = conv.bias.data.clone() if conv.bias is not None else None
    
    if in_keep_idx is not None: 
        W = W[:, in_keep_idx, :, :]
    W = W[keep_idx, :, :, :]
    if B is not None: 
        B = B[keep_idx]
        
    new_conv = nn.Conv2d(W.shape[1], W.shape[0], conv.kernel_size, conv.stride, conv.padding, bias=(conv.bias is not None))
    new_conv.weight.data = W.clone()
    if B is not None: 
        new_conv.bias.data = B.clone()
    return new_conv

def prune_vgg_blockwise(model, block_keep_indices, device, num_classes=100):
    model = copy.deepcopy(model).cpu()
    conv_items = []
    
    # features 내의 Conv2d 찾기
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name.startswith("features."):
            conv_items.append((int(name.split(".")[1]), name, module))
    conv_items.sort(key=lambda x: x[0])

    prev_out_indices = None
    last_conv_name = None
    
    for idx, full_name, conv in conv_items:
        keep_idx = block_keep_indices.get(full_name, list(range(conv.out_channels)))
        new_conv = prune_conv_vgg(conv, keep_idx, prev_out_indices)
        
        # 실제 모델 레이어 교체
        parent = model.features
        parent[int(full_name.split(".")[1])] = new_conv
        
        # BN이 있는 경우 BN도 함께 프루닝 (vgg16_bn 기준)
        bn_idx = int(full_name.split(".")[1]) + 1
        if bn_idx < len(model.features) and isinstance(model.features[bn_idx], nn.BatchNorm2d):
            bn = model.features[bn_idx]
            new_bn = nn.BatchNorm2d(len(keep_idx))
            new_bn.weight.data = bn.weight.data[keep_idx].clone()
            new_bn.bias.data = bn.bias.data[keep_idx].clone()
            new_bn.running_mean = bn.running_mean[keep_idx].clone()
            new_bn.running_var = bn.running_var[keep_idx].clone()
            model.features[bn_idx] = new_bn
            
        prev_out_indices = keep_idx
        last_conv_name = full_name

    # 분류기 첫 번째 레이어 재구성
    last_conv = model.features[int(last_conv_name.split(".")[1])]
    model.classifier[0] = nn.Linear(last_conv.out_channels, 4096)
    
    return model.to(device)