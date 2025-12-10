# models.py
import torch.nn as nn
import torchvision
from torchvision.models import resnet as _tv_resnet

from pruning.efficientnet_utils import (
    build_efficientnet,
    find_prunable_blocks_efficientnet,
)

# ---------------------------------------------------------------
# Filter counts 설정
# ---------------------------------------------------------------

RESNET18_FILTER_COUNTS = {
    "layer1.0": 64,
    "layer1.1": 64,
    "layer2.0": 128,
    "layer2.1": 128,
    "layer3.0": 256,
    "layer3.1": 256,
    "layer4.0": 512,
    "layer4.1": 512,
}

RESNET152_FILTER_COUNTS = {}
for i in range(3):
    RESNET152_FILTER_COUNTS[f"layer1.{i}"] = 64
for i in range(8):
    RESNET152_FILTER_COUNTS[f"layer2.{i}"] = 128
for i in range(36):
    RESNET152_FILTER_COUNTS[f"layer3.{i}"] = 256
for i in range(3):
    RESNET152_FILTER_COUNTS[f"layer4.{i}"] = 512

VGG16_FILTER_COUNTS = {
    "features.0": 64,
    "features.2": 64,
    "features.5": 128,
    "features.7": 128,
    "features.10": 256,
    "features.12": 256,
    "features.14": 256,
    "features.17": 512,
    "features.19": 512,
    "features.21": 512,
    "features.24": 512,
    "features.26": 512,
    "features.28": 512,
}

# EfficientNet-B0: Conv2d 필터 수만 있으면 되므로, 전체 Conv2d를 dynamic하게 파악 가능
# 하지만 전략에서는 고정된 dict가 필요하므로, block 이름 + out_channels 기반으로 동적 구성 가능
EFFICIENTNET_FILTER_COUNTS = None  # 실행 시 생성하도록 함


# ---------------------------------------------------------------
# build_model
# ---------------------------------------------------------------
def build_model(model_id: str, num_classes: int = 100):
    """
    모델 이름에 따라 backbone 생성.
    """
    if model_id == "resnet18":
        m = torchvision.models.resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    elif model_id == "resnet152":
        m = torchvision.models.resnet152(weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    elif model_id == "vgg16":
        m = torchvision.models.vgg16(weights=None)
        m.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        in_feat = 512
        m.classifier = nn.Sequential(
            nn.Linear(in_feat, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        return m

    elif model_id == "efficientnet_b0":
        return build_efficientnet(num_classes)

    else:
        raise ValueError(f"Unknown model_id: {model_id}")


# ---------------------------------------------------------------
# find_prunable_blocks
# ---------------------------------------------------------------
def find_prunable_blocks(model, model_id: str):
    """
    전략에서 사용되는 블록 단위를 반환.
    """
    if model_id in ["resnet18", "resnet152"]:
        blocks = {
            n: md
            for n, md in model.named_modules()
            if isinstance(md, (_tv_resnet.BasicBlock, _tv_resnet.Bottleneck))
        }
        return blocks

    elif model_id == "vgg16":
        blocks = {}
        for idx, m in enumerate(model.features):
            if isinstance(m, nn.Conv2d):
                name = f"features.{idx}"
                blocks[name] = m
        return blocks

    elif model_id == "efficientnet_b0":
        return find_prunable_blocks_efficientnet(model)

    else:
        raise ValueError(f"Unknown model_id: {model_id}")


# ---------------------------------------------------------------
# get_filter_counts
# ---------------------------------------------------------------
def get_filter_counts(model_id: str):
    """
    block-wise pruning 전략에서 block별 원래 필터 수를 리턴.
    """
    global EFFICIENTNET_FILTER_COUNTS

    if model_id == "resnet18":
        return RESNET18_FILTER_COUNTS

    elif model_id == "resnet152":
        return RESNET152_FILTER_COUNTS

    elif model_id == "vgg16":
        return VGG16_FILTER_COUNTS

    elif model_id == "efficientnet_b0":
        # 최초 1회 호출 시 build
        if EFFICIENTNET_FILTER_COUNTS is None:
            EFFICIENTNET_FILTER_COUNTS = {}

            tmp = build_efficientnet(num_classes=100)
            blocks = find_prunable_blocks_efficientnet(tmp)

            for name, m in blocks.items():
                EFFICIENTNET_FILTER_COUNTS[name] = m.out_channels

        return EFFICIENTNET_FILTER_COUNTS

    else:
        raise ValueError(f"Unknown model_id: {model_id}")
