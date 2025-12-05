# models.py
import torch.nn as nn
import torchvision
from torchvision.models import resnet as _tv_resnet

# ResNet18에서 우리가 "블록"이라고 보는 단위별 필터 개수
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

# ResNet152에서 Bottleneck 블록별 mid-channel(conv2.out_channels) 개수
# layer1: 3개(64), layer2: 8개(128), layer3: 36개(256), layer4: 3개(512)
RESNET152_FILTER_COUNTS = {}

for i in range(3):
    RESNET152_FILTER_COUNTS[f"layer1.{i}"] = 64
for i in range(8):
    RESNET152_FILTER_COUNTS[f"layer2.{i}"] = 128
for i in range(36):
    RESNET152_FILTER_COUNTS[f"layer3.{i}"] = 256
for i in range(3):
    RESNET152_FILTER_COUNTS[f"layer4.{i}"] = 512

# VGG16에서 "블록"으로 쓸 conv 레이어들 (features의 Conv2d들)
# conv 위치: 0,2,5,7,10,12,14,17,19,21,24,26,28
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


def build_model(model_id: str, num_classes: int = 100):
    """
    backbone 종류에 따라 모델 생성.
    - resnet18 / resnet152: CIFAR용으로 conv1/stride 수정 + maxpool 제거
    - vgg16: CIFAR용으로 classifier/avgpool 수정
    """
    if model_id == "resnet18":
        m = torchvision.models.resnet18(weights=None)
        # CIFAR-100용 설정
        m.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    elif model_id == "resnet152":
        m = torchvision.models.resnet152(weights=None)
        # CIFAR-100용 설정 (18과 동일하게 3x3, stride=1, maxpool 제거)
        m.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    elif model_id == "vgg16":
        m = torchvision.models.vgg16(weights=None)

        # CIFAR-100용: 마지막을 1x1로 줄이도록 avgpool 설정
        m.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # features 출력: [N, 512, 1, 1] 기준으로 classifier 구성
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
        raise NotImplementedError("efficientnet_b0 아직 안 붙였음")

    else:
        raise ValueError(f"Unknown model_id: {model_id}")


def find_prunable_blocks(model, model_id: str):
    """
    전략에서 사용할 '블록 단위'를 리턴.
    각 블록 이름에 해당하는 모듈 dict를 돌려줌.
    - ResNet18/152: BasicBlock 또는 Bottleneck 단위
    - VGG16: features 내부 Conv2d 레이어 단위
    """
    # 공통 ResNet 계열(18, 34, 50, 101, 152 등)에 대해 BasicBlock/Bottleneck 모두 지원
    if model_id in ["resnet18", "resnet152"]:
        blocks = {
            n: md
            for n, md in model.named_modules()
            if isinstance(md, (_tv_resnet.BasicBlock, _tv_resnet.Bottleneck))
        }
        return blocks

    elif model_id == "vgg16":
        # VGG16: features 내부 Conv2d 레이어를 블록으로 사용
        blocks = {}
        for idx, m in enumerate(model.features):
            if isinstance(m, nn.Conv2d):
                name = f"features.{idx}"
                blocks[name] = m
        return blocks

    elif model_id == "efficientnet_b0":
        raise NotImplementedError("efficientnet_b0 블록 정의 아직 안 함")

    else:
        raise ValueError(f"Unknown model_id: {model_id}")


def get_filter_counts(model_id: str):
    """
    전략 1(vanilla)에서 쓰는 '원본 필터 개수' dict를 리턴.
    모델별로 다르게 정의.
    - ResNet: block별 mid-channel(conv2 out_channels) 기준
    - VGG16: Conv out_channels 기준
    """
    if model_id == "resnet18":
        return RESNET18_FILTER_COUNTS

    elif model_id == "resnet152":
        return RESNET152_FILTER_COUNTS

    elif model_id == "vgg16":
        return VGG16_FILTER_COUNTS

    elif model_id == "efficientnet_b0":
        raise NotImplementedError("efficientnet_b0 filter count 아직 안 정의")

    else:
        raise ValueError(f"Unknown model_id: {model_id}")
