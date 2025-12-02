# models.py
import torch.nn as nn
import torchvision

# 🔹 ResNet18에서 우리가 "블록"이라고 보는 단위별 필터 개수 (이미 쓰던 거)
RESNET18_FILTER_COUNTS = {
    'layer1.0': 64,
    'layer1.1': 64,
    'layer2.0': 128,
    'layer2.1': 128,
    'layer3.0': 256,
    'layer3.1': 256,
    'layer4.0': 512,
    'layer4.1': 512,
}


def build_model(model_id: str, num_classes: int = 100):
    """
    backbone 종류에 따라 모델 생성.
    지금은 resnet18만 구현, 나중에 vgg/efficientnet 추가.
    """
    if model_id == "resnet18":
        m = torchvision.models.resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    elif model_id == "vgg16":
        # TODO: 나중에 VGG 버전 추가
        # 예시:
        # m = torchvision.models.vgg16(weights=None)
        # m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        # return m
        raise NotImplementedError("vgg16 아직 안 붙였음")

    elif model_id == "efficientnet_b0":
        # TODO: 나중에 EfficientNet 버전 추가
        raise NotImplementedError("efficientnet_b0 아직 안 붙였음")

    else:
        raise ValueError(f"Unknown model_id: {model_id}")


def find_prunable_blocks(model, model_id: str):
    """
    전략에서 사용할 '블록 단위'를 리턴.
    각 블록 이름에 해당하는 모듈 dict를 돌려줌.
    """
    if model_id == "resnet18":
        return {
            n: md
            for n, md in model.named_modules()
            if isinstance(md, torchvision.models.resnet.BasicBlock)
        }

    elif model_id == "vgg16":
        # 여기서는 예를 들어 conv layer 그룹을 block으로 묶는 식으로 구현해야 함.
        # ex) "features.0~1"을 block1, "features.3~4"를 block2 이런 식
        raise NotImplementedError("vgg16 블록 정의 아직 안 함")

    elif model_id == "efficientnet_b0":
        # 마찬가지로 MBConv 단위로 block 정의
        raise NotImplementedError("efficientnet_b0 블록 정의 아직 안 함")

    else:
        raise ValueError(f"Unknown model_id: {model_id}")


def get_filter_counts(model_id: str):
    """
    전략 1(vanilla)에서 쓰는 '원본 필터 개수' dict를 리턴.
    모델별로 다르게 정의해야 함.
    """
    if model_id == "resnet18":
        return RESNET18_FILTER_COUNTS

    elif model_id == "vgg16":
        # TODO: VGG에서 블록을 어떻게 정의할지 정한 뒤 dict 작성
        raise NotImplementedError("vgg16 filter count 아직 안 정의")

    elif model_id == "efficientnet_b0":
        raise NotImplementedError("efficientnet_b0 filter count 아직 안 정의")

    else:
        raise ValueError(f"Unknown model_id: {model_id}")
