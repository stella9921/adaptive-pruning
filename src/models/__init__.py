# src/models/__init__.py
from .resnet import get_resnet, prune_resnet_blockwise
from .vgg import get_vgg16, prune_vgg_blockwise
from .efficientnet import get_efficientnet, prune_efficientnet_blockwise

def get_model(model_cfg):
    """
    YAML의 model 섹션 설정을 받아 모델 객체를 생성함
    """
    name = model_cfg['name'].lower()
    num_classes = model_cfg.get('num_classes', 100)
    
    # name에 'resnet18', 'resnet152' 등이 들어와도 get_resnet이 처리함
    if "resnet" in name:
        return get_resnet(name, num_classes)
    elif "vgg" in name:
        return get_vgg16(num_classes)
    elif "efficientnet" in name:
        return get_efficientnet(num_classes)
    else:
        raise ValueError(f"지원하지 않는 모델 이름입니다: {name}")

def get_physical_prune_fn(model_name):
    """
    PAT 결과가 나온 후, 실제로 모델 채널을 자르는 함수를 반환함
    """
    name = model_name.lower()
    if "resnet" in name:
        return prune_resnet_blockwise
    elif "vgg" in name:
        return prune_vgg_blockwise
    elif "efficientnet" in name:
        return prune_efficientnet_blockwise
    else:
        raise ValueError(f"물리적 프루닝 함수를 찾을 수 없습니다: {name}")

def find_prunable_blocks(model, model_name):
    """
    모델별로 민감도 측정이나 마스킹 타겟이 되는 블록(레이어)들을 찾아줌
    """
    name = model_name.lower()
    blocks = {}
    
    if "resnet" in name:
        from torchvision.models import resnet as _tv_resnet
        for n, m in model.named_modules():
            # ResNet18(BasicBlock)과 ResNet152(Bottleneck) 모두 대응
            if isinstance(m, (_tv_resnet.BasicBlock, _tv_resnet.Bottleneck)):
                blocks[n] = m
                
    elif "vgg" in name:
        import torch.nn as nn
        for n, m in model.named_modules():
            # VGG는 features 내의 Conv2d가 프루닝 대상
            if isinstance(m, nn.Conv2d) and n.startswith("features."):
                blocks[n] = m
                
    elif "efficientnet" in name:
        import torch.nn as nn
        for n, m in model.named_modules():
            # EfficientNet은 MBConv 내부의 pointwise conv 등을 탐색
            # 주신 로직에 따라 "conv"가 이름에 포함된 레이어를 타겟팅
            if isinstance(m, nn.Conv2d) and (".conv" in n or ".pw" in n or ".dw" in n):
                blocks[n] = m
                
    return blocks