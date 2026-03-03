import torch.nn as nn
from .resnet import get_resnet, prune_resnet_blockwise
from .vgg import get_vgg16, prune_vgg_blockwise
from .efficientnet import get_efficientnet, prune_efficientnet_blockwise
from .mobilenet import get_mobilenet, prune_mobilenet_blockwise


def get_model(model_cfg):
    """YAML의 model 섹션 설정을 받아 모델 객체를 생성함"""
    name = model_cfg['name'].lower()
    num_classes = model_cfg.get('num_classes', 100)
    
    if "resnet" in name:
        return get_resnet(name, num_classes)
    elif "vgg" in name:
        return get_vgg16(num_classes)
    elif "efficientnet" in name:
        return get_efficientnet(num_classes)
    elif "mobilenet" in name:
        return get_mobilenet(name, num_classes)
    else:
        raise ValueError(f"지원하지 않는 모델 이름입니다: {name}")

def get_prune_fn(model_name):
    """PAT 결과가 나온 후, 실제로 모델 채널을 자르는 함수를 반환함"""
    name = model_name.lower()
    if "resnet" in name:
        return prune_resnet_blockwise
    elif "vgg" in name:
        return prune_vgg_blockwise
    elif "efficientnet" in name:
        return prune_efficientnet_blockwise
    elif "mobilenet" in name:
        return prune_mobilenet_blockwise
    else:
        raise ValueError(f"물리적 프루닝 함수를 찾을 수 없습니다: {name}")

def find_prunable_blocks(model, model_name, topology_groups=None):
    """
    [Stage 1 반영] 모델별로 프루닝 타겟이 되는 블록들을 찾고, 
    FX Topology 분석 결과가 있다면 이를 기반으로 그룹핑함.
    """
    name = model_name.lower()
    all_modules = dict(model.named_modules())
    blocks = {}
    
    # CASE 1: FX Topology Manager가 분석한 그룹 정보가 있는 경우 (Stage 1 핵심)
    if topology_groups:
        print(f"[*] Linking FX Topology groups to model layers...")
        for idx, group in enumerate(topology_groups):
            # group: ['layer1.0.conv1', 'layer1.0.conv2']
            valid_layers = []
            for layer_name in group:
                if layer_name in all_modules:
                    m = all_modules[layer_name]
                    if isinstance(m, nn.Conv2d):
                        valid_layers.append(m)
            
            if valid_layers:
                # 그룹 내 레이어들을 리스트로 묶어서 반환
                blocks[f"group_{idx}"] = valid_layers
        return blocks

    # CASE 2: 그룹 정보가 없을 때 (기존 방식 - Fallback)
    if "resnet" in name:
        from torchvision.models import resnet as _tv_resnet
        for n, m in model.named_modules():
            if isinstance(m, (_tv_resnet.BasicBlock, _tv_resnet.Bottleneck)):
                blocks[n] = m
                
    elif "vgg" in name:
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d) and n.startswith("features."):
                blocks[n] = m
                
    elif "efficientnet" in name:
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d) and (".conv" in n or ".pw" in n or ".dw" in n):
                blocks[n] = m
    elif "mobilenet" in name:
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                blocks[n] = m

                
    return blocks