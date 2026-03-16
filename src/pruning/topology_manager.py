import torch
import torch.nn as nn
import torch.fx as fx
import operator


def get_model_topology(model):
    """
    모델의 구조를 분석하여 의존성이 있는 레이어들을 그룹화합니다.
    - EfficientNet: MBConv 블록 단위 전략
    - VGG: Conv 채널 일치 그룹 + Classifier Linear 레이어 독립 그룹
    - MobileNet: InvertedResidual 블록 단위
    - ResNet 등: FX Graph 분석 기반 Residual Topology
    """
    model_name = model.__class__.__name__.lower()
    groups = []

    # 1. EfficientNet
    if "efficientnet" in model_name:
        print(f"\n[Stage 1] Applying MBConv Block-wise Topology for {model_name}...")
        groups = _get_efficientnet_topology(model)

    # 2. VGG
    elif "vgg" in model_name:
        print(f"\n[Stage 1] Analyzing VGG-style Topology (Conv + Classifier) for {model_name}...")
        groups = _get_vgg_topology(model)

    # 3. MobileNet
    elif "mobilenet" in model_name:
        print(f"\n[Stage 1] Analyzing MobileNet InvertedResidual Topology for {model_name}...")
        groups = _get_mobilenet_topology(model)
    elif "vit" in model_name or "transformer" in model_name:
        print(f"\n[Stage 1] Analyzing ViT Block-wise Topology for {model_name}...")
        groups = _get_vit_topology(model)

    # 4. Residual (ResNet 등)
    else:
        print(f"\n[Stage 1] Analyzing Residual Topology for {model_name} via PyTorch FX...")
        groups = _get_residual_topology(model)

    # ----------------------------
    # 공통 그룹 후처리 및 출력
    # ----------------------------
    final_groups = []

    if groups:
        print(f"\n{'='*20} Final Topology Groups {'='*20}")

        for group_seeds in groups:
            valid_sub_group = []
            channels = "N/A"

            for seed in group_seeds:
                for name, module in model.named_modules():
                    if name == seed or name.startswith(seed + "."):
                        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                            valid_sub_group.append(name)

                            if channels == "N/A":
                                if hasattr(module, 'out_channels'):
                                    channels = module.out_channels
                                elif hasattr(module, 'out_features'):
                                    channels = module.out_features
                                elif hasattr(module, 'num_features'):
                                    channels = module.num_features

            if valid_sub_group:
                valid_sub_group = sorted(list(set(valid_sub_group)))
                final_groups.append(valid_sub_group)

                display_layers = f"{valid_sub_group[0]} ... ({len(valid_sub_group)} layers)"
                print(f" Group {len(final_groups):2d} | Dimension: {str(channels):>4} | Layers: {display_layers}")

        print(f"{'='*63}")
        print(f"[*] Total {len(final_groups)} groups identified.\n")

    else:
        print("\n[!] No dependency groups found. Layers will be treated independently.")

    return final_groups


# ----------------------------
# EfficientNet
# ----------------------------
def _get_efficientnet_topology(model):
    eb0_blocks = [
        ['features.1.0'], ['features.2.0'], ['features.2.1'],
        ['features.3.0'], ['features.3.1'], ['features.4.0'],
        ['features.4.1'], ['features.4.2'], ['features.5.0'],
        ['features.5.1'], ['features.5.2'], ['features.6.0'],
        ['features.6.1'], ['features.6.2'], ['features.6.3'],
        ['features.7.0'],
    ]
    return eb0_blocks


# ----------------------------
# VGG
# ----------------------------
def _get_vgg_topology(model):
    groups = []
    current_group = []
    last_out_channels = -1
    max_layers_per_group = 3

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            curr_out_channels = m.out_channels

            if curr_out_channels == last_out_channels and len(current_group) < max_layers_per_group:
                current_group.append(name)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [name]
                last_out_channels = curr_out_channels

    if current_group:
        groups.append(current_group)

    print("[Stage 1] Adding Classifier Linear layers to VGG groups...")
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            is_last_layer = "6" in name or "fc_out" in name or "classifier.6" in name
            if is_last_layer:
                continue
            groups.append([name])

    return groups


# ----------------------------
# MobileNet
# ----------------------------
def _get_mobilenet_topology(model):
    groups = []

    for name, module in model.named_modules():
        if module.__class__.__name__ == "InvertedResidual":

            block_group = []

            for sub_name, sub_module in module.named_modules():
                full_name = f"{name}.{sub_name}" if sub_name != "" else name

                if isinstance(sub_module, nn.Conv2d):
                    block_group.append(full_name)

            if block_group:
                groups.append(block_group)

    print(f"[*] MobileNet: {len(groups)} inverted residual groups found.")
    return groups


# ----------------------------
# Residual (ResNet 등)
# ----------------------------
def _get_residual_topology(model):
    try:
        traced = fx.symbolic_trace(model)
        graph = traced.graph
        groups = []

        for node in graph.nodes:

            is_add = (
                (node.op == 'call_function' and node.target in [torch.add, operator.add]) or
                (node.op == 'call_method' and node.target == 'add')
            )

            if is_add:
                group = set()

                for arg in node.args:
                    if isinstance(arg, fx.Node):
                        found = _find_source_layers(arg, traced)
                        for ln in found:
                            group.add(ln)

                if len(group) >= 2:
                    groups.append(sorted(list(group)))

        return groups

    except Exception as e:
        print(f"[*] Residual Analysis Error: {e}")
        return []


def _find_source_layers(node, traced_model, visited=None):
    if visited is None:
        visited = set()
    if node in visited:
        return []
    visited.add(node)

    if node.op == 'call_module':
        module = dict(traced_model.named_modules())[node.target]

        if isinstance(module, (nn.Conv2d, nn.Linear)):
            return [str(node.target)]

    layers = []
    for arg in node.args:
        if isinstance(arg, fx.Node):
            layers.extend(_find_source_layers(arg, traced_model, visited))

    return layers



# ----------------------------
# Vision Transformer (ViT)
# ----------------------------
def _get_vit_topology(model):
    """
    ViT의 의존성을 고려한 세부 그룹화:
    1. QKV Group: 입력 차원을 공유하는 Q, K, V 레이어
    2. Output Group: Residual Connection에 직접 닿는 레이어들 (Proj, FC2)
    3. Intermediate Group: 독립적으로 확장이 가능한 레이어 (FC1)
    """
    groups = []
    
    for name, module in model.named_modules():
        module_type = module.__class__.__name__
        
        # Transformer Block 단위로 진입
        if any(keyword in module_type for keyword in ["Block", "LayerScale", "EncoderBlock"]):
            
            # 1. QKV Group (Attention의 입력부)
            qkv_group = []
            # 2. Output Group (Residual Connection과 맞닿는 부분 - $D$ 차원 유지)
            output_group = []
            # 3. Intermediate Group (MLP 확장부 - 프루닝 자유도 높음)
            inter_group = []

            for sub_name, sub_module in module.named_modules():
                full_name = f"{name}.{sub_name}" if sub_name != "" else name
                if not isinstance(sub_module, nn.Linear) or "head" in full_name:
                    continue

                # 레이어 이름 키워드에 따른 그룹 분기
                if any(k in full_name for k in ["qkv", "query", "key", "value"]):
                    qkv_group.append(full_name)
                elif any(k in full_name for k in ["proj", "fc2", "mlp.dwconv"]): # 출력부
                    output_group.append(full_name)
                elif any(k in full_name for k in ["fc1", "mlp.fc1"]): # 중간 확장부
                    inter_group.append(full_name)

            # 유효한 그룹들만 추가
            if qkv_group: groups.append(qkv_group)
            if inter_group: groups.append(inter_group)
            if output_group: groups.append(output_group)

    print(f"[*] ViT Topology: {len(groups)} dependency-aware groups identified.")
    return groups