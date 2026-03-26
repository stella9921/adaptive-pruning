import torch
import torch.nn as nn
import torch.fx as fx
import operator


def get_model_topology(model):
    model_name = model.__class__.__name__.lower()

    if "efficientnet" in model_name:
        groups = _get_efficientnet_topology(model)
    elif "vgg" in model_name:
        groups = _get_vgg_topology(model)
    elif "mobilenet" in model_name:
        groups = _get_mobilenet_topology(model)
    elif "vit" in model_name or "transformer" in model_name or "deit" in model_name:
        # ViT는 dict 형태 그룹을 바로 리턴 (후처리 불필요)
        return _get_vit_topology(model)
    else:
        groups = _get_residual_topology(model)

    # --- CNN용 후처리 (ViT는 여기 안 옴) ---
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
            if valid_sub_group:
                valid_sub_group = sorted(list(set(valid_sub_group)))
                final_groups.append(valid_sub_group)
        print(f"[*] Total {len(final_groups)} groups identified.\n")
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
    DeiT/ViT 전용 topology.
    
    그룹 타입 2가지:
    - type 'ffn':  [fc1, fc2] — 뉴런 단위, fc1 출력 = fc2 입력
    - type 'attn': [qkv, proj] — head 단위, head_dim 슬라이스 동기화
    
    각 그룹에 meta 정보를 함께 저장해서
    ViTPDTPruner가 마스킹 방식을 구분할 수 있게 함.
    """
    groups = []

    all_modules = dict(model.named_modules())
    all_linear_names = {
        name for name, m in all_modules.items()
        if isinstance(m, nn.Linear) and name != 'head'
    }

    block_indices = set()
    for name in all_linear_names:
        parts = name.split('.')
        if parts[0] == 'blocks' and parts[1].isdigit():
            block_indices.add(int(parts[1]))

    # head_dim 자동 추출
    first_block = f'blocks.0'
    qkv_layer = all_modules.get(f'{first_block}.attn.qkv')
    proj_layer = all_modules.get(f'{first_block}.attn.proj')
    num_heads = None
    head_dim = None
    if qkv_layer is not None and proj_layer is not None:
        embed_dim = proj_layer.out_features       # 384
        qkv_out   = qkv_layer.out_features        # 1152
        num_heads_x3 = qkv_out // embed_dim       # 3
        # num_heads는 모델에서 직접 꺼내는 게 가장 안전
        for name, m in model.named_modules():
            if 'blocks.0.attn' == name and hasattr(m, 'num_heads'):
                num_heads = m.num_heads
                head_dim  = embed_dim // num_heads
                break
        if num_heads is None:
            # fallback: timm default
            num_heads = 6
            head_dim  = embed_dim // num_heads

    for i in sorted(block_indices):
        p    = f'blocks.{i}'
        qkv  = f'{p}.attn.qkv'
        proj = f'{p}.attn.proj'
        fc1  = f'{p}.mlp.fc1'
        fc2  = f'{p}.mlp.fc2'

        # Attention head 그룹
        if qkv in all_linear_names and proj in all_linear_names:
            groups.append({
                'type'     : 'attn',
                'names'    : [qkv, proj],
                'num_heads': num_heads,
                'head_dim' : head_dim,
            })

        # FFN intermediate 그룹
        if fc1 in all_linear_names and fc2 in all_linear_names:
            groups.append({
                'type' : 'ffn',
                'names': [fc1, fc2],
            })

    print(f"[*] ViT Topology: {len(block_indices)} blocks "
          f"→ {len(groups)} groups "
          f"(num_heads={num_heads}, head_dim={head_dim})")
    return groups