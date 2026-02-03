import torch
import torch.nn as nn
import torch.fx as fx

def get_model_topology(model):
    """
    모델의 구조를 분석하여 의존성이 있는 레이어들을 그룹화합니다.
    EfficientNet의 경우 MBConv 블록 단위의 16개 그룹 전략을 우선 적용합니다.
    """
    model_name = model.__class__.__name__.lower()
    groups = []
    
    # 1. EfficientNet 계열 (16개 MBConv 블록 수동 매핑)
    if "efficientnet" in model_name:
        print(f"\n[Stage 1] Applying MBConv Block-wise Topology for {model_name}...")
        groups = _get_efficientnet_topology(model)

    # 2. VGG 계열 (Sequential 구조)
    elif "vgg" in model_name:
        print(f"\n[Stage 1] Analyzing VGG-style Channel Topology for {model_name}...")
        groups = _get_vgg_topology(model)
    
    # 3. 기타 (ResNet 등 Residual/Add 구조) - FX 활용
    else:
        print(f"\n[Stage 1] Analyzing Residual Topology for {model_name} via PyTorch FX...")
        groups = _get_residual_topology(model)

    # --- [공통 로그 출력 및 필터링 로직] ---
    final_groups = []
    all_modules = dict(model.named_modules())

    if groups:
        print(f"\n{'='*20} Final Topology Groups {'='*20}")
        for group_seeds in groups:
            valid_sub_group = []
            channels = "N/A"
            
            # 각 그룹 시드(예: 'features.1.0')를 바탕으로 하위 레이어를 모두 수집
            for seed in group_seeds:
                for name, module in model.named_modules():
                    # 시드 이름으로 시작하는 모든 Conv, Linear, BN을 하나의 그룹으로 묶음
                    if name.startswith(seed):
                        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                            valid_sub_group.append(name)
                            
                            # 채널 수 추출 (가장 먼저 발견되는 값 사용)
                            if channels == "N/A":
                                if hasattr(module, 'out_channels'):
                                    channels = module.out_channels
                                elif hasattr(module, 'out_features'):
                                    channels = module.out_features
                                elif hasattr(module, 'num_features'):
                                    channels = module.num_features

            if valid_sub_group:
                # 중복 제거 및 정렬
                valid_sub_group = sorted(list(set(valid_sub_group)))
                final_groups.append(valid_sub_group)
                
                # 출력 가독성을 위해 레이어 리스트 요약
                display_layers = f"{valid_sub_group[0]} ... ({len(valid_sub_group)} layers)"
                print(f" Group {len(final_groups):2d} | Channels: {str(channels):>4} | Layers: {display_layers}")
        
        print(f"{'='*63}")
        print(f"[*] Total {len(final_groups)} groups identified.\n")
    else:
        print("\n[!] No dependency groups found. Layers will be treated independently.")

    return final_groups

def _get_efficientnet_topology(model):
    """
    EfficientNet-B0의 16개 MBConv 블록 구조를 수동으로 정의합니다.
    """
    # MBConv 블록의 베이스 경로만 지정하면 공통 로직에서 하위 레이어(0, 1, 2, 3)를 다 긁어옵니다.
    eb0_blocks = [
        ['features.1.0'], # Group 1
        ['features.2.0'], ['features.2.1'], # Group 2, 3
        ['features.3.0'], ['features.3.1'], # Group 4, 5
        ['features.4.0'], ['features.4.1'], ['features.4.2'], # Group 6, 7, 8
        ['features.5.0'], ['features.5.1'], ['features.5.2'], # Group 9, 10, 11
        ['features.6.0'], ['features.6.1'], ['features.6.2'], ['features.6.3'], # 12, 13, 14, 15
        ['features.7.0'], # Group 16
    ]
    return eb0_blocks

def _get_vgg_topology(model):
    groups = []
    current_group = []
    last_out_channels = -1
    
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            curr_out_channels = m.out_channels
            if curr_out_channels == last_out_channels:
                current_group.append(name)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [name]
                last_out_channels = curr_out_channels
    if current_group:
        groups.append(current_group)
    return groups

def _get_residual_topology(model):
    try:
        traced = fx.symbolic_trace(model)
        graph = traced.graph
        groups = []
        
        for node in graph.nodes:
            is_add = (node.op == 'call_function' and (node.target == torch.add or "add" in str(node.target))) or \
                     (node.op == 'call_method' and node.target == 'add')
            
            if is_add:
                group = set()
                for arg in node.args:
                    if isinstance(arg, fx.Node):
                        found_layers = _trace_back_to_layers(arg)
                        for layer_name in found_layers:
                            group.add(layer_name)
                
                if len(group) >= 2:
                    groups.append(sorted(list(group)))
        return groups
    except Exception as e:
        print(f"[*] Residual Analysis Error: {e}")
        return []

def _trace_back_to_layers(node, depth=0):
    if depth > 10:
        return []
    if node.op == 'call_module':
        return [str(node.target)]
    layers = []
    for arg in node.args:
        if isinstance(arg, fx.Node):
            layers.extend(_trace_back_to_layers(arg, depth + 1))
    return layers