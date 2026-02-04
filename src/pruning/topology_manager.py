import torch
import torch.nn as nn
import torch.fx as fx

def get_model_topology(model):
    """
    모델의 구조를 분석하여 의존성이 있는 레이어들을 그룹화합니다.
    - EfficientNet: MBConv 블록 단위 전략
    - VGG: Conv 채널 일치 그룹 + Classifier Linear 레이어 독립 그룹
    - ResNet 등: FX Graph 분석 기반 Residual Topology
    """
    model_name = model.__class__.__name__.lower()
    groups = []
    
    # 1. EfficientNet 계열
    if "efficientnet" in model_name:
        print(f"\n[Stage 1] Applying MBConv Block-wise Topology for {model_name}...")
        groups = _get_efficientnet_topology(model)

    # 2. VGG 계열 (Sequential + Classifier 확장)
    elif "vgg" in model_name:
        print(f"\n[Stage 1] Analyzing VGG-style Topology (Conv + Classifier) for {model_name}...")
        groups = _get_vgg_topology(model)
    
    # 3. 기타 (ResNet 등 Residual/Add 구조) - FX 활용
    else:
        print(f"\n[Stage 1] Analyzing Residual Topology for {model_name} via PyTorch FX...")
        groups = _get_residual_topology(model)

    # --- [공통 로그 출력 및 필터링 로직] ---
    final_groups = []

    if groups:
        print(f"\n{'='*20} Final Topology Groups {'='*20}")
        for group_seeds in groups:
            valid_sub_group = []
            channels = "N/A"
            
            # 각 그룹 시드(레이어 이름)를 바탕으로 Conv, Linear, BN을 하나의 그룹으로 수집
            for seed in group_seeds:
                for name, module in model.named_modules():
                    if name == seed or name.startswith(seed + "."):
                        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                            valid_sub_group.append(name)
                            
                            # 채널(혹은 피처) 수 추출
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
                
                # 가독성을 위한 출력
                display_layers = f"{valid_sub_group[0]} ... ({len(valid_sub_group)} layers)"
                print(f" Group {len(final_groups):2d} | Channels: {str(channels):>4} | Layers: {display_layers}")
        
        print(f"{'='*63}")
        print(f"[*] Total {len(final_groups)} groups identified.\n")
    else:
        print("\n[!] No dependency groups found. Layers will be treated independently.")

    return final_groups

def _get_efficientnet_topology(model):
    """EfficientNet-B0의 MBConv 블록 구조 수동 정의"""
    eb0_blocks = [
        ['features.1.0'], ['features.2.0'], ['features.2.1'], 
        ['features.3.0'], ['features.3.1'], ['features.4.0'], 
        ['features.4.1'], ['features.4.2'], ['features.5.0'], 
        ['features.5.1'], ['features.5.2'], ['features.6.0'], 
        ['features.6.1'], ['features.6.2'], ['features.6.3'], 
        ['features.7.0'],
    ]
    return eb0_blocks

def _get_vgg_topology(model):
    """
    VGG의 Conv 레이어 그룹화 (최대 3개씩 세분화) 및 Classifier(Linear) 레이어 추가
    """
    groups = []
    current_group = []
    last_out_channels = -1
    max_layers_per_group = 3  # [수정] 한 그룹에 너무 많은 레이어가 묶이지 않도록 제한 (불균형 해소)
    
    # --- Part A: Convolutional Layers ---
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            curr_out_channels = m.out_channels
            
            # 채널이 같고, 현재 그룹의 레이어 수가 3개 미만인 경우에만 같은 그룹으로 유지
            if curr_out_channels == last_out_channels and len(current_group) < max_layers_per_group:
                current_group.append(name)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [name]
                last_out_channels = curr_out_channels
                
    if current_group:
        groups.append(current_group)

    # --- Part B: Classifier(Linear) Layers ---
    print("[Stage 1] Adding Classifier Linear layers to VGG groups...")
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            is_last_layer = "6" in name or "fc_out" in name or "classifier.6" in name
            if is_last_layer:
                print(f" [System] Identified final output layer (Skipping): {name}")
                continue
            
            groups.append([name])
            print(f" [System] Linear layer group added: {name}")

    return groups

def _get_residual_topology(model):
    """FX를 활용한 Residual 연결 구조 분석"""
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