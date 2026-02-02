import torch
import torch.nn as nn
import torch.fx as fx

def get_model_topology(model):
    """
    모델의 구조를 분석하여 의존성이 있는 레이어들을 그룹화합니다.
    ResNet의 Bottleneck 및 Shortcut 구조를 정확히 파악하여 채널 정보를 추출합니다.
    """
    model_name = model.__class__.__name__.lower()
    groups = []
    
    # 1. VGG 계열 (Sequential 구조) 처리
    if "vgg" in model_name:
        print(f"\n[Stage 1] Analyzing VGG-style Channel Topology for {model_name}...")
        groups = _get_vgg_topology(model)
    
    # 2. ResNet/EfficientNet 등 (Residual/Add 구조) 처리
    else:
        print(f"\n[Stage 1] Analyzing Residual Topology for {model_name} via PyTorch FX...")
        groups = _get_residual_topology(model)

    # --- [공통 로그 출력 및 필터링 로직] ---
    final_groups = []
    if groups:
        print(f"\n{'='*20} Final Topology Groups {'='*20}")
        all_modules = dict(model.named_modules())

        for i, group in enumerate(groups):
            # 그룹 내에서 실제 파라미터가 있는 레이어(Conv, Linear, BN)만 필터링
            valid_sub_group = []
            channels = "N/A"
            
            for layer_name in group:
                module = all_modules.get(layer_name)
                if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
                    valid_sub_group.append(layer_name)
                    
                    # 채널 수 추출 (가장 먼저 발견되는 유효한 채널 값 사용)
                    if channels == "N/A":
                        if hasattr(module, 'out_channels'):
                            channels = module.out_channels
                        elif hasattr(module, 'out_features'):
                            channels = module.out_features
                        elif hasattr(module, 'num_features'):
                            channels = module.num_features

            if valid_sub_group:
                final_groups.append(valid_sub_group)
                print(f" Group {len(final_groups):2d} | Channels: {str(channels):>4} | Layers: {valid_sub_group}")
        
        print(f"{'='*63}")
        print(f"[*] Total {len(final_groups)} groups identified.\n")
    else:
        print("\n[!] No dependency groups found. Layers will be treated independently.")

    return final_groups

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
        # leaf_module로 지정하여 BatchNorm 내부까지 쪼개지지 않도록 방지
        traced = fx.symbolic_trace(model)
        graph = traced.graph
        groups = []
        
        for node in graph.nodes:
            # torch.add(a, b) 또는 a.add(b) 형태 탐색
            is_add = (node.op == 'call_function' and (node.target == torch.add or "add" in str(node.target))) or \
                     (node.op == 'call_method' and node.target == 'add')
            
            if is_add:
                group = set()
                for arg in node.args:
                    if isinstance(arg, fx.Node):
                        # Add 연산에 들어오는 입력 노드들로부터 실제 레이어 이름을 역추적
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
    """
    FX 노드에서 거꾸로 올라가며 실제 파라미터가 있는 nn.Module(Conv, BN 등)을 찾습니다.
    """
    if depth > 10: # 무한 루프 방지
        return []

    # 1. 현재 노드가 실제 모듈 호출인 경우
    if node.op == 'call_module':
        return [str(node.target)]

    # 2. ReLU나 다른 함수 노드인 경우 더 위로 추적
    layers = []
    for arg in node.args:
        if isinstance(arg, fx.Node):
            layers.extend(_trace_back_to_layers(arg, depth + 1))
    
    return layers