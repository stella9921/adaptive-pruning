import torch
import torch.fx as fx

def get_model_topology(model):
    print("[Stage 1] Analyzing Model Topology using PyTorch FX...")
    traced = fx.symbolic_trace(model)
    
    # 1. 노드 간의 의존성 분석 (간단한 예시: ResNet Skip-connection 대응)
    # 실제로는 Conv-BN-ReLU 시퀀스나 Add 노드를 추적
    groups = []
    for node in traced.graph.nodes:
        if node.op == 'call_function' and node.target == torch.add:
            groups.append([n.name for n in node.args if isinstance(n, fx.Node)])
    
    print(f"[*] Found {len(groups)} residual connection groups.")
    return groups