def get_model_topology(model):
    print("[Stage 1] Analyzing Model Topology using PyTorch FX...")
    try:
        traced = fx.symbolic_trace(model)
        graph = traced.graph
        
        groups = []
        for node in graph.nodes:
            # ResNet의 Skip-connection은 보통 'add' 함수나 'torch.add'로 나타납니다.
            if (node.op == 'call_function' and node.target in [torch.add, nn.functional.add]) or \
               (node.op == 'call_method' and node.target == 'add'):
                
                group = []
                for arg in node.args:
                    if isinstance(arg, fx.Node):
                        # 실제 Conv 레이어의 이름을 역추적해서 가져옵니다.
                        group.append(arg.name)
                if group:
                    groups.append(group)
        
        print(f"[*] Found {len(groups)} residual connection groups.")
        return groups
    except Exception as e:
        print(f"[*] Topology Analysis Error: {e}")
        return []