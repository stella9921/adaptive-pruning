import torch
import torch.nn as nn
import torch.fx as fx

def get_model_topology(model):
    print("[Stage 1] Analyzing Model Topology using PyTorch FX...")
    try:
        traced = fx.symbolic_trace(model)
        graph = traced.graph
        
        groups = []
        for node in graph.nodes:
            # Skip-connection (Add) 노드 찾는 로직 수정
            is_add = False
            
            # 1. torch.add(a, b) 형태 체크
            if node.op == 'call_function' and node.target == torch.add:
                is_add = True
            # 2. a.add(b) 메서드 형태 체크
            elif node.op == 'call_method' and node.target == 'add':
                is_add = True
            # 3. a + b (operator.add) 형태 체크
            elif node.op == 'call_function' and "add" in str(node.target):
                is_add = True

            if is_add:
                group = []
                for arg in node.args:
                    if isinstance(arg, fx.Node):
                        # 이전 노드(보통 Conv)의 이름을 가져옴
                        group.append(arg.name)
                if len(group) >= 2:
                    groups.append(group)
        
        print(f"[*] Found {len(groups)} residual connection groups via FX.")
        return groups
    except Exception as e:
        print(f"[*] Topology Analysis Error: {e}")
        return []