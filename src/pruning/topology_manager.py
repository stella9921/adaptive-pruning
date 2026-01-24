import torch
import torch.nn as nn
import torch.fx as fx

def get_model_topology(model):
    print("[Stage 1] Analyzing Model Topology using PyTorch FX...")
    try:
        # 1. 모델을 심볼릭 트레이싱하여 연산 그래프 생성
        traced = fx.symbolic_trace(model)
        graph = traced.graph
        
        groups = []
        
        # 2. 그래프의 모든 노드를 순회하며 '결합 지점(Add)' 탐색
        for node in graph.nodes:
            is_add = False
            # 다양한 형태의 Add 연산 포착 (torch.add, x.add, a + b)
            if node.op == 'call_function' and (node.target == torch.add or "add" in str(node.target)):
                is_add = True
            elif node.op == 'call_method' and node.target == 'add':
                is_add = True

            if is_add:
                # [핵심] Add 연산에 들어가는 입력 노드들로부터 실제 레이어 이름을 추적
                group = set()
                for arg in node.args:
                    if isinstance(arg, fx.Node):
                        # 실제 레이어(Conv2d 등)가 나올 때까지 역추적하는 헬퍼 함수 호출
                        layer_name = _find_source_layer(arg)
                        if layer_name:
                            group.add(layer_name)
                
                # 2개 이상의 레이어가 연계되어 있다면 하나의 그룹으로 등록
                if len(group) >= 2:
                    groups.append(list(group))
        
        print(f"[*] Found {len(groups)} residual connection groups via FX.")
        #의존성 전파를 통한 그룹화 완료
        return groups
        
    except Exception as e:
        print(f"[*] Topology Analysis Error: {e}")
        return []

def _find_source_layer(node):
    """
    FX 노드로부터 실제 레이어(nn.Module) 이름을 역추적하는 함수
    BatchNorm이나 ReLU 등을 건너뛰고 실제 채널 수를 결정하는 Conv 레이어를 찾음
    """
    curr = node
    # 최대 10단계까지 역추적 (중간에 정규화나 활성화 함수가 껴있을 수 있음)
    for _ in range(10):
        if curr.op == 'call_module':
            return str(curr.target) 
        
        # 입력이 하나뿐인 노드라면 계속 거슬러 올라감
        if len(curr.args) > 0 and isinstance(curr.args[0], fx.Node):
            curr = curr.args[0]
        else:
            break
    return None