import torch.fx as fx  # Stage 1용
from torch.profiler import profile, record_function, ProfilerActivity # Stage 4용
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils.config_loader import load_config
from src.data.dataloader import get_dataloaders
from src.models import get_model, get_prune_fn
from src.pruning.sensitivity import maybe_load_or_compute_sensitivity
from src.pruning.pat_strategies import PATPruner
from src.pruning.pdt_strategies import PDTPruner

def analyze_topology_and_profiling(model, device, tag="Before Pruning"):
    print(f"\n=== [Stage 1 & 4] {tag} Analysis ===")
    inputs = torch.randn(1, 3, 32, 32).to(device) # CIFAR-100 기준 데이터셋 크기

    # [Stage 1] Topology Parsing (FX)
    try:
        traced = fx.symbolic_trace(model)
        print(f"[*] Topology Parsed: {len(list(traced.graph.nodes))} nodes found.")
    except Exception as e:
        print(f"[*] FX Parsing Warning: {e}")

    # [Stage 4] Resource Profiling
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                 with_flops=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            model(inputs)
    
    # 핵심 지표 출력
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    return prof


def main():
    # 1. 설정 및 데이터 준비
    config, args = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    print(f"Experimental Mode: {config['strategy']['method']}")
    print(f"Target Model: {config['model']['name']}")

    # 2. 모델 초기화
    model = get_model(config['model']).to(device)
    
    if config['strategy']['method'] == 'PAT':
        base_path = os.path.join(config['save_dir'], f"{config['model']['name']}_base.pth")
        if os.path.exists(base_path):
            model.load_state_dict(torch.load(base_path, map_location=device))
            print(f"Loaded base checkpoint from {base_path}")
        else:
            print("Warning: Base checkpoint not found. Proceeding with random weights.")

    # 3. 전략에 따른 실행 분기
    if config['strategy']['method'] == 'PAT':
        execute_pat_experiment(model, config, train_loader, val_loader, test_loader, device)
    elif config['strategy']['method'] == 'PDT':
        execute_pdt_experiment(model, config, train_loader, val_loader, test_loader, device)

def execute_pat_experiment(model, config, train_loader, val_loader, test_loader, device):
    si_data = maybe_load_or_compute_sensitivity(config, train_loader, test_loader, device)
    pat_engine = PATPruner(model, config, si_data)
    prune_fn = get_prune_fn(config['model']['name'])
    
    n_rounds = config['strategy'].get('n_rounds', 5)
    
    # [최초 측정] 프루닝 전 베이스라인
    analyze_topology_and_profiling(model, device, tag="PAT Initial Baseline")
    
    for r in range(1, n_rounds + 1):
        print(f"\n--- PAT Round {r}/{n_rounds} ---")
        
        # 1. 프루닝 인덱스 계산 및 실제 모델 구조 변경 (Channel Pruning)
        target_keep_indices = pat_engine.compute_all_keep_indices(round_idx=r) 
        model = prune_fn(model, target_keep_indices, device)
        
        # [Stage 1 & 4 측정] 프루닝 직후 리소스 변화 확인
        # 모델이 물리적으로 줄어들었으므로, 여기서 FX Node 수와 CUDA 메모리가 확 줄어든 게 찍혀야 합니다.
        analyze_topology_and_profiling(model, device, tag=f"PAT Round {r} - After Pruning")
        
        # 2. 파라미터가 변했으므로 옵티마이저 재설정
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        # 3. Finetuning
        finetune_eps = config['strategy'].get('finetune_epochs', 3)
        for epoch in range(1, finetune_eps + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_acc = evaluate(model, val_loader, device)
            print(f"FT Epoch {epoch} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")

def execute_pdt_experiment(model, config, train_loader, val_loader, test_loader, device):
    # PDT 엔진 초기화
    pdt_engine = PDTPruner(model, config)
    
    optimizer = optim.SGD(model.parameters(), 
                          lr=config['model']['base_lr'], 
                          momentum=0.9, 
                          weight_decay=config['model']['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    total_epochs = config['model']['epochs']
    prune_every = config['strategy'].get('prune_every', 10)
    start_epoch = config['strategy'].get('start_epoch', 1)

    # [최초 측정] 프루닝 전 베이스라인 상태 확인
    analyze_topology_and_profiling(model, device, tag="PDT Initial Baseline")
    
    for epoch in range(1, total_epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            output = model(x)
            loss = criterion(output, y)
            
            # Hessian 업데이트 및 프루닝 결정 시점
            is_hessian_step = (epoch >= start_epoch and epoch % prune_every == 0 and batch_idx == 0)
            
            if is_hessian_step:
                loss.backward(retain_graph=True) 
                
                print(f"\n>>> [SNOWS Update] Epoch {epoch}: Computing Hessian-Vector Products...")
                pdt_engine.step_pruning(loss=loss)
                
                # 마스크 적용 직전/직후 리소스 측정 (Stage 4)
                # PDT는 가중치를 0으로 만드는 것이므로, 연산 속도나 메모리 변화를 여기서 관찰합니다.
                pdt_engine.apply_mask_to_weights(optimizer=optimizer)
                
                # [측정] 마스크 업데이트 후 분석
                analyze_topology_and_profiling(model, device, tag=f"PDT Epoch {epoch} - After Mask Update")
                
                print(f">>> Current Sparsity: {pdt_engine.get_current_sparsity():.2f}%")
            else:
                loss.backward()
            
            pdt_engine.update_ema_and_mask_grad()
            optimizer.step()
            pdt_engine.apply_mask_to_weights()
            total_loss += loss.item()

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{total_epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")
# -------------------------------------------------------------------------
# 공통 유틸리티 함수 (기존 유지)
# -------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

if __name__ == "__main__":
    main()