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

# -------------------------------------------------------------------------
# [Method A] PAT: 민감도 분석 기반 반복적 프루닝 
# -------------------------------------------------------------------------
def execute_pat_experiment(model, config, train_loader, val_loader, test_loader, device):
    si_data = maybe_load_or_compute_sensitivity(config, train_loader, test_loader, device)
    pat_engine = PATPruner(model, config, si_data)
    prune_fn = get_prune_fn(config['model']['name'])
    
    n_rounds = config['strategy'].get('n_rounds', 5)
    
    for r in range(1, n_rounds + 1):
        print(f"\n--- PAT Round {r}/{n_rounds} ---")
        target_keep_indices = pat_engine.compute_all_keep_indices(round_idx=r) 
        model = prune_fn(model, target_keep_indices, device)
        
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        finetune_eps = config['strategy'].get('finetune_epochs', 3)
        for epoch in range(1, finetune_eps + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_acc = evaluate(model, val_loader, device)
            print(f"FT Epoch {epoch} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")

# -------------------------------------------------------------------------
# [Method B] PDT: SNOWS Hessian 엔진 기반 동적 마스킹 
# -------------------------------------------------------------------------
def execute_pdt_experiment(model, config, train_loader, val_loader, test_loader, device):
    # PDT 엔진 초기화 (Hessian-free SNOWS 엔진 포함)
    pdt_engine = PDTPruner(model, config)
    
    optimizer = optim.SGD(model.parameters(), 
                          lr=config['model']['base_lr'], 
                          momentum=0.9, 
                          weight_decay=config['model']['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    total_epochs = config['model']['epochs']
    prune_every = config['strategy'].get('prune_every', 10)
    start_epoch = config['strategy'].get('start_epoch', 1)
    
    for epoch in range(1, total_epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            output = model(x)
            loss = criterion(output, y)
            
            # [수정 포인트] Hessian을 업데이트해야 하는 배치인지 먼저 확인
            is_hessian_step = (epoch >= start_epoch and epoch % prune_every == 0 and batch_idx == 0)
            
            if is_hessian_step:
                # 1. Hessian 연산을 위해 retain_graph=True로 설정하여 그래프 유지
                loss.backward(retain_graph=True) 
                
                print(f"\n>>> [SNOWS Update] Epoch {epoch}: Computing Hessian-Vector Products...")
                # 2. 이제 그래프가 살아있으므로 Hessian 연산 가능
                pdt_engine.step_pruning(loss=loss)
                
                # 3. 업데이트된 점수로 마스크 적용 및 모멘텀 초기화
                pdt_engine.apply_mask_to_weights(optimizer=optimizer)
                print(f">>> Current Sparsity: {pdt_engine.get_current_sparsity():.2f}%")
            else:
                # 일반적인 상황에서는 그래프를 유지할 필요가 없음 (메모리 절약)
                loss.backward()
            
            # [Step 1] 매 배치마다 Gradient EMA 업데이트
            pdt_engine.update_ema_and_mask_grad()
            
            optimizer.step()
            
            # [Step 3] 가중치에 마스크 강제 적용 (프루닝 상태 유지)
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