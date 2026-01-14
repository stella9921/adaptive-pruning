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
    # load_config()는 argparse를 통해 --model, --strategy 인자를 받아 YAML을 통합함
    config, args = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # dataloader.py에서 정의된 로직으로 데이터 로드
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    print(f"Experimental Mode: {config['strategy']['method']}")
    print(f"Target Model: {config['model']['name']}")

    # 2. 모델 초기화
    model = get_model(config['model']).to(device)
    
    # PAT(Post-training) 방식은 사전 학습된 베이스 모델이 필요함
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
# [Method A] PAT: 민감도 분석 기반 반복적 프루닝 및 파인튜닝
# -------------------------------------------------------------------------
def execute_pat_experiment(model, config, train_loader, val_loader, test_loader, device):
    # 단계 1: 민감도 데이터 확보 (기존 JSON 로드 또는 신규 계산)
    si_data = maybe_load_or_compute_sensitivity(config, train_loader, test_loader, device)
    
    # 단계 2: PAT 엔진 초기화 (ASE-OneShot 및 Amplification 수식 적용용)
    pat_engine = PATPruner(model, config, si_data)
    prune_fn = get_prune_fn(config['model']['name'])
    
    n_rounds = config['strategy'].get('n_rounds', 5)
    
    for r in range(1, n_rounds + 1):
        print(f"\n--- PAT Round {r}/{n_rounds} ---")
        
        # 블록별 최적 유지 채널 계산
        target_keep_indices = pat_engine.compute_all_keep_indices(round_idx=r) 
        
        # 모델 구조를 물리적으로 변경 (채널 삭제)
        model = prune_fn(model, target_keep_indices, device)
        
        # 파인튜닝을 통한 성능 회복
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        finetune_eps = config['strategy'].get('finetune_epochs', 3)
        for epoch in range(1, finetune_eps + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_acc = evaluate(model, val_loader, device)
            print(f"FT Epoch {epoch} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")

# -------------------------------------------------------------------------
# [Method B] PDT: 학습 중 EMA 기반 동적 마스킹 및 실시간 프루닝
# -------------------------------------------------------------------------
def execute_pdt_experiment(model, config, train_loader, val_loader, test_loader, device):
    # PDT 엔진 초기화 (EMA 히스토리 관리 및 마스크 생성)
    pdt_engine = PDTPruner(model, config)
    
    optimizer = optim.SGD(model.parameters(), 
                          lr=config['model']['base_lr'], 
                          momentum=0.9, 
                          weight_decay=config['model']['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    total_epochs = config['model']['epochs']
    
    for epoch in range(1, total_epochs + 1):
        model.train()
        total_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            
            # 논문 4.2절: 그래디언트 EMA 업데이트 및 마스킹 적용
            pdt_engine.update_ema_and_mask_grad()
            optimizer.step()
            
            # 가중치에 마스크 강제 적용하여 희소성 유지
            pdt_engine.apply_mask_to_weights()
            total_loss += loss.item()

        # 논문 4.3절: 정해진 주기마다 프루닝 임계값 업데이트
        if epoch >= config['strategy']['start_epoch'] and epoch % config['strategy']['prune_every'] == 0:
            pdt_engine.step_pruning()
            print(f"\n>>> [PDT Update] Current Sparsity: {pdt_engine.get_current_sparsity():.2f}%")

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{total_epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

# -------------------------------------------------------------------------
# 공통 유틸리티 함수
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