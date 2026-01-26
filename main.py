import torch.fx as fx  # Stage 1용
from torch.profiler import profile, record_function, ProfilerActivity # Stage 4용
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 기존 모듈 임포트
from src.pruning.topology_manager import get_model_topology
from src.pruning.optimizer import lagrangian_optimization
from src.utils.config_loader import load_config
from src.data.dataloader import get_dataloaders
from src.models import get_model, get_prune_fn
from src.pruning.sensitivity import maybe_load_or_compute_sensitivity
from src.pruning.pat_strategies import PATPruner
from src.pruning.pdt_strategies import PDTPruner

def analyze_topology_and_profiling(model, device, config, tag="Before Pruning"):
    """Stage 1(Topology)과 Stage 4(Resource)를 동시에 분석"""
    print(f"\n=== [Stage 1 & 4] {tag} Analysis ===")
    
    model_config = config.get('model', {})
    input_size = model_config.get('input_size', 224)
    # 분석용 입력은 1개로 제한하여 OOM 방지
    inputs = torch.randn(1, 3, input_size, input_size).to(device)

    # [Stage 1] Topology Status (FX)
    try:
        traced = fx.symbolic_trace(model)
        print(f"[*] Topology Status: {len(list(traced.graph.nodes))} nodes currently in graph.")
    except Exception as e:
        print(f"[*] FX Parsing Warning: {e}")

    # [Stage 4] Resource Profiling
    model.eval()
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                     with_flops=True, profile_memory=True) as prof:
            with record_function("model_inference"):
                model(inputs)
    
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    return prof


def main():
    # 1. 설정 및 데이터 준비
    config, args = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_cfg = config.get('dataset', {})
    dataset_name = dataset_cfg if isinstance(dataset_cfg, str) else dataset_cfg.get('name', 'cifar100')

    # 데이터 로더 로드
    loaders = get_dataloaders(config)
    
    if len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, val_loader = loaders
        test_loader = val_loader 
    
    print(f"Experimental Mode: {config['strategy']['method']}")
    print(f"Target Model: {config['model']['name']} | Dataset: {dataset_name}")

    # 2. 모델 초기화
    model = get_model(config['model']).to(device)
    
    # -------------------------------------------------------------------------
    # [CRITICAL FIX] FC 레이어 가중치 이식 (ImageNet-100 대응)
    # 0.48% 정확도 문제를 해결하기 위해 기존 1,000개 클래스 지능을 100개로 복사합니다.
    # -------------------------------------------------------------------------
    if dataset_name == 'imagenet' or 'imagenet' in dataset_name:
        print(">>> [System] Transferring Pretrained Weights to FC Layer (1000 -> 100)...")
        # 임시로 1000개 출력인 모델을 하나 더 생성하여 가중치만 추출
        temp_config = config['model'].copy()
        temp_config['num_classes'] = 1000
        full_model = get_model(temp_config).to(device)
        
        with torch.no_grad():
            # 기존 1000개 클래스 중 앞의 100개 가중치를 현재 모델(100개 출력)로 복사
            # 만약 클래스 매핑 리스트가 있다면 인덱스를 맞출 수 있으나, 
            # 일반적인 실험에선 상위 100개 슬라이싱을 기본으로 합니다.
            model.fc.weight.copy_(full_model.fc.weight[:100])
            model.fc.bias.copy_(full_model.fc.bias[:100])
        del full_model # 메모리 확보를 위해 삭제
        print("✅ FC Weight Transfer Successful. Initial Accuracy should be normal.")
    # -------------------------------------------------------------------------

    # [Stage 1] Topology Parsing 수행 (실제 그룹핑용)
    topology_groups = get_model_topology(model)

    # 3. 전략에 따른 실행 분기
    if config['strategy']['method'] == 'PAT':
        execute_pat_experiment(model, config, train_loader, val_loader, test_loader, device, topology_groups)
    elif config['strategy']['method'] == 'PDT':
        execute_pdt_experiment(model, config, train_loader, val_loader, test_loader, device, topology_groups)


def execute_pdt_experiment(model, config, train_loader, val_loader, test_loader, device, topology_groups):
    pdt_engine = PDTPruner(model, config, topology_groups=topology_groups)
    
    model_cfg = config.get('model', {})
    optimizer = optim.SGD(model.parameters(), 
                          lr=model_cfg.get('base_lr', 0.01), 
                          momentum=model_cfg.get('momentum', 0.9), 
                          weight_decay=model_cfg.get('weight_decay', 1e-4))
    criterion = nn.CrossEntropyLoss()
    
    total_epochs = model_cfg.get('epochs', 90)
    strategy_cfg = config.get('strategy', {})
    prune_every = strategy_cfg.get('prune_every', 10)
    start_epoch = strategy_cfg.get('start_epoch', 1)

    analyze_topology_and_profiling(model, device, config, tag="PDT Initial Baseline")
    
    for epoch in range(1, total_epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            output = model(x)
            loss = criterion(output, y)
            
            # [Stage 2] Hessian 계산 및 Pruning 실행 시점
            is_hessian_step = (epoch >= start_epoch and epoch % prune_every == 0 and batch_idx == 0)
            
            if is_hessian_step:
                # OOM 방지를 위해 분석 시에는 이미지 1장만 사용 (수동 패치 로직)
                x_single, y_single = x[:1], y[:1]
                optimizer.zero_grad()
                out_s = model(x_single)
                loss_s = criterion(out_s, y_single)
                loss_s.backward(retain_graph=True) 
                
                print(f"\n>>> [SNOWS Update] Epoch {epoch}: Computing Hessian-Vector Products...")
                
                pdt_engine.step_pruning(loss=loss_s)
                pdt_engine.apply_mask_to_weights(optimizer=optimizer)
                
                analyze_topology_and_profiling(model, device, config, tag=f"PDT Epoch {epoch} - After Mask Update")
                print(f">>> Current Sparsity: {pdt_engine.get_current_sparsity():.2f}%")
                
                torch.cuda.empty_cache()
            else:
                loss.backward()
            
            pdt_engine.update_ema_and_mask_grad()
            optimizer.step()
            pdt_engine.apply_mask_to_weights()
            total_loss += loss.item()

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{total_epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

# [나머지 PAT 및 유틸리티 함수는 기존과 동일하게 유지]
def execute_pat_experiment(model, config, train_loader, val_loader, test_loader, device, topology_groups):
    si_data = maybe_load_or_compute_sensitivity(config, train_loader, test_loader, device)
    pat_engine = PATPruner(model, config, si_data, topology_groups=topology_groups)
    prune_fn = get_prune_fn(config['model']['name'])
    
    n_rounds = config['strategy'].get('n_rounds', 5)
    analyze_topology_and_profiling(model, device, config, tag="PAT Initial Baseline")
    
    for r in range(1, n_rounds + 1):
        print(f"\n--- PAT Round {r}/{n_rounds} ---")
        target_keep_indices = pat_engine.compute_all_keep_indices(round_idx=r) 
        model = prune_fn(model, target_keep_indices, device)
        analyze_topology_and_profiling(model, device, config, tag=f"PAT Round {r} - After Pruning")
        
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        finetune_eps = config['strategy'].get('finetune_epochs', 3)
        for epoch in range(1, finetune_eps + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_acc = evaluate(model, val_loader, device)
            print(f"FT Epoch {epoch} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")

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