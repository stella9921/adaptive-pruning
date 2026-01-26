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
    
    # [입력 크기 대응] CIFAR-100(32) vs ImageNet(224)
    # config/model/resnet152_imagenet.yaml에 정의된 input_size를 사용
    input_size = config['model'].get('input_size', 224)
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
    
    # ImageNet은 보통 train/val만 존재하므로 유연하게 대응
    loaders = get_dataloaders(config)
    if len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, val_loader = loaders
        test_loader = val_loader # Test가 없으면 Validation으로 대체
    
    print(f"Experimental Mode: {config['strategy']['method']}")
    print(f"Target Model: {config['model']['name']} | Dataset: {config['dataset'].get('name', 'N/A')}")

    # 2. 모델 초기화 (weights='IMAGENET1K_V1' 등은 get_model 내부에서 처리되어야 함)
    model = get_model(config['model']).to(device)
    
    # [Stage 1] Topology Parsing 수행 (실제 그룹핑용)
    topology_groups = get_model_topology(model)

    # 3. 전략에 따른 실행 분기
    if config['strategy']['method'] == 'PAT':
        execute_pat_experiment(model, config, train_loader, val_loader, test_loader, device, topology_groups)
    elif config['strategy']['method'] == 'PDT':
        execute_pdt_experiment(model, config, train_loader, val_loader, test_loader, device, topology_groups)


# -------------------------------------------------------------------------
# [Method A] PAT: topology_groups 반영 (기존 유지)
# -------------------------------------------------------------------------
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


# -------------------------------------------------------------------------
# [Method B] PDT: ImageNet 및 ResNet-152 최적화 반영
# -------------------------------------------------------------------------
def execute_pdt_experiment(model, config, train_loader, val_loader, test_loader, device, topology_groups):
    pdt_engine = PDTPruner(model, config, topology_groups=topology_groups)
    
    # Config에서 학습률 및 하이퍼파라미터 로드
    optimizer = optim.SGD(model.parameters(), 
                          lr=config['model'].get('base_lr', 0.01), 
                          momentum=config['model'].get('momentum', 0.9), 
                          weight_decay=config['model'].get('weight_decay', 1e-4))
    criterion = nn.CrossEntropyLoss()
    
    total_epochs = config['model']['epochs']
    prune_every = config['strategy'].get('prune_every', 10)
    start_epoch = config['strategy'].get('start_epoch', 1)

    analyze_topology_and_profiling(model, device, config, tag="PDT Initial Baseline")
    
    for epoch in range(1, total_epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            output = model(x)
            loss = criterion(output, y)
            
            # Pruning 시점 결정 (Hessian 계산)
            is_hessian_step = (epoch >= start_epoch and epoch % prune_every == 0 and batch_idx == 0)
            
            if is_hessian_step:
                # Hessian 연산 시 메모리 확보를 위해 명시적 backward 조절
                loss.backward(retain_graph=True) 
                print(f"\n>>> [SNOWS Update] Epoch {epoch}: Computing Hessian-Vector Products...")
                
                # 라그랑주 최적화 엔진 호출
                pdt_engine.step_pruning(loss=loss)
                pdt_engine.apply_mask_to_weights(optimizer=optimizer)
                
                # 프루닝 직후 리소스 변화 분석
                analyze_topology_and_profiling(model, device, config, tag=f"PDT Epoch {epoch} - After Mask Update")
                print(f">>> Current Sparsity: {pdt_engine.get_current_sparsity():.2f}%")
                
                # Hessian 계산 후 불필요한 캐시 정리 (ResNet-152 OOM 방지)
                torch.cuda.empty_cache()
            else:
                loss.backward()
            
            # 매 배치마다 Grad EMA 업데이트 및 가중치 마스킹
            pdt_engine.update_ema_and_mask_grad()
            optimizer.step()
            pdt_engine.apply_mask_to_weights()
            total_loss += loss.item()

        # 에폭 종료 후 평가
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