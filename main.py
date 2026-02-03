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
    inputs = torch.randn(1, 3, input_size, input_size).to(device)

    # [Stage 1] Topology Status (FX)
    try:
        traced = fx.symbolic_trace(model)
        print(f"[*] Topology Status: {len(list(traced.graph.nodes))} nodes currently in graph.")
    except Exception as e:
        print(f"[*] FX Parsing Warning: {e}")

    # [Stage 4] Resource Profiling
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
    
    # [System] CLI 인자가 YAML 설정을 덮어쓰도록 명시적으로 우선순위 부여
    if hasattr(args, 'channel_keep_ratio') and args.channel_keep_ratio is not None:
        config['strategy']['channel_keep_ratio'] = args.channel_keep_ratio
    if hasattr(args, 'group_selection_ratio') and args.group_selection_ratio is not None:
        config['strategy']['group_selection_ratio'] = args.group_selection_ratio
    if hasattr(args, 'lambda_h') and args.lambda_h is not None:
        config['strategy']['lambda_h'] = args.lambda_h
    if hasattr(args, 'start_epoch') and args.start_epoch is not None:
        config['strategy']['start_epoch'] = args.start_epoch
    if hasattr(args, 'lr') and args.lr is not None:
        config['model']['base_lr'] = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_cfg = config.get('dataset', {})
    dataset_name = dataset_cfg if isinstance(dataset_cfg, str) else dataset_cfg.get('name', 'cifar100')

    loaders = get_dataloaders(config)
    
    if len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, val_loader = loaders
        test_loader = val_loader 
    
    print(f"Experimental Mode: {config['strategy'].get('method', 'N/A')}")
    print(f"Target Model: {config['model']['name']} | Dataset: {dataset_name}")

    # 2. 모델 초기화
    model = get_model(config['model']).to(device)
    
    # [System] ImageNet-100 대응 FC 가중치 이식
    if dataset_name == 'imagenet' or 'imagenet' in dataset_name:
        print(">>> [System] Transferring Pretrained Weights to FC Layer (1000 -> 100)...")
        temp_config = config['model'].copy()
        temp_config['num_classes'] = 1000
        full_model = get_model(temp_config).to(device)
        
        with torch.no_grad():
            model.fc.weight.copy_(full_model.fc.weight[:100])
            model.fc.bias.copy_(full_model.fc.bias[:100])
        del full_model
        print("✅ FC Weight Transfer Successful.")

    # [Stage 1] Topology Parsing
    topology_groups = get_model_topology(model)

    # 3. 전략에 따른 실행 분기
    strategy_method = config['strategy'].get('method', 'pdt').lower()

    if strategy_method == 'pat':
        execute_pat_experiment(model, config, train_loader, val_loader, test_loader, device, topology_groups,args)
    elif strategy_method == 'pdt':
        execute_pdt_experiment(model, config, train_loader, val_loader, test_loader, device, topology_groups, args)


def execute_pdt_experiment(model, config, train_loader, val_loader, test_loader, device, topology_groups, args):
    # 1. Pruner 엔진 생성
    # pdt_engine = PDTPruner(model, config, topology_groups=topology_groups)


    strat_cfg = config.get('strategy', {})

    # 2. [핵심] CLI 인자가 들어왔다면 YAML 설정을 무시하고 덮어쓰기
    # 이렇게 해야 Pruner 내부로 들어갈 때 CLI 값이 최우선이 됩니다.
    if args.group_selection_ratio is not None:
        strat_cfg['group_selection_ratio'] = args.group_selection_ratio
    if args.channel_keep_ratio is not None:
        strat_cfg['channel_keep_ratio'] = args.channel_keep_ratio
    if args.min_survival_ratio is not None:
        strat_cfg['min_survival_ratio'] = args.min_survival_ratio
    
    # 갱신된 설정을 다시 config에 저장
    config['strategy'] = strat_cfg
    pdt_engine = PDTPruner(model, config, args=args, topology_groups=topology_groups)
    
    model_cfg = config.get('model', {})
    strategy_cfg = config.get('strategy', {})
    
    optimizer = optim.SGD(model.parameters(), 
                          lr=model_cfg.get('base_lr', 0.01), 
                          momentum=model_cfg.get('momentum', 0.9), 
                          weight_decay=model_cfg.get('weight_decay', 1e-4))
    criterion = nn.CrossEntropyLoss()
    
    total_epochs = model_cfg.get('epochs', 200)
    prune_every = strategy_cfg.get('prune_every', 20)
    start_epoch = strategy_cfg.get('start_epoch', 50) 

    print(f"\n>>> [Pilot Check] Pruning starts at Epoch {start_epoch}, every {prune_every} epochs.")
    analyze_topology_and_profiling(model, device, config, tag="PDT Initial Baseline")
    
    for epoch in range(1, total_epochs + 1):
        model.train() # 모델을 학습 모드로 설정
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            output = model(x)
            loss = criterion(output, y)
            
            # [Stage 2] 프루닝 시점 판정
            # 선형 스케줄링을 위해 epoch 정보를 전달하도록 수정
            is_hessian_step = (epoch >= start_epoch and (epoch - start_epoch) % prune_every == 0 and batch_idx == 0)
            
            if is_hessian_step:
                optimizer.zero_grad()
                loss.backward(retain_graph=True) 
                before_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
                print(f"\n>>> [SNOWS Update] Epoch {epoch}: Computing Hessian-Vector Products...")
                
                # [핵심 수정] 현재 에폭과 전체 에폭 정보를 Pruner에게 전달하여 스케줄링 수행
                pdt_engine.step_pruning(loss=loss, current_epoch=epoch, total_epochs=total_epochs)
                pdt_engine.apply_mask_to_weights(optimizer=optimizer)
                
                torch.cuda.empty_cache() # 캐시 비워줘야 정확한 수치
                after_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
                
                print(f"[Resource Check] Memory: {before_mem:.2f}MB -> {after_mem:.2f}MB (Reduction: {before_mem - after_mem:.2f}MB)")
                analyze_topology_and_profiling(model, device, config, tag=f"PDT Epoch {epoch} - After Mask Update")
                print(f">>>> Current Model Sparsity: {pdt_engine.get_current_sparsity():.2f}%")
                
                torch.cuda.empty_cache()
            else:
                loss.backward()
            
            pdt_engine.update_ema_and_mask_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 매 스텝마다 마스크를 적용하여 죽은 채널이 학습되지 않도록 고정
            pdt_engine.apply_mask_to_weights()
            total_loss += loss.item()

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{total_epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

def execute_pat_experiment(model, config, train_loader, val_loader, test_loader, device, topology_groups,args):
    
    checkpoint_dir = config.get('save_dir', './exp/checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    base_ckpt_path = os.path.join(checkpoint_dir, f"{config['model']['name']}_base.pth")
    
    if not os.path.exists(base_ckpt_path):
        print(f">>> [System] Saving base weights for sensitivity analysis to {base_ckpt_path}")
        torch.save(model.state_dict(), base_ckpt_path)
    
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