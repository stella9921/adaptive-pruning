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
from src.pruning.pdt_strategies import PDTPruner,HAPPruner,SNOWSPruner

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

    # if strategy_method == 'pat':
    #     import torchvision.models as tv_models
    #     print(">>> [System] PAT Mode: Loading Torchvision Pretrained Weights for Sensitivity Analysis...")
    #     pretrained_vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
    #     model.load_state_dict(pretrained_vgg.state_dict(), strict=False)
    #     print("✅ Pretrained Weights Loaded for PAT only.")
    # else:
    #     print(">>> [System] PDT Mode: Starting from Scratch (No Pretrained Weights).")



    if strategy_method == 'pat':
        execute_pat_experiment(model, config, train_loader, val_loader, test_loader, device, topology_groups,args)
    elif strategy_method == 'pdt':
        execute_pdt_experiment(model, config, train_loader, val_loader, test_loader, device, topology_groups, args)


def execute_pdt_experiment(model, config, train_loader, val_loader, test_loader, device, topology_groups, args):
    # --- [0] 저장 경로 설정 (추가됨) ---
    checkpoint_dir = config.get('save_dir', './exp/checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- [1] CLI 인자가 YAML 설정을 덮어쓰도록 우선 적용 ---
    strat_cfg = config.get('strategy', {})
    if args.group_selection_ratio is not None:
        strat_cfg['group_selection_ratio'] = args.group_selection_ratio
    if args.channel_keep_ratio is not None:
        strat_cfg['channel_keep_ratio'] = args.channel_keep_ratio
    if args.min_survival_ratio is not None:
        strat_cfg['min_survival_ratio'] = args.min_survival_ratio
    
    config['strategy'] = strat_cfg 

    # --- [2] Pruner 전략 분기 및 엔진 생성 ---
    strategy_type = getattr(args, 'strategy', 'pdt').lower()

    if strategy_type == 'hap':
        pdt_engine = HAPPruner(model, config, args, topology_groups)
        print("[System] Initializing HAPPruner (Hessian Inverse Strategy)")
    elif strategy_type == 'snows':
        pdt_engine = SNOWSPruner(model, config, args, topology_groups)
        print("[System] Initializing SNOWSPruner (Pure Hessian Trace Strategy)")
    else:
        pdt_engine = PDTPruner(model, config, args, topology_groups)
        print("[System] Initializing PDTPruner (Proposed Grad-EMA Strategy)")

    # --- [3] 학습 설정 준비 ---
    model_cfg = config.get('model', {})
    optimizer = optim.SGD(model.parameters(), 
                          lr=model_cfg.get('base_lr', 0.01), 
                          momentum=model_cfg.get('momentum', 0.9), 
                          weight_decay=model_cfg.get('weight_decay', 1e-4))
    criterion = nn.CrossEntropyLoss()
    
    total_epochs = model_cfg.get('epochs', 400) 
    prune_every = strat_cfg.get('prune_every', 20)
    start_epoch = strat_cfg.get('start_epoch', 120) 

    print(f"\n>>> [Pilot Check] Strategy: {strategy_type.upper()} | Pruning starts at Epoch {start_epoch}, every {prune_every} epochs.")
    analyze_topology_and_profiling(model, device, config, tag="Baseline Before Pruning")
    
    # --- [4] 메인 학습 루프 ---
    for epoch in range(1, total_epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            output = model(x)
            loss = criterion(output, y)
            
            is_hessian_step = (epoch >= start_epoch and (epoch - start_epoch) % prune_every == 0 and batch_idx == 0)
            
            if is_hessian_step:
                optimizer.zero_grad()
                loss.backward(retain_graph=True) 
                
                pdt_engine.step_pruning(loss=loss, current_epoch=epoch, total_epochs=total_epochs)
                pdt_engine.apply_mask_to_weights(optimizer=optimizer)
                
                # ---------------- [추가: 프루닝 시점 저장] ----------------
                current_sp = pdt_engine.get_current_sparsity()
                ckpt_name = f"{model_cfg['name']}_{strategy_type}_ep{epoch}_sp{current_sp:.1f}.pth"
                save_path = os.path.join(checkpoint_dir, ckpt_name)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'sparsity': current_sp,
                }, save_path)
                print(f"💾 Checkpoint saved: {save_path}")
                # -------------------------------------------------------

                torch.cuda.empty_cache()
                analyze_topology_and_profiling(model, device, config, tag=f"{strategy_type.upper()} Epoch {epoch} Update")
            else:
                loss.backward()
            
            pdt_engine.update_ema_and_mask_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            pdt_engine.apply_mask_to_weights()
            total_loss += loss.item()

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{total_epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

    # ---------------- [추가: 최종 학습 완료 저장] ----------------
    final_path = os.path.join(checkpoint_dir, f"{model_cfg['name']}_{strategy_type}_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"🏁 Final Model Saved: {final_path}")
    # ----------------------------------------------------------

def execute_pat_experiment(model, config, train_loader, val_loader, test_loader, device, topology_groups,args):
    
    checkpoint_dir = config.get('save_dir', './exp/checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    base_ckpt_path = os.path.join(checkpoint_dir, f"{config['model']['name']}_base.pth")
    
    if not os.path.exists(base_ckpt_path):
        print(f"\n>>> [PAT Pre-train] Base weights not found. Training for 120 epochs first...")
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        # CIFAR-100 정석 스케줄러 (Cos-Annealing)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, 121):
            # 기존에 main.py에 정의된 train_one_epoch와 evaluate 함수 활용
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_acc = evaluate(model, val_loader, device)
            scheduler.step()
            
            if epoch % 10 == 0 or epoch == 1:
                print(f" [Pre-train] Epoch {epoch}/120 | Loss: {train_loss:.4f} | Acc: {val_acc:.2f}%")
        
        # 학습 완료된 '똑똑한 모델' 저장
        torch.save(model.state_dict(), base_ckpt_path)
        print(f"✅ Pre-training complete. Saved to {base_ckpt_path}")
    else:
        print(f">>> [System] Found existing base model at {base_ckpt_path}. Loading for PAT...")
        model.load_state_dict(torch.load(base_ckpt_path, map_location=device))
    
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