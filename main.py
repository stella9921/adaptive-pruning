import torch.fx as fx  # Stage 1용
from torch.profiler import profile, record_function, ProfilerActivity # Stage 4용
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch_pruning as tp
import sys
from datetime import datetime
import json

# 기존 모듈 임포트
from src.pruning.topology_manager import get_model_topology
from src.pruning.optimizer import lagrangian_optimization
from src.utils.config_loader import load_config
from src.data.dataloader import get_dataloaders
from src.models import get_model, get_prune_fn
from src.pruning.sensitivity import maybe_load_or_compute_sensitivity
from src.pruning.pat_strategies import PATPruner
from src.pruning.pdt_strategies import PDTPruner,HAPPruner,SNOWSPruner, ATOPruner, STPruner,DFPCPruner,TPPPruner

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

    # --- 로그 파일 자동 저장 설정 시작 ---
    log_dir = config.get('save_dir', './exp/logs')
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{config['model']['name']}_{getattr(args, 'strategy', 'pdt')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(os.path.join(log_dir, log_filename), "a")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self): pass

    sys.stdout = Logger() # 이제 모든 print문이 파일로 
    print(f"📝 Logging started: {log_filename}")
    
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
    # --- [0] 저장 및 로드 경로 설정 ---
    checkpoint_dir = config.get('save_dir', './exp/checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_cfg = config.get('model', {})
    model_name = model_cfg['name']

    # --- [1] 중요: Pruner 엔진을 먼저 생성 (모델에 mask, hessian 공간 확보) ---
    strat_cfg = config.get('strategy', {})
    # CLI 인자 반영
    if args.start_epoch is not None: strat_cfg['start_epoch'] = args.start_epoch
    if args.channel_keep_ratio is not None: strat_cfg['channel_keep_ratio'] = args.channel_keep_ratio
    config['strategy'] = strat_cfg
    strategy_type = getattr(args, 'strategy', 'pdt').lower()

    # 여기서 Pruner를 먼저 만들어야 model에 "mask", "hessian_score" 등이 등록됩니다.
    pruner_class_name = f"{strategy_type.upper()}Pruner"
    if pruner_class_name in globals():
        pdt_engine = globals()[pruner_class_name](model, config, args, topology_groups)
    else:
        pdt_engine = PDTPruner(model, config, args, topology_groups)

    # --- [2] 이제 가중치 로드 (순서 변경됨) ---
    base_candidates = [
        os.path.join(checkpoint_dir, f"{model_name}_base.pth"),
        os.path.join(checkpoint_dir, f"{model_name}_pdt_ep120_sp0.0.pth")
    ]
    
    loaded_from_base = False
    for path in base_candidates:
        if os.path.exists(path):
            print(f"\n>>> [System] Loading pre-trained model: {path}")
            state_dict = torch.load(path, map_location=device)
            
            # 딕셔너리 구조 대응
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                sd = state_dict['model_state_dict']
            elif isinstance(state_dict, dict) and 'model' in state_dict:
                sd = state_dict['model']
            else:
                sd = state_dict
            
            # 🚨 핵심: strict=False를 주어 Hessian/Mask 키가 달라도 로드되게 함
            model.load_state_dict(sd, strict=False)
            print(f"✅ Success: Weights loaded from {os.path.basename(path)}")
            loaded_from_base = True
            break

    # --- [3] 나머지 학습 설정 ---
    optimizer = optim.SGD(model.parameters(), 
                          lr=model_cfg.get('base_lr', 0.01), 
                          momentum=model_cfg.get('momentum', 0.9), 
                          weight_decay=model_cfg.get('weight_decay', 1e-4))
    criterion = nn.CrossEntropyLoss()
    
    total_epochs = model_cfg.get('epochs', 400) 
    prune_every = strat_cfg.get('prune_every', 20)
    start_epoch = strat_cfg.get('start_epoch', 120) 

    # 모델 로드 성공 시 루프 시작점을 start_epoch로 고정하여 즉시 프루닝 유도
    loop_start = start_epoch if loaded_from_base else 1

    # 통합 로그 파일 설정
    history_file = os.path.join(checkpoint_dir, f"{model_name}_{strategy_type}_history.json")
    history_data = []

    print(f"\n>>> [Pilot Check] Strategy: {strategy_type.upper()} | Start Epoch: {start_epoch} | Target Ratio: {strat_cfg.get('channel_keep_ratio')}")
    analyze_topology_and_profiling(model, device, config, tag="Initial State")
    
    # --- [5] 메인 학습 & 프루닝 루프 ---
    for epoch in range(loop_start, total_epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            output = model(x)
            loss = criterion(output, y)
            
            # 프루닝 스텝 판별 (start_epoch 당일 첫 배치에서 즉시 작동)
            is_hessian_step = (epoch >= start_epoch and (epoch - start_epoch) % prune_every == 0 and batch_idx == 0)
            
            if is_hessian_step:
                optimizer.zero_grad()
                loss.backward(retain_graph=True) 
                
                print(f"\n[Action] Triggering {strategy_type.upper()} Pruning at Epoch {epoch}...")
                pdt_engine.step_pruning(loss=loss, current_epoch=epoch, total_epochs=total_epochs)
                pdt_engine.apply_mask_to_weights(optimizer=optimizer)
                
                # 🟢 학회용 리소스 지표 즉시 출력
                eff = pdt_engine.get_model_efficiency()
                print(f"\n[Scientific Metrics - Epoch {epoch}]")
                print(f" 🟢 Model Size: {eff['curr_mb']:.2f} MB | 🔵 Sparsity: {eff['sparsity']:.2f} % | 🟡 Speedup: {eff['speedup']:.2f}x")

                # 💾 JSON 로그 및 체크포인트 저장
                val_acc = evaluate(model, val_loader, device)
                history_data.append({'epoch': epoch, 'val_acc': val_acc, 'sparsity': eff['sparsity'], 'size_mb': eff['curr_mb']})
                with open(history_file, 'w') as f: json.dump(history_data, f, indent=4)

                save_path = os.path.join(checkpoint_dir, f"{model_name}_{strategy_type}_ep{epoch}_sp{eff['sparsity']:.1f}.pth")
                torch.save({'model_state_dict': model.state_dict(), 'sparsity': eff['sparsity']}, save_path)
                print(f"💾 Checkpoint saved: {save_path}")
                torch.cuda.empty_cache()
            else:
                loss.backward()
            
            pdt_engine.update_ema_and_mask_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            pdt_engine.apply_mask_to_weights()
            total_loss += loss.item()

        # 에폭 종료 후 결과 출력
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{total_epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

    # --- [6] 최종 물리적 압축 (Stage 5) ---
    final_path = os.path.join(checkpoint_dir, f"{model_name}_{strategy_type}_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"🏁 Final Model Saved: {final_path}")

    print("\n" + "="*30 + " FINAL PHYSICAL COMPRESSION " + "="*30)


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