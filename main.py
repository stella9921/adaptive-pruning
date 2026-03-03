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
    # [main.py 하단]
    elif strategy_method == 'pdt':
        # 인자 순서: model, config, train, val, test, device, topology_groups, args
        execute_pdt_experiment(
            model, 
            config, 
            train_loader, 
            val_loader, 
            test_loader, 
            device, 
            topology_groups,  # <--- 이 변수가 정확히 7번째 또는 8번째에 있는지 확인!
            args
        )
    # elif strategy_method == 'pdt':
    #     execute_pdt_experiment(model, config, train_loader, val_loader, test_loader, device, topology_groups, args)

# def execute_pdt_experiment(model, config, train_loader, val_loader, test_loader, device, topology_groups, args):
#     # --- [0] 저장 경로 설정 ---
#     checkpoint_dir = config.get('save_dir', './exp/checkpoints')
#     os.makedirs(checkpoint_dir, exist_ok=True)

#     # --- [통합 로그 파일 설정] ---
#     strategy_type = getattr(args, 'strategy', 'pdt').lower()
#     history_file = os.path.join(checkpoint_dir, f"{config['model']['name']}_{strategy_type}_history.json")
#     history_data = [] # 에폭별 수치를 담을 리스트

#     # --- [1] CLI 인자가 YAML 설정을 덮어쓰도록 우선 적용 ---
#     strat_cfg = config.get('strategy', {})
#     if args.group_selection_ratio is not None:
#         strat_cfg['group_selection_ratio'] = args.group_selection_ratio
#     if args.channel_keep_ratio is not None:
#         strat_cfg['channel_keep_ratio'] = args.channel_keep_ratio
#     if args.min_survival_ratio is not None:
#         strat_cfg['min_survival_ratio'] = args.min_survival_ratio
    
#     config['strategy'] = strat_cfg 

#     # --- [2] Pruner 전략 분기 및 엔진 생성 ---
#     if strategy_type == 'hap':
#         pdt_engine = HAPPruner(model, config, args, topology_groups)
#         print("[System] Initializing HAPPruner (Hessian Inverse Strategy)")
#     elif strategy_type == 'snows':
#         pdt_engine = SNOWSPruner(model, config, args, topology_groups)
#         print("[System] Initializing SNOWSPruner (Pure Hessian Trace Strategy)")
#     elif strategy_type == 'ato':
#         pdt_engine = ATOPruner(model, config, args, topology_groups)
#         print("[System] Initializing ATOPruner (Magnitude Strategy)")
#     elif strategy_type == 'st':
#         pdt_engine = STPruner(model, config, args, topology_groups)
#         print("[System] Initializing STPruner (SuperTickets Strategy)")
#     elif strategy_type == 'dfpc':
#         pdt_engine = DFPCPruner(model, config, args, topology_groups)
#         print("[System] Initializing DFPCPruner (Geometric Uniqueness Strategy)")
#     elif strategy_type == 'tpp':
#         pdt_engine = TPPPruner(model, config, args, topology_groups)
#         print("[System] Initializing TPPPruner (Personalized Importance Strategy)")
#     else:
#         pdt_engine = PDTPruner(model, config, args, topology_groups)
#         print("[System] Initializing PDTPruner (Proposed Grad-EMA Strategy)")

#     # --- [3] 학습 설정 준비 ---
#     model_cfg = config.get('model', {})
#     optimizer = optim.SGD(model.parameters(), 
#                           lr=model_cfg.get('base_lr', 0.01), 
#                           momentum=model_cfg.get('momentum', 0.9), 
#                           weight_decay=model_cfg.get('weight_decay', 1e-4))
#     criterion = nn.CrossEntropyLoss()
    
#     total_epochs = model_cfg.get('epochs', 400) 
#     prune_every = strat_cfg.get('prune_every', 20)
#     start_epoch = strat_cfg.get('start_epoch', 120) 

#     print(f"\n>>> [Pilot Check] Strategy: {strategy_type.upper()} | Pruning starts at Epoch {start_epoch}, every {prune_every} epochs.")
#     # Baseline 분석은 처음에 한 번만 수행
#     analyze_topology_and_profiling(model, device, config, tag="Baseline Before Pruning")
    
#     # --- [4] 메인 학습 루프 ---
#     for epoch in range(1, total_epochs + 1):
#         model.train()
#         total_loss = 0.0
        
#         for batch_idx, (x, y) in enumerate(train_loader):
#             x, y = x.to(device), y.to(device)
#             optimizer.zero_grad()
            
#             output = model(x)
#             loss = criterion(output, y)
            
#             # 프루닝 스텝 판별 (지정된 에폭 및 배치의 첫 번째 step)
#             is_hessian_step = (epoch >= start_epoch and (epoch - start_epoch) % prune_every == 0 and batch_idx == 0)
            
#             if is_hessian_step:
#                 optimizer.zero_grad()
#                 loss.backward(retain_graph=True) 
                
#                 # [순서 변경] 1. 프루닝 실행 (상세 표 출력)
#                 pdt_engine.step_pruning(loss=loss, current_epoch=epoch, total_epochs=total_epochs)
#                 pdt_engine.apply_mask_to_weights(optimizer=optimizer)
                
#                 # [추가] 2. 학회용 요약 지표 (MB, Sparsity) 즉시 출력
#                 # pdt_strategies.py의 각 Pruner에 추가된 print_scientific_metrics가 호출됨
#                 if hasattr(pdt_engine, 'print_scientific_metrics'):
#                     pdt_engine.print_scientific_metrics(epoch, strategy_type.upper())
#                 else:
#                     eff = pdt_engine.get_model_efficiency()
#                     print(f"\n[Scientific Metrics - Epoch {epoch}]")
#                     print(f" 🟢 Model Size: {eff['curr_mb']:.2f} MB | 🔵 Sparsity: {eff['sparsity']:.2f} %")

#                 # 3. 데이터 수집 및 JSON 로그 업데이트
#                 val_acc = evaluate(model, val_loader, device)
#                 current_sp = pdt_engine.get_current_sparsity()
#                 eff = pdt_engine.get_model_efficiency()
                
#                 epoch_stats = {
#                     'epoch': epoch,
#                     'loss': total_loss / (batch_idx + 1),
#                     'val_acc': val_acc,
#                     'sparsity': current_sp,
#                     'model_size_mb': eff['curr_mb'],
#                     'reduction_mb': eff['orig_mb'] - eff['curr_mb'],
#                     'speedup': eff['speedup']
#                 }
#                 history_data.append(epoch_stats)

#                 with open(history_file, 'w') as f:
#                     json.dump({
#                         'config': {
#                             'model': config['model']['name'],
#                             'strategy': strategy_type,
#                             'target_keep_ratio': config['strategy']['channel_keep_ratio']
#                         },
#                         'history': history_data
#                     }, f, indent=4)

#                 # 4. 프루닝 시점 체크포인트 저장 (.pth)
#                 ckpt_name = f"{model_cfg['name']}_{strategy_type}_ep{epoch}_sp{current_sp:.1f}.pth"
#                 save_path = os.path.join(checkpoint_dir, ckpt_name)
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': model.state_dict(),
#                     'sparsity': current_sp,
#                     'stats': epoch_stats
#                 }, save_path)
#                 print(f"💾 Checkpoint saved: {save_path}")
                
#                 # 5. 프로파일러는 가장 마지막에 실행 (로그 밀림 방지 위해 필요 시에만 주석 해제)
#                 torch.cuda.empty_cache()
#                 # analyze_topology_and_profiling(model, device, config, tag=f"{strategy_type.upper()} Epoch {epoch} Update")
                
#             else:
#                 loss.backward()
            
#             pdt_engine.update_ema_and_mask_grad()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
            
#             pdt_engine.apply_mask_to_weights()
#             total_loss += loss.item()

#         # 에폭 종료 후 결과 출력
#         val_acc = evaluate(model, val_loader, device)
#         print(f"Epoch {epoch}/{total_epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

#     # --- [5] 최종 학습 완료 저장 ---
#     final_path = os.path.join(checkpoint_dir, f"{model_cfg['name']}_{strategy_type}_final.pth")
#     torch.save(model.state_dict(), final_path)
#     print(f"🏁 Final Model Saved: {final_path}")

#     # --- [6] 물리적 압축 및 메모리 제약 해소 검증 (Stage 5) ---
#     print("\n" + "="*30 + " FINAL PHYSICAL COMPRESSION " + "="*30)
#     torch.cuda.empty_cache()
#     mem_before = torch.cuda.memory_allocated(device) / (1024**2)
    
#     model.cpu() 
#     example_inputs = torch.randn(1, 3, config['model'].get('input_size', 224), config['model'].get('input_size', 224))
#     DG = tp.DependencyGraph().build_graph(model, example_inputs=example_inputs)

#     for name, m in model.named_modules():
#         if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, 'mask'):
#             prune_indices = torch.where(m.mask == 0)[0].tolist()
#             if len(prune_indices) > 0:
#                 pruning_plan = DG.get_pruning_plan(m, tp.prune_conv_out_channels, idxs=prune_indices)
#                 pruning_plan.exec()

#     model.to(device)
#     torch.cuda.empty_cache()
#     mem_after = torch.cuda.memory_allocated(device) / (1024**2)
    
#     print(f"[*] Before: {mem_before:.2f} MB | After: {mem_after:.2f} MB")
#     print(f"[*] Memory Saved: {mem_before - mem_after:.2f} MB ({((mem_before-mem_after)/mem_before)*100:.1f}% reduction)")
    
#     compressed_path = os.path.join(checkpoint_dir, f"{model_cfg['name']}_{strategy_type}_FINAL_COMPRESSED.pth")
#     torch.save(model.state_dict(), compressed_path)
#     print(f"✅ Physically Compressed Model Saved: {compressed_path}")
#     print("="*89 + "\n")
# def execute_pdt_experiment(model, config, train_loader, val_loader, test_loader, device, topology_groups, args):
#     # --- [0] 저장 경로 설정 ---
#     checkpoint_dir = config.get('save_dir', './exp/checkpoints')
#     os.makedirs(checkpoint_dir, exist_ok=True)

#     # --- [통합 로그 파일 설정] ---
#     strategy_type = getattr(args, 'strategy', 'pdt').lower()
#     history_file = os.path.join(checkpoint_dir, f"{config['model']['name']}_{strategy_type}_history.json")
#     history_data = [] 

#     # --- [1] CLI 인자가 YAML 설정을 덮어쓰도록 우선 적용 ---
#     strat_cfg = config.get('strategy', {})
#     if args.group_selection_ratio is not None:
#         strat_cfg['group_selection_ratio'] = args.group_selection_ratio
#     if args.channel_keep_ratio is not None:
#         strat_cfg['channel_keep_ratio'] = args.channel_keep_ratio
#     if args.min_survival_ratio is not None:
#         strat_cfg['min_survival_ratio'] = args.min_survival_ratio
    
#     config['strategy'] = strat_cfg 

#     # --- [2] Pruner 전략 분기 및 엔진 생성 ---
#     if strategy_type == 'hap':
#         pdt_engine = HAPPruner(model, config, args, topology_groups)
#         print("[System] Initializing HAPPruner (Hessian Inverse Strategy)")
#     elif strategy_type == 'snows':
#         pdt_engine = SNOWSPruner(model, config, args, topology_groups)
#         print("[System] Initializing SNOWSPruner (Pure Hessian Trace Strategy)")
#     elif strategy_type == 'ato':
#         pdt_engine = ATOPruner(model, config, args, topology_groups)
#         print("[System] Initializing ATOPruner (Magnitude Strategy)")
#     elif strategy_type == 'st':
#         pdt_engine = STPruner(model, config, args, topology_groups)
#         print("[System] Initializing STPruner (SuperTickets Strategy)")
#     elif strategy_type == 'dfpc':
#         pdt_engine = DFPCPruner(model, config, args, topology_groups)
#         print("[System] Initializing DFPCPruner (Geometric Uniqueness Strategy)")
#     elif strategy_type == 'tpp':
#         pdt_engine = TPPPruner(model, config, args, topology_groups)
#         print("[System] Initializing TPPPruner (Personalized Importance Strategy)")
#     else:
#         pdt_engine = PDTPruner(model, config, args, topology_groups=topology_groups)
#         print("[System] Initializing PDTPruner (Proposed Grad-EMA Strategy)")

#     # --- [3] 학습 설정 및 명령어 에폭 연동 ---
#     model_cfg = config.get('model', {})
#     optimizer = optim.SGD(model.parameters(), 
#                           lr=model_cfg.get('base_lr', 0.01), 
#                           momentum=model_cfg.get('momentum', 0.9), 
#                           weight_decay=model_cfg.get('weight_decay', 1e-4))
#     criterion = nn.CrossEntropyLoss()
    
#     # 명령어(--epochs, --prune_every 등) 입력을 최우선으로 반영
#     total_epochs = args.epochs if (hasattr(args, 'epochs') and args.epochs) else model_cfg.get('epochs', 400) 
#     prune_every = args.prune_every if (hasattr(args, 'prune_every') and args.prune_every) else strat_cfg.get('prune_every', 20)
#     start_epoch = args.start_epoch if (hasattr(args, 'start_epoch') and args.start_epoch) else strat_cfg.get('start_epoch', 120) 

#     # Pruner 엔진 내부의 total_epochs 변수 갱신 (진행률 계산 정확도 확보)
#     pdt_engine.total_epochs = total_epochs

#     print(f"\n>>> [Pilot Check] Strategy: {strategy_type.upper()} | Target Epochs: {total_epochs}")
#     print(f">>> Pruning starts at Epoch {start_epoch}, every {prune_every} epochs.")
    
#     analyze_topology_and_profiling(model, device, config, tag="Baseline Before Pruning")
    
#     # --- [4] 메인 학습 루프 ---
#     for epoch in range(1, total_epochs + 1):
#         model.train()
#         total_loss = 0.0
        
#         # 매 에폭 시작 시 GPU Peak 메모리 통계 리셋
#         torch.cuda.reset_peak_memory_stats(device)
        
#         for batch_idx, (x, y) in enumerate(train_loader):
#             x, y = x.to(device), y.to(device)
#             optimizer.zero_grad()
            
#             output = model(x)
#             loss = criterion(output, y)
            
#             # 프루닝 스텝 판별 (명령어 인자 기반)
#             is_hessian_step = (epoch >= start_epoch and (epoch - start_epoch) % prune_every == 0 and batch_idx == 0)
            
#             if is_hessian_step:
#                 optimizer.zero_grad()
#                 loss.backward(retain_graph=True) 
                
#                 # 프루닝 실행 및 마스크 적용
#                 pdt_engine.step_pruning(loss=loss, current_epoch=epoch, total_epochs=total_epochs)
#                 pdt_engine.apply_mask_to_weights(optimizer=optimizer)
#                 torch.cuda.empty_cache()
#             else:
#                 loss.backward()
            
#             pdt_engine.update_ema_and_mask_grad()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
            
#             pdt_engine.apply_mask_to_weights()
#             total_loss += loss.item()

#         # 에폭 종료 후 검증 및 Peak VRAM 출력
#         val_acc = evaluate(model, val_loader, device)
#         peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)
        
#         print(f"Epoch {epoch}/{total_epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}% | Peak VRAM: {peak_vram:.2f} MB")

#         # --- [중간 체크포인트 & 로그 저장] ---
#         if is_hessian_step:
#             current_sp = pdt_engine.get_current_sparsity()
#             eff = pdt_engine.get_model_efficiency()
            
#             epoch_stats = {
#                 'epoch': epoch,
#                 'val_acc': val_acc,
#                 'sparsity': current_sp,
#                 'peak_vram_mb': peak_vram,
#                 'model_size_mb': eff['curr_mb'],
#                 'speedup': eff['speedup']
#             }
#             history_data.append(epoch_stats)
#             with open(history_file, 'w') as f:
#                 json.dump({'history': history_data}, f, indent=4)

#             ckpt_name = f"{model_cfg['name']}_{strategy_type}_ep{epoch}_sp{current_sp:.1f}.pth"
#             torch.save({'model_state_dict': model.state_dict(), 'sparsity': current_sp}, os.path.join(checkpoint_dir, ckpt_name))
#             print(f"💾 Checkpoint saved: {ckpt_name}")

#     # --- [5] 최종 학습 완료 저장 ---
#     final_path = os.path.join(checkpoint_dir, f"{model_cfg['name']}_{strategy_type}_final.pth")
#     torch.save(model.state_dict(), final_path)
#     print(f"🏁 Final Model Saved: {final_path}")

#     # --- [6] 물리적 압축 및 메모리 제약 해소 검증 (Stage 5) ---
#     print("\n" + "="*30 + " FINAL PHYSICAL COMPRESSION " + "="*30)
#     torch.cuda.empty_cache()
#     model.to(device)
#     torch.cuda.synchronize()
#     mem_before = torch.cuda.memory_allocated(device) / (1024**2)
    
#     # 물리적 압축을 위한 의존성 그래프 빌드 (GPU 상에서 수행)
#     example_inputs = torch.randn(1, 3, config['model'].get('input_size', 224), config['model'].get('input_size', 224)).to(device)
#     DG = tp.DependencyGraph().build(model, example_inputs=example_inputs)

#     for name, m in model.named_modules():
#         if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, 'mask'):
#             prune_indices = torch.where(m.mask == 0)[0].tolist()
#             if len(prune_indices) > 0:
#                 pruning_plan = DG.get_pruning_plan(m, tp.prune_conv_out_channels, idxs=prune_indices)
#                 pruning_plan.exec()

#     torch.cuda.synchronize()
#     mem_after = torch.cuda.memory_allocated(device) / (1024**2)
    
#     print(f"[*] Before: {mem_before:.2f} MB | After: {mem_after:.2f} MB")
#     print(f"[*] Memory Saved: {mem_before - mem_after:.2f} MB ({((mem_before-mem_after)/mem_before)*100:.1f}% reduction)")
    
#     compressed_path = os.path.join(checkpoint_dir, f"{model_cfg['name']}_{strategy_type}_FINAL_COMPRESSED.pth")
#     torch.save(model.state_dict(), compressed_path)
#     print(f"✅ Physically Compressed Model Saved: {compressed_path}")
#     print("="*89 + "\n")


def execute_pdt_experiment(model, config, train_loader, val_loader, test_loader, device, topology_groups, args):

    import torch_pruning as tp
    import json

    checkpoint_dir = config.get('save_dir', './exp/checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    strategy_type = getattr(args, 'strategy', 'pdt').lower()

    # -------------------------
    # Pruner 생성
    # -------------------------
    pdt_engine = PDTPruner(model, config, args, topology_groups=topology_groups)
    print("[System] Initializing PDTPruner (Stable Version)")

    optimizer = optim.SGD(
        model.parameters(),
        lr=config['model'].get('base_lr', 0.01),
        momentum=config['model'].get('momentum', 0.9),
        weight_decay=config['model'].get('weight_decay', 1e-4)
    )
    criterion = nn.CrossEntropyLoss()

    total_epochs = args.epochs
    prune_every = args.prune_every
    start_epoch = args.start_epoch

    print(f"\n>>> Strategy: {strategy_type.upper()} | Total Epochs: {total_epochs}")
    print(f">>> Pruning starts at Epoch {start_epoch}, every {prune_every} epochs.")

    stop_pruning = False
    history_data = []

    # ============================================================
    # ======================= TRAIN LOOP =========================
    # ============================================================
    for epoch in range(1, total_epochs + 1):

        model.train()
        total_loss = 0.0
        torch.cuda.reset_peak_memory_stats(device)

         # 🔥 step-level memory trace 저장용
        step_mem_trace = []
        prune_step_index = None

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            output = model(x)
            loss = criterion(output, y)




            # --------------------------------------------------------
            # [추가] 프루닝 시작 직전 에폭(start_epoch - 1)에 Dense 수치 저장
            # --------------------------------------------------------
            if epoch == start_epoch and batch_idx == 0:
                print(f"\n[System] Saving Baseline Dense Model before pruning starts...")
                
                # Dense 상태의 효율성 측정 (Sparsity 0%)
                dense_eff = pdt_engine.get_model_efficiency()
                dense_acc = evaluate(model, val_loader, device)
                
                dense_ckpt_name = f"{config['model']['name']}_DENSE_BASELINE.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch - 1,
                    'sparsity': 0.0,
                    'val_acc': dense_acc,
                    'metrics': dense_eff
                }, os.path.join(checkpoint_dir, dense_ckpt_name))
                
                print(f"📊 Dense Baseline - Acc: {dense_acc:.2f}%, Size: {dense_eff['orig_mb']:.2f} MB")
                print(f"💾 Saved Baseline: {dense_ckpt_name}\n")

            should_prune = (
                not stop_pruning
                and epoch >= start_epoch
                and (epoch - start_epoch) % prune_every == 0
                and batch_idx == 0
            )

            if should_prune:
                prune_step_index = batch_idx
                print(f"\n[DEBUG] >>> PRUNING @ Epoch {epoch}")

                loss.backward(retain_graph=True)

                pdt_engine.step_pruning(
                    loss=loss,
                    current_epoch=epoch,
                    total_epochs=total_epochs
                )

                pdt_engine.apply_mask_to_weights(optimizer=optimizer)

                current_sp = pdt_engine.get_current_sparsity()
                eff = pdt_engine.get_model_efficiency()

                # -------------------------
                # Scientific metrics 출력
                # -------------------------
                print(f"\n[Scientific Metrics - Epoch {epoch}]")
                print(f" 🟢 Model Size: {eff['orig_mb']:.2f} MB -> {eff['curr_mb']:.2f} MB")
                print(f" 🔵 Sparsity: {current_sp:.2f} %")
                print(f" 🟡 Speedup: {eff['speedup']:.2f}x")

                # -------------------------
                # Checkpoint 저장
                # -------------------------
                ckpt_name = f"{config['model']['name']}_{strategy_type}_ep{epoch}_sp{current_sp:.2f}.pth"
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'epoch': epoch,
                        'sparsity': current_sp,
                        'metrics': eff
                    },
                    os.path.join(checkpoint_dir, ckpt_name)
                )
                print(f"💾 Checkpoint saved: {ckpt_name}")

                # -------------------------
                # 안전 제한
                # -------------------------
                # if current_sp >= 45:
                #     print("🛑 Sparsity limit reached. Stop further pruning.")
                #     stop_pruning = True

                torch.cuda.empty_cache()

            else:
                loss.backward()

            pdt_engine.update_ema_and_mask_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            pdt_engine.apply_mask_to_weights()

            # 🔥 step-level peak 기록
            torch.cuda.synchronize()
            step_peak = torch.cuda.max_memory_allocated(device) / (1024**2)
            step_mem_trace.append(step_peak)

            total_loss += loss.item()

        # -------------------------
        # Epoch 종료
        # -------------------------
        val_acc = evaluate(model, val_loader, device)
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)

        print(
            f"Epoch {epoch}/{total_epochs} | "       
            f"Loss: {total_loss/len(train_loader):.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Peak VRAM: {peak_vram:.2f} MB"
        )
        # 🔥 pruning 전후 20 step 중앙값 계산
        if prune_step_index is not None and \
        prune_step_index >= 20 and \
        prune_step_index + 20 < len(step_mem_trace):

            import numpy as np

            pre_median  = np.median(step_mem_trace[prune_step_index-20:prune_step_index])
            prune_peak  = step_mem_trace[prune_step_index]
            post_median = np.median(step_mem_trace[prune_step_index+1:prune_step_index+21])

            print("\n🔥 Pruning Stability Analysis")
            print(f"   Pre Median : {pre_median:.2f} MB")
            print(f"   Prune Peak : {prune_peak:.2f} MB")
            print(f"   Post Median: {post_median:.2f} MB")


        # -------------------------
        # 🔥 매 Epoch 체크포인트 저장
        # -------------------------
        epoch_ckpt_name = f"{config['model']['name']}_{strategy_type}_ep{epoch}.pth"

        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'peak_vram': peak_vram,
                'sparsity': pdt_engine.get_current_sparsity()
            },
            os.path.join(checkpoint_dir, epoch_ckpt_name)
        )

        print(f"💾 Epoch Checkpoint Saved: {epoch_ckpt_name}")

        history_data.append({
            "epoch": epoch,
            "val_acc": val_acc,
            "peak_vram": peak_vram
        })

    # ============================================================
    # ===================== FINAL SAVE ============================
    # ============================================================

    final_path = os.path.join(checkpoint_dir, f"{config['model']['name']}_{strategy_type}_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\n🏁 Final Model Saved: {final_path}")

    # ============================================================
    # ================= PHYSICAL COMPRESSION =====================
    # ============================================================

    print("\n================ FINAL PHYSICAL COMPRESSION ================\n")

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    example_inputs = torch.randn(
        1, 3,
        config['model'].get('input_size', 32),
        config['model'].get('input_size', 32)
    ).to(device)

    DG = tp.DependencyGraph().build(model, example_inputs=example_inputs)

    for module in model.modules():

        # ---------------- Conv ----------------
        if isinstance(module, torch.nn.Conv2d):

            # 🔥 Depthwise 보호 (MobileNet 대응)
            if module.groups == module.in_channels and module.in_channels == module.out_channels:
                pruning_fn = tp.prune_conv_in_channels
            else:
                pruning_fn = tp.prune_conv_out_channels

            if hasattr(module, "mask"):
                mask = module.mask.detach().cpu()
                idxs = torch.nonzero(mask == 0).squeeze().tolist()

                if isinstance(idxs, int):
                    idxs = [idxs]

                if len(idxs) == 0:
                    continue

                if len(idxs) >= module.out_channels:
                    continue

                pruning_group = DG.get_pruning_group(
                    module,
                    pruning_fn,
                    idxs
                )
                pruning_group.prune()

        # ---------------- Linear ----------------
        elif isinstance(module, torch.nn.Linear):

            if hasattr(module, "mask"):
                mask = module.mask.detach().cpu()
                idxs = torch.nonzero(mask == 0).squeeze().tolist()

                if isinstance(idxs, int):
                    idxs = [idxs]

                if len(idxs) == 0:
                    continue

                if len(idxs) >= module.out_features:
                    continue

                pruning_group = DG.get_pruning_group(
                    module,
                    tp.prune_linear_out_channels,
                    idxs
                )
                pruning_group.prune()

    compressed_path = os.path.join(
        checkpoint_dir,
        f"{config['model']['name']}_{strategy_type}_FINAL_COMPRESSED.pth"
    )

    torch.save(model.state_dict(), compressed_path)
    print(f"✅ Physically Compressed Model Saved: {compressed_path}")
    print("=============================================================\n")



# def execute_pdt_experiment(model, config, train_loader, val_loader, test_loader, device, topology_groups, args):

#     checkpoint_dir = config.get('save_dir', './exp/checkpoints')
#     os.makedirs(checkpoint_dir, exist_ok=True)

#     strategy_type = getattr(args, 'strategy', 'pdt').lower()

#     # -------------------------
#     # Pruner 생성
#     # -------------------------
#     pdt_engine = PDTPruner(model, config, args, topology_groups=topology_groups)
#     print("[System] Initializing PDTPruner (Stable Version)")

#     optimizer = optim.SGD(
#         model.parameters(),
#         lr=config['model'].get('base_lr', 0.01),
#         momentum=config['model'].get('momentum', 0.9),
#         weight_decay=config['model'].get('weight_decay', 1e-4)
#     )
#     criterion = nn.CrossEntropyLoss()

#     total_epochs = args.epochs
#     prune_every = args.prune_every
#     start_epoch = args.start_epoch

#     print(f"\n>>> Strategy: {strategy_type.upper()} | Total Epochs: {total_epochs}")
#     print(f">>> Pruning starts at Epoch {start_epoch}, every {prune_every} epochs.")

#     stop_pruning = False   # 🔥 중요

#     for epoch in range(1, total_epochs + 1):

#         model.train()
#         total_loss = 0.0
#         torch.cuda.reset_peak_memory_stats(device)

#         for batch_idx, (x, y) in enumerate(train_loader):
#             x, y = x.to(device), y.to(device)
#             optimizer.zero_grad()

#             output = model(x)
#             loss = criterion(output, y)

#             should_prune = (
#                 not stop_pruning
#                 and epoch >= start_epoch
#                 and (epoch - start_epoch) % prune_every == 0
#                 and batch_idx == 0
#             )

#             if should_prune:
#                 print(f"\n[DEBUG] >>> PRUNING @ Epoch {epoch}")

#                 loss.backward(retain_graph=True)

#                 pdt_engine.step_pruning(
#                     loss=loss,
#                     current_epoch=epoch,
#                     total_epochs=total_epochs
#                 )
#                 pdt_engine.apply_mask_to_weights(optimizer=optimizer)

#                 current_sp = pdt_engine.get_current_sparsity()

#                 # 🔥 무조건 즉시 저장
#                 ckpt_name = f"{config['model']['name']}_{strategy_type}_ep{epoch}_sp{current_sp:.2f}.pth"
#                 torch.save(
#                     {
#                         'model_state_dict': model.state_dict(),
#                         'epoch': epoch,
#                         'sparsity': current_sp
#                     },
#                     os.path.join(checkpoint_dir, ckpt_name)
#                 )
#                 print(f"💾 Checkpoint saved: {ckpt_name}")

#                 # 🔴 45% 이상이면 멈추기 (망가짐 방지)
#                 if current_sp >= 45:
#                     print("🛑 Sparsity limit reached. Stop further pruning.")
#                     stop_pruning = True

#                 torch.cuda.empty_cache()

#             else:
#                 loss.backward()

#             pdt_engine.update_ema_and_mask_grad()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             pdt_engine.apply_mask_to_weights()

#             total_loss += loss.item()

#         val_acc = evaluate(model, val_loader, device)
#         peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)

#         print(
#             f"Epoch {epoch}/{total_epochs} | "
#             f"Loss: {total_loss/len(train_loader):.4f} | "
#             f"Val Acc: {val_acc:.2f}% | "
#             f"Peak VRAM: {peak_vram:.2f} MB"
#         )

#     # -------------------------
#     # Final 저장
#     # -------------------------
#     final_path = os.path.join(checkpoint_dir, f"{config['model']['name']}_{strategy_type}_final.pth")
#     torch.save(model.state_dict(), final_path)
#     print(f"🏁 Final Model Saved: {final_path}")



# def execute_pdt_experiment(model, config, train_loader, val_loader, test_loader, device, topology_groups, args):

#     import os, json
#     import torch
#     import torch.nn as nn
#     import torch.optim as optim
#     import torch_pruning as tp

#     # ----------------------------
#     # [0] 저장 경로
#     # ----------------------------
#     checkpoint_dir = config.get('save_dir', './exp/checkpoints')
#     os.makedirs(checkpoint_dir, exist_ok=True)

#     strategy_type = getattr(args, 'strategy', 'pdt').lower()
#     history_file = os.path.join(
#         checkpoint_dir,
#         f"{config['model']['name']}_{strategy_type}_history.json"
#     )
#     history_data = []

#     # ----------------------------
#     # [1] CLI override
#     # ----------------------------
#     strat_cfg = config.get('strategy', {})

#     if args.group_selection_ratio is not None:
#         strat_cfg['group_selection_ratio'] = args.group_selection_ratio
#     if args.channel_keep_ratio is not None:
#         strat_cfg['channel_keep_ratio'] = args.channel_keep_ratio
#     if args.min_survival_ratio is not None:
#         strat_cfg['min_survival_ratio'] = args.min_survival_ratio

#     config['strategy'] = strat_cfg

#     # ----------------------------
#     # [2] Pruner 초기화
#     # ----------------------------
#     pdt_engine = PDTPruner(model, config, args, topology_groups=topology_groups)
#     print("[System] Initializing PDTPruner (Stable Version)")

#     # ----------------------------
#     # [3] 학습 설정
#     # ----------------------------
#     model_cfg = config.get('model', {})

#     optimizer = optim.SGD(
#         model.parameters(),
#         lr=model_cfg.get('base_lr', 0.01),
#         momentum=model_cfg.get('momentum', 0.9),
#         weight_decay=model_cfg.get('weight_decay', 1e-4)
#     )

#     criterion = nn.CrossEntropyLoss()

#     total_epochs = args.epochs
#     prune_every = args.prune_every
#     start_epoch = args.start_epoch

#     pdt_engine.total_epochs = total_epochs

#     print(f"\n>>> Strategy: {strategy_type.upper()} | Total Epochs: {total_epochs}")
#     print(f">>> Pruning starts at Epoch {start_epoch}, every {prune_every} epochs.")

#     stop_pruning = False

#     # ----------------------------
#     # [4] 학습 루프
#     # ----------------------------
#     stop_pruning = False
#     for epoch in range(1, total_epochs + 1):

#         model.train()
#         total_loss = 0.0
#         torch.cuda.reset_peak_memory_stats(device)

#         for batch_idx, (x, y) in enumerate(train_loader):

#             x, y = x.to(device), y.to(device)
#             optimizer.zero_grad()

#             output = model(x)
#             loss = criterion(output, y)

#             is_prune_step = (
#                 epoch >= start_epoch
#                 and (epoch - start_epoch) % prune_every == 0
#                 and batch_idx == 0
#                 and not stop_pruning
#             )

#             if is_prune_step:

#                 loss.backward(retain_graph=True)

#                 print(f"\n[DEBUG] >>> PRUNING @ Epoch {epoch}")

#                 pdt_engine.step_pruning(
#                     loss=loss,
#                     current_epoch=epoch,
#                     total_epochs=total_epochs
#                 )

#                 # current_sp = pdt_engine.get_current_sparsity()

#                 # # 🔴 과도한 프루닝 방지
#                 # if current_sparsity >= 50:
#                 #     print("🛑 Sparsity limit reached. Stop further pruning.")
#                 #     stop_pruning = True

#                 pdt_engine.apply_mask_to_weights(optimizer=optimizer)
#                 torch.cuda.empty_cache()

#                 current_sp = pdt_engine.get_current_sparsity()

#                 if current_sp >= 50:
#                     print("🛑 Sparsity limit reached. Stop further pruning.")
#                     stop_pruning = True

                


#             else:
#                 loss.backward()

#             pdt_engine.update_ema_and_mask_grad()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             pdt_engine.apply_mask_to_weights()

#             total_loss += loss.item()

#         # ---- validation ----
#         val_acc = evaluate(model, val_loader, device)
#         peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)

#         print(
#             f"Epoch {epoch}/{total_epochs} | "
#             f"Loss: {total_loss/len(train_loader):.4f} | "
#             f"Val Acc: {val_acc:.2f}% | "
#             f"Peak VRAM: {peak_vram:.2f} MB"
#         )

#         # ---- checkpoint 저장 ----
#         if is_prune_step:

#             eff = pdt_engine.get_model_efficiency()

#             epoch_stats = {
#                 'epoch': epoch,
#                 'val_acc': val_acc,
#                 'sparsity': current_sp,
#                 'peak_vram_mb': peak_vram,
#                 'model_size_mb': eff['curr_mb'],
#                 'speedup': eff['speedup']
#             }

#             history_data.append(epoch_stats)

#             with open(history_file, 'w') as f:
#                 json.dump({'history': history_data}, f, indent=4)

#             ckpt_name = (
#                 f"{model_cfg['name']}_{strategy_type}_"
#                 f"ep{epoch}_sp{current_sp:.1f}.pth"
#             )

#             torch.save(
#                 {
#                     'model_state_dict': model.state_dict(),
#                     'sparsity': current_sp
#                 },
#                 os.path.join(checkpoint_dir, ckpt_name)
#             )

#             print(f"💾 Checkpoint saved: {ckpt_name}")

#     # ----------------------------
#     # [5] Final 저장
#     # ----------------------------
#     final_path = os.path.join(
#         checkpoint_dir,
#         f"{model_cfg['name']}_{strategy_type}_final.pth"
#     )

#     torch.save(model.state_dict(), final_path)
#     print(f"🏁 Final Model Saved: {final_path}")

#     # ----------------------------
#     # [6] Physical Compression
#     # ----------------------------
#     print("\n================ FINAL PHYSICAL COMPRESSION ================\n")

#     torch.cuda.empty_cache()
#     torch.cuda.synchronize()

#     mem_before = torch.cuda.memory_allocated(device) / (1024**2)

#     input_size = 32  # CIFAR-100
#     example_inputs = torch.randn(1, 3, input_size, input_size).to(device)

#     DG = tp.DependencyGraph().build_dependency(
#         model,
#         example_inputs=example_inputs
#     )

#     for m in model.modules():

#         if not hasattr(m, "mask"):
#             continue

#         # Conv
#         if isinstance(m, nn.Conv2d):

#             if m.groups == m.in_channels:
#                 continue

#             prune_idx = torch.where(m.mask == 0)[0].tolist()

#             if len(prune_idx) == 0:
#                 continue

#             if len(prune_idx) >= m.out_channels:
#                 continue

#             pruning_group = DG.get_pruning_group(
#                 m,
#                 tp.prune_conv_out_channels,
#                 prune_idx
#             )
#             pruning_group.prune()

#         # Linear
#         elif isinstance(m, nn.Linear):

#             prune_idx = torch.where(m.mask == 0)[0].tolist()

#             if len(prune_idx) == 0:
#                 continue

#             if len(prune_idx) >= m.out_features:
#                 continue

#             pruning_group = DG.get_pruning_group(
#                 m,
#                 tp.prune_linear_out_channels,
#                 prune_idx
#             )
#             pruning_group.prune()

#     torch.cuda.synchronize()

#     mem_after = torch.cuda.memory_allocated(device) / (1024**2)

#     print(f"Before: {mem_before:.2f} MB")
#     print(f"After : {mem_after:.2f} MB")
#     print(
#         f"Saved : {mem_before - mem_after:.2f} MB "
#         f"({((mem_before - mem_after)/mem_before)*100:.1f}% reduction)"
#     )

#     compressed_path = os.path.join(
#         checkpoint_dir,
#         f"{model_cfg['name']}_{strategy_type}_FINAL_COMPRESSED.pth"
#     )

#     torch.save(model.state_dict(), compressed_path)
#     print(f"✅ Physically Compressed Model Saved: {compressed_path}\n")

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