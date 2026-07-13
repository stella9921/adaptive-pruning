import torch.fx as fx
from torch.profiler import profile, record_function, ProfilerActivity
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch_pruning as tp
import sys
import time
import json
from contextlib import nullcontext

from src.pruning.topology_manager import get_model_topology
from src.pruning.optimizer import lagrangian_optimization
from src.utils.config_loader import load_config
from src.utils.measure import (
    collect_epoch_metrics,
    measure_module_memory,
    measure_pruning_structure,
    save_epoch_metrics,
    save_pruning_tables,
    save_run_metadata,
    peak_memory_mb,
    reset_peak_memory,
    synchronize_device,
)
from src.utils.experiment import prepare_experiment, TeeLogger
from src.utils.checkpoint import (
    experiment_signature,
    load_model_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)
from src.utils.visualization import save_history_plots, save_pruning_plots
from src.utils.reproducibility import set_reproducibility
from src.utils.training import build_scheduler
from src.data.dataloader import get_dataloaders
from src.models import get_model, get_prune_fn, is_transformer_model
from src.pruning.sensitivity import maybe_load_or_compute_sensitivity
from src.pruning.pat_strategies import PATPruner
from src.pruning.pdt_strategies import PDTPruner,HAPPruner,SNOWSPruner, ATOPruner, STPruner,DFPCPruner,TPPPruner,ViTPDTPruner
from src.pruning.registry import annotate_pruner_config

def analyze_topology_and_profiling(model, device, config, tag="Before Pruning"):
    """Analyze topology and optional PyTorch resource profile."""
    print(f"\n=== [Stage 1 & 4] {tag} Analysis ===")
    
    model_config = config.get('model', {})
    input_size = model_config.get('input_size', 224)
    if "vit" in model_config.get('name', '').lower():
        input_size = 224
    inputs = torch.randn(1, 3, input_size, input_size).to(device)

    # [Stage 1] Topology Status (FX)
    try:
        traced = fx.symbolic_trace(model)
        print(f"[*] Topology Status: {len(list(traced.graph.nodes))} nodes currently in graph.")
    except Exception as e:
        print(f"[*] FX Parsing Warning: {e}")

    if not config.get('profiling', {}).get('pytorch', False):
        print("[*] Detailed PyTorch profiling skipped (use --profile_pytorch).")
        return None

    # [Stage 4] Resource Profiling
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    with torch.no_grad():
        with profile(activities=activities,
                     with_flops=True, profile_memory=True) as prof:
            with record_function("model_inference"):
                model(inputs)
    
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    return prof

# Raw CLI arguments
print("\n" + "="*50, flush=True)
print(f"[System] Raw Args -> {sys.argv}", flush=True)
print("="*50 + "\n", flush=True)
def main():
    config, args = load_config()
    pruner_name, pruner_details = annotate_pruner_config(config)
    seed = args.seed if args.seed is not None else config.get('seed', 42)
    config['reproducibility'] = set_reproducibility(
        seed=seed,
        deterministic=(args.deterministic or config.get('deterministic', False)),
    )
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.smoke_test:
        config['batch_size'] = 8
        config['model']['epochs'] = 1
        config['strategy'].update({
            'start_epoch': 1,
            'prune_every': 1,
            'hessian_iter': 1,
            'k_horizon': 1,
            'group_selection_ratio': 0.5,
        })
    
    print(f"[Config] batch_size={config['batch_size']}", flush=True)
    if is_transformer_model(config['model']['name']):
        config['model']['input_size'] = 224
        print(">>> [System] ViT detected. Input size forced to 224 for dataloader.")    

 
    log_filename = prepare_experiment(config, args)
    log_path = os.path.join(config['run_dir'], 'logs', log_filename)

    sys.stdout = TeeLogger(log_path)
    print(
        f"[Reproducibility] seed={config['reproducibility']['seed']} "
        f"deterministic={config['reproducibility']['deterministic']}"
    )
    print(f"[Logging] started: {log_filename}")
    
    # Explicit CLI options override YAML.
    if hasattr(args, 'group_selection_ratio') and args.group_selection_ratio is not None:
        config['strategy']['group_selection_ratio'] = args.group_selection_ratio
    if hasattr(args, 'lambda_h') and args.lambda_h is not None:
        config['strategy']['lambda_h'] = args.lambda_h
    if hasattr(args, 'start_epoch') and args.start_epoch is not None:
        config['strategy']['start_epoch'] = args.start_epoch
    if hasattr(args, 'lr') and args.lr is not None:
        config['model']['base_lr'] = args.lr
    metadata_path = save_run_metadata(config, sys.argv)
    print(f"[Experiment] run_id={config['run_id']}")
    print(f"[Experiment] metadata={metadata_path}")
    print(
        f"[Pruner] name={pruner_name} status={pruner_details['status']} "
        f"criterion={pruner_details['criterion']}"
    )
    if pruner_details.get('limitation'):
        print(f"[Pruner Limitation] {pruner_details['limitation']}")
    if args.profile_pytorch:
        print("[Profiler] PyTorch operation and memory profiling enabled.")
    if args.profile_nvtx:
        print("[Profiler] NVTX ranges enabled for Nsight Systems/Compute.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_cfg = config.get('dataset', {})
    dataset_name = dataset_cfg if isinstance(dataset_cfg, str) else dataset_cfg.get('name', 'cifar100')

    if is_transformer_model(config['model']['name']):
        config['model']['input_size'] = 224
        print(">>> [System] ViT detected. Input size forced to 224 for dataloader.")



    if args.smoke_test:
        input_size = config['model'].get('input_size', 32)
        num_classes = config['model'].get('num_classes', 100)
        generator = torch.Generator().manual_seed(config['reproducibility']['seed'])
        images = torch.randn(16, 3, input_size, input_size, generator=generator)
        labels = torch.randint(0, num_classes, (16,), generator=generator)
        smoke_dataset = TensorDataset(images, labels)
        smoke_loader = DataLoader(smoke_dataset, batch_size=8, shuffle=False)
        loaders = smoke_loader, smoke_loader, smoke_loader
        print("[Smoke Test] Using 16 synthetic samples; dataset download is skipped.")
    else:
        loaders = get_dataloaders(config)
    
    if len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, val_loader = loaders
        test_loader = val_loader 
    
    print(f"Experimental Mode: {config['strategy'].get('method', 'N/A')}")
    print(f"Target Model: {config['model']['name']} | Dataset: {dataset_name}")

    model = get_model(config['model']).to(device)

    is_vit_model = is_transformer_model(config['model']['name'])
    
    if is_vit_model and 'imagenet100' in str(dataset_name):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, 100).to(device)
        print(f">>> [System] ViT head replaced: {in_features} -> 100")

        # Warm up the classifier head before full fine-tuning.
        print(">>> [System] Head warmup started (5 epochs, head only).")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True

        warmup_opt = optim.AdamW(model.head.parameters(), lr=0.001)
        warmup_criterion = nn.CrossEntropyLoss()

        for ep in range(5):
            model.train()
            for x, y in train_loader:
                if x.shape[-1] != 224:
                    x = torch.nn.functional.interpolate(
                        x, size=(224, 224), mode='bilinear', align_corners=False)
                x, y = x.to(device), y.to(device)
                warmup_opt.zero_grad()
                warmup_criterion(model(x), y).backward()
                warmup_opt.step()

            val_acc = evaluate(model, val_loader, device)
            print(f"  [Head Warmup] Epoch {ep+1}/5 | Val Acc: {val_acc:.2f}%")

        for param in model.parameters():
            param.requires_grad = True
        print(">>> [System] Head warmup complete. Starting full fine-tuning.")
    
    if (
        'imagenet' in str(dataset_name)
        and not is_vit_model
        and hasattr(model, 'fc')
        and config['model'].get('num_classes') == 100
        and config['model'].get('pretrained', False)
    ):
        print(">>> [System] Transferring Pretrained Weights to FC Layer (1000 -> 100)...")
        temp_config = config['model'].copy()
        temp_config['num_classes'] = 1000
        full_model = get_model(temp_config).to(device)

        with torch.no_grad():
            model.fc.weight.copy_(full_model.fc.weight[:100])
            model.fc.bias.copy_(full_model.fc.bias[:100])
        del full_model
        print("FC weight transfer successful.")

    # [Stage 1] Topology Parsing
    topology_groups = get_model_topology(model)

    strategy_method = config['strategy'].get('method', 'pdt').lower()

    if strategy_method == 'pat':
        execute_pat_experiment(model, config, train_loader, val_loader, test_loader, device, topology_groups,args)
    elif strategy_method == 'pdt' or strategy_method.startswith('pdt-'):
        execute_pdt_experiment(
            model, 
            config, 
            train_loader, 
            val_loader, 
            test_loader, 
            device, 
            topology_groups,
            args
        )
def execute_pdt_experiment(model, config, train_loader, val_loader, test_loader, device, topology_groups, args):

    import torch_pruning as tp
    import json

    is_vit_model = is_transformer_model(config['model']['name'])
    if 'vit' in config['model']['name'].lower() or \
       'deit' in config['model']['name'].lower():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print(">>> [System] Flash Attention disabled for Hessian computation.")


    checkpoint_dir = config.get('checkpoint_dir', config.get('save_dir', './exp/checkpoints'))
    os.makedirs(checkpoint_dir, exist_ok=True)

    strategy_type = config['strategy'].get('preset', 'pdt').lower()

    model_name = config['model']['name'].lower()
    if is_transformer_model(model_name):
        pdt_engine = ViTPDTPruner(model, config, args, topology_groups=topology_groups)
        print(f"[System] Initializing ViTPDTPruner for {model_name}")
    else:
        pruner_name = config['strategy'].get('pruner', 'mcprune').lower()
        pruner_classes = {
            'mcprune': PDTPruner,
            'hap': HAPPruner,
            'snows': SNOWSPruner,
            'ato': ATOPruner,
            'st': STPruner,
            'dfpc': DFPCPruner,
            'tpp': TPPPruner,
        }
        if pruner_name not in pruner_classes:
            raise ValueError(f"Unknown PDT-family pruner: {pruner_name}")
        pruner_class = pruner_classes[pruner_name]
        pdt_engine = pruner_class(
            model, config, args, topology_groups=topology_groups
        )
        print(f"[System] Initializing {pruner_class.__name__} for {model_name}")
    if is_vit_model:
        optimizer = optim.AdamW(
        model.parameters(),
        lr=config['model'].get('base_lr', 1e-4),
        weight_decay=config['model'].get('weight_decay', 0.05)
    )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['model'].get('base_lr', 0.01),
            momentum=config['model'].get('momentum', 0.9),
            weight_decay=config['model'].get('weight_decay', 1e-4)
        )
    criterion = nn.CrossEntropyLoss()

    model_cfg = config.get('model', {})
    strat_cfg = config.get('strategy', {})
    total_epochs = args.epochs if args.epochs is not None else model_cfg.get('epochs', config.get('epochs', 300))
    prune_every = args.prune_every if args.prune_every is not None else strat_cfg.get('prune_every', 20)
    start_epoch = args.start_epoch if args.start_epoch is not None else strat_cfg.get('start_epoch', 1)
    scheduler = build_scheduler(optimizer, model_cfg, total_epochs)
    pdt_engine.total_epochs = total_epochs
    if args.smoke_test:
        pruning_epochs = list(range(start_epoch, total_epochs + 1, prune_every))
    else:
        pruning_epochs = list(range(start_epoch, total_epochs, prune_every))
    if not pruning_epochs:
        raise ValueError(
            "No pruning epochs remain after reserving final recovery epoch: "
            f"start_epoch={start_epoch}, epochs={total_epochs}, "
            f"prune_every={prune_every}"
        )
    pruning_step_by_epoch = {
        pruning_epoch: step_index + 1
        for step_index, pruning_epoch in enumerate(pruning_epochs)
    }

    print(f"\n>>> Strategy: {strategy_type.upper()} | Total Epochs: {total_epochs}")
    print(
        f">>> Target pruning ratio: {strat_cfg['pruning_ratio']:.1%} "
        f"| Target keep ratio: {1.0 - strat_cfg['pruning_ratio']:.1%}"
    )
    print(f">>> Pruning starts at Epoch {start_epoch}, every {prune_every} epochs.")
    print(f">>> Scheduled pruning epochs: {pruning_epochs}")
    if args.smoke_test:
        print(">>> Final recovery window: disabled for smoke test.")
    else:
        print(
            f">>> Final recovery window: epochs {pruning_epochs[-1] + 1}"
            f"-{total_epochs}"
        )
    print(
        f">>> LR scheduler: "
        f"{scheduler.__class__.__name__ if scheduler is not None else 'disabled'}"
    )

    stop_pruning = False
    history_data = []
    first_epoch = 1
    best_val_acc = float('-inf')
    best_checkpoint_path = None
    if args.resume:
        resume_state = load_training_checkpoint(
            args.resume, model, optimizer, scheduler, device,
            expected_signature=experiment_signature(config),
        )
        first_epoch = int(resume_state['epoch']) + 1
        if first_epoch > total_epochs:
            raise ValueError(
                f"resume checkpoint already reached epoch {first_epoch - 1}, "
                f"but configured epochs={total_epochs}"
            )
        history_data = list(resume_state.get('history', []))
        best_val_acc = float(resume_state.get('best_val_acc', best_val_acc))
        best_checkpoint_path = resume_state.get('best_checkpoint_path')
        loader_state = resume_state.get('train_loader_generator_state')
        if loader_state is not None and train_loader.generator is not None:
            train_loader.generator.set_state(loader_state.cpu())
        print(
            f"[Checkpoint] resumed={args.resume} "
            f"last_epoch={first_epoch - 1} next_epoch={first_epoch}"
        )
    debug_stop_after_first_prune = (
        str(config.get('strategy', {}).get('debug_stop_after_first_prune', '')).lower() in ('1', 'true', 'yes')
        or os.getenv('MCPRUNE_STOP_AFTER_FIRST_PRUNE') == '1'
    )

    for epoch in range(first_epoch, total_epochs + 1):
        model.train()
        total_loss = 0.0
        reset_peak_memory(device)

        # Step-level memory trace
        step_mem_trace = []
        prune_step_index = None

        for batch_idx, (x, y) in enumerate(train_loader):
            if x.shape[-1] != 224 and hasattr(model, 'patch_embed'):
                x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # Pruning step
            output = model(x)
            loss = criterion(output, y)
            
            # Detach scalar loss before graph cleanup
            current_loss_val = loss.item()

            should_prune = (
                not stop_pruning
                and epoch in pruning_step_by_epoch
                and batch_idx == 0
            )

            if should_prune:
                prune_step_index = batch_idx
                pruning_step = pruning_step_by_epoch[epoch]
                pdt_engine.scheduled_pruning_progress = (
                    pruning_step / len(pruning_epochs)
                )
                print(f"\n[DEBUG] >>> PRUNING @ Epoch {epoch}")
                print(
                    f"[Pruning Schedule] step={pruning_step}/{len(pruning_epochs)} "
                    f"progress={pdt_engine.scheduled_pruning_progress:.4f}"
                )

                # Retain graph for Hessian computation
                loss.backward(retain_graph=True)
                pdt_engine.update_ema_and_mask_grad()

                nvtx_context = (
                    torch.cuda.nvtx.range("MCPrune/PruningStep")
                    if args.profile_nvtx and torch.cuda.is_available()
                    else nullcontext()
                )
                if args.profile_pytorch:
                    activities = [ProfilerActivity.CPU]
                    if torch.cuda.is_available():
                        activities.append(ProfilerActivity.CUDA)
                    with profile(
                        activities=activities,
                        record_shapes=True,
                        profile_memory=True,
                        with_stack=False,
                    ) as prune_profiler:
                        with nvtx_context:
                            pdt_engine.step_pruning(
                                loss=loss,
                                current_epoch=epoch,
                                total_epochs=total_epochs,
                            )
                    profile_prefix = (
                        f"{config['run_id']}__pytorch-profile__epoch-{epoch:03d}"
                    )
                    profile_dir = os.path.join(config['run_dir'], 'profiles')
                    trace_path = os.path.join(profile_dir, f"{profile_prefix}.json")
                    summary_path = os.path.join(profile_dir, f"{profile_prefix}.txt")
                    prune_profiler.export_chrome_trace(trace_path)
                    with open(summary_path, 'w', encoding='utf-8') as stream:
                        stream.write(
                            prune_profiler.key_averages().table(
                                sort_by=(
                                    'self_cuda_time_total'
                                    if torch.cuda.is_available()
                                    else 'self_cpu_time_total'
                                ),
                                row_limit=100,
                            )
                        )
                    print(f"[Profiler] PyTorch trace saved: {trace_path}")
                else:
                    with nvtx_context:
                        pdt_engine.step_pruning(
                            loss=loss,
                            current_epoch=epoch,
                            total_epochs=total_epochs,
                        )

                pdt_engine.apply_mask_to_weights(optimizer=optimizer)
                snapshot = measure_pruning_structure(
                    model,
                    topology_groups,
                    getattr(pdt_engine, 'last_group_selection', []),
                )
                snapshot['module_memory'] = measure_module_memory(model, x)
                snapshot_paths = save_pruning_tables(
                    snapshot, config['run_dir'], config['run_id'], epoch
                )
                snapshot_paths += save_pruning_plots(
                    snapshot, config['run_dir'], config['run_id'], epoch
                )
                if snapshot_paths:
                    print(
                        f"[Visualization] saved {len(snapshot_paths)} files "
                        f"under {config['run_dir']}"
                    )
                eff = pdt_engine.get_model_efficiency()
                current_sp = eff['sparsity']
                channel_sp = pdt_engine.get_current_sparsity()
                target_sp = strat_cfg['pruning_ratio'] * 100.0
                sparsity_gap = current_sp - target_sp
                print(
                    f"[Ratio Check] basis=params target={target_sp:.2f}% "
                    f"actual={current_sp:.2f}% gap={sparsity_gap:+.2f}%p "
                    f"channel_sparsity={channel_sp:.2f}%"
                )
                if debug_stop_after_first_prune:
                    print("[DEBUG] Stop after first pruning step requested. Exiting PDT experiment early.")
                    return

                # Release the pruning graph
                import gc
                del loss  # 1. loss 객체 삭제
                gc.collect() # 2. 가비지 컬렉션
                torch.cuda.empty_cache() # 3. GPU 메모리 반환
                print("Post-pruning memory cleared and graph released.")

                print(f"\n[Scientific Metrics - Epoch {epoch}]")
                print(f"Model Size: {eff['orig_mb']:.2f} MB -> {eff['curr_mb']:.2f} MB")
                print(f"Sparsity: {current_sp:.2f} %")
                print(f"Speedup: {eff['speedup']:.2f}x")

                ckpt_name = (
                    f"{config['run_id']}__pruning-snapshot__epoch-{epoch:03d}"
                    f"__sparsity-{current_sp:.2f}.pth"
                )
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'sparsity': current_sp,
                    'target_pruning_ratio': strat_cfg['pruning_ratio'],
                    'sparsity_gap_percent_point': sparsity_gap,
                    'metrics': eff
                }, os.path.join(checkpoint_dir, ckpt_name))
                print(f"Checkpoint saved: {ckpt_name}")

            else:
                # Standard training step
                t_step_start = time.time()
                loss.backward()
                pdt_engine.update_ema_and_mask_grad()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                pdt_engine.apply_mask_to_weights(optimizer=optimizer)
                t_step_end = time.time()
                if batch_idx % 100 == 0:
                    print(f" [Normal step time] batch {batch_idx}: {t_step_end - t_step_start:.4f}s")
                total_loss += current_loss_val
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            pdt_engine.apply_mask_to_weights(optimizer=optimizer)

            # Step-level peak memory
            synchronize_device(device)
            step_peak = peak_memory_mb(device)
            step_mem_trace.append(step_peak)

            # Accumulate batch loss
            total_loss += current_loss_val 
            # End training step

        # Epoch 종료
        val_acc = evaluate(model, val_loader, device)
        peak_vram = peak_memory_mb(device)

        print(
            f"Epoch {epoch}/{total_epochs} | "       
            f"Loss: {total_loss/len(train_loader):.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Peak VRAM: {peak_vram:.2f} MB | "
            f"LR: {optimizer.param_groups[0]['lr']:.6g}"
        )
        # Median memory around pruning
        if prune_step_index is not None and \
        prune_step_index >= 20 and \
        prune_step_index + 20 < len(step_mem_trace):

            import numpy as np

            pre_median  = np.median(step_mem_trace[prune_step_index-20:prune_step_index])
            prune_peak  = step_mem_trace[prune_step_index]
            post_median = np.median(step_mem_trace[prune_step_index+1:prune_step_index+21])

            print("\nPruning Stability Analysis")
            print(f"   Pre Median : {pre_median:.2f} MB")
            print(f"   Prune Peak : {prune_peak:.2f} MB")
            print(f"   Post Median: {post_median:.2f} MB")


        epoch_metrics = collect_epoch_metrics(model, optimizer, x)
        print(
            "[Parameter Metrics] "
            f"total={epoch_metrics['total_params']:,} "
            f"prunable={epoch_metrics['prunable_weight_params']:,} "
            f"prunable_sparsity={epoch_metrics['parameter_sparsity']:.2%} "
            f"model_reduction={epoch_metrics['model_parameter_reduction']:.2%}"
        )
        history_data.append({
            'epoch': epoch,
            'train_loss': total_loss / len(train_loader),
            'val_accuracy': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'target_pruning_ratio': strat_cfg['pruning_ratio'],
            **epoch_metrics,
        })
        save_epoch_metrics(history_data, config['run_dir'], config['run_id'])
        save_history_plots(history_data, config['run_dir'], config['run_id'])
        if scheduler is not None:
            scheduler.step()

        is_new_best = val_acc > best_val_acc
        if is_new_best:
            best_val_acc = val_acc
            best_checkpoint_path = os.path.join(
                checkpoint_dir, f"{config['run_id']}__checkpoint__best.pth"
            )

        checkpoint_extra = {
            'best_val_acc': best_val_acc,
            'best_checkpoint_path': best_checkpoint_path,
            'target_pruning_ratio': strat_cfg['pruning_ratio'],
            'run_id': config['run_id'],
            'experiment_signature': experiment_signature(config),
            'train_loader_generator_state': (
                train_loader.generator.get_state()
                if train_loader.generator is not None else None
            ),
        }
        if is_new_best:
            save_training_checkpoint(
                best_checkpoint_path, model, optimizer, scheduler, epoch, history_data,
                **checkpoint_extra,
            )
            print(
                f"[Checkpoint] new best val_acc={val_acc:.2f}%: "
                f"{best_checkpoint_path}"
            )
        last_path = os.path.join(
            checkpoint_dir, f"{config['run_id']}__checkpoint__last.pth"
        )
        save_training_checkpoint(
            last_path, model, optimizer, scheduler, epoch, history_data,
            **checkpoint_extra,
        )


    last_test_acc = evaluate(model, test_loader, device)
    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        best_state = load_model_checkpoint(best_checkpoint_path, model, device)
        final_test_acc = evaluate(model, test_loader, device)
        print(
            f"[Final Metrics] Last Test Accuracy: {last_test_acc:.2f}% | "
            f"Best(epoch {best_state['epoch']}) Test Accuracy: {final_test_acc:.2f}%"
        )
    else:
        final_test_acc = last_test_acc
        print(f"[Final Metrics] Test Accuracy: {final_test_acc:.2f}%")
    if history_data:
        history_data[-1]['test_accuracy'] = final_test_acc
        save_epoch_metrics(history_data, config['run_dir'], config['run_id'])

    final_path = os.path.join(
        checkpoint_dir, f"{config['run_id']}__checkpoint__final.pth"
    )
    torch.save(model.state_dict(), final_path)
    if args.smoke_test:
        print(f"[Smoke Test] Completed. Results: {config['run_dir']}")
        return
    print(f"\nFinal model saved: {final_path}")

    print("\n================ FINAL PHYSICAL COMPRESSION ================\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    synchronize_device(device)



    example_inputs = torch.randn(
        1, 3,
        config['model'].get('input_size', 224 if is_vit_model else 32),
        config['model'].get('input_size', 224 if is_vit_model else 32)
    ).to(device)

   
    try:
        DG = tp.DependencyGraph().build_graph(model, example_inputs=example_inputs)
    except AttributeError:
        DG = tp.DependencyGraph()
        DG.build_dependency(model, example_inputs=example_inputs)

    if is_vit_model:
        # ViT pruning by group type
        all_modules = dict(model.named_modules())

        for g in topology_groups:
            if g['type'] == 'ffn':
                # Prune fc1 output rows
                fc1 = all_modules.get(g['names'][0])
                fc2 = all_modules.get(g['names'][1])

                if fc1 is not None and hasattr(fc1, 'mask'):
                    idxs = torch.where(fc1.mask == 0)[0].tolist()
                    if 0 < len(idxs) < fc1.out_features:
                        pruning_group = DG.get_pruning_group(
                            fc1, tp.prune_linear_out_channels, idxs)
                        pruning_group.prune()
                        print(f"[Compressed] {g['names'][0]}: {len(idxs)} neurons removed")

                # DependencyGraph propagates fc1 output pruning to fc2 inputs

            elif g['type'] == 'attn':
                qkv  = all_modules.get(g['names'][0])
                proj = all_modules.get(g['names'][1])

                if qkv is not None and hasattr(qkv, 'mask'):
                    num_heads = g['num_heads']
                    head_dim  = g['head_dim']

                    # Convert head indices to qkv rows
                    dead_heads = torch.where(qkv.mask == 0)[0].tolist()
                    if not dead_heads:
                        continue

                    # Collect Q/K/V rows
                    qkv_idxs = []
                    for h in dead_heads:
                        for offset in range(3):  # Q, K, V
                            start = offset * (num_heads * head_dim) + h * head_dim
                            qkv_idxs.extend(range(start, start + head_dim))

                    if 0 < len(qkv_idxs) < qkv.out_features:
                        pruning_group = DG.get_pruning_group(
                            qkv, tp.prune_linear_out_channels, qkv_idxs)
                        pruning_group.prune()
                        print(f"[Compressed] {g['names'][0]}: {len(dead_heads)} heads removed")

    else:
        # CNN physical pruning
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                if module.groups == module.in_channels == module.out_channels:
                    pruning_fn = tp.prune_conv_in_channels
                else:
                    pruning_fn = tp.prune_conv_out_channels

                if hasattr(module, "mask"):
                    mask = module.mask.detach().cpu()
                    idxs = torch.nonzero(mask == 0).squeeze().tolist()
                    if isinstance(idxs, int): idxs = [idxs]
                    if len(idxs) == 0 or len(idxs) >= module.out_channels:
                        continue
                    DG.get_pruning_group(module, pruning_fn, idxs).prune()

            elif isinstance(module, torch.nn.Linear):
                if hasattr(module, "mask"):
                    mask = module.mask.detach().cpu()
                    idxs = torch.nonzero(mask == 0).squeeze().tolist()
                    if isinstance(idxs, int): idxs = [idxs]
                    if len(idxs) == 0 or len(idxs) >= module.out_features:
                        continue
                    DG.get_pruning_group(
                        module, tp.prune_linear_out_channels, idxs).prune()

    compressed_path = os.path.join(
        checkpoint_dir,
        f"{config['run_id']}__checkpoint__physically-compressed.pth"
    )
    torch.save(model.state_dict(), compressed_path)
    print(f"Physically compressed model saved: {compressed_path}")
    print("=============================================================\n")





def execute_pat_experiment(model, config, train_loader, val_loader, test_loader, device, topology_groups,args):
    
    checkpoint_dir = config.get('checkpoint_dir', config.get('save_dir', './exp/checkpoints'))
    os.makedirs(checkpoint_dir, exist_ok=True)
    base_ckpt_path = os.path.join(checkpoint_dir, f"{config['model']['name']}_base.pth")
    
    if not os.path.exists(base_ckpt_path):
        print(f"\n>>> [PAT Pre-train] Base weights not found. Training for 120 epochs first...")
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        # CIFAR-100 cosine schedule
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, 121):
            # Shared training and evaluation functions
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_acc = evaluate(model, val_loader, device)
            scheduler.step()
            
            if epoch % 10 == 0 or epoch == 1:
                print(f" [Pre-train] Epoch {epoch}/120 | Loss: {train_loss:.4f} | Acc: {val_acc:.2f}%")
        
        # Save the trained baseline
        torch.save(model.state_dict(), base_ckpt_path)
        print(f"Pre-training complete. Saved to {base_ckpt_path}")
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
            if x.shape[-1] != 224 and hasattr(model, 'patch_embed'):
                x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

if __name__ == "__main__":
    main()
