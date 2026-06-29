import csv
import json
import os
import re
import time
from collections import defaultdict

import torch
import torch.nn as nn


def _group_names(group):
    if isinstance(group, dict):
        return list(group.get('names', []))
    return list(group)


def _stage_name(layer_name):
    match = re.match(r'^(layer\d+)', layer_name)
    if match:
        return match.group(1)
    match = re.match(r'^(features|blocks)\.(\d+)', layer_name)
    if match:
        family, index = match.groups()
        start = (int(index) // 4) * 4
        return f"{family}.{start:02d}-{start + 3:02d}"
    return layer_name.split('.', 1)[0]


def _write_csv(path, rows):
    if not rows:
        return None
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def measure_model_resources(model):
    total_params = 0
    remaining_params = 0.0
    for layer in model.modules():
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
            continue
        count = layer.weight.numel()
        total_params += count
        keep_ratio = (
            layer.mask.float().mean().item() if hasattr(layer, 'mask') else 1.0
        )
        remaining_params += count * keep_ratio
    sparsity = 1.0 - remaining_params / total_params if total_params else 0.0
    return {
        'total_params': int(total_params),
        'remaining_params': int(round(remaining_params)),
        'parameter_sparsity': sparsity,
        'original_size_mb': total_params * 4 / 1024**2,
        'remaining_size_mb': remaining_params * 4 / 1024**2,
        'theoretical_speedup': total_params / max(remaining_params, 1.0),
    }


def measure_cuda_memory():
    if not torch.cuda.is_available():
        return {
            'cuda_allocated_mb': 0.0,
            'cuda_reserved_mb': 0.0,
            'cuda_peak_allocated_mb': 0.0,
            'cuda_peak_reserved_mb': 0.0,
        }
    return {
        'cuda_allocated_mb': torch.cuda.memory_allocated() / 1024**2,
        'cuda_reserved_mb': torch.cuda.memory_reserved() / 1024**2,
        'cuda_peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
        'cuda_peak_reserved_mb': torch.cuda.max_memory_reserved() / 1024**2,
    }


def measure_training_state_memory(model, optimizer):
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    gradient_bytes = sum(
        p.grad.numel() * p.grad.element_size()
        for p in model.parameters() if p.grad is not None
    )
    optimizer_bytes = 0
    for state in optimizer.state.values():
        optimizer_bytes += sum(
            value.numel() * value.element_size()
            for value in state.values() if torch.is_tensor(value)
        )
    return {
        'parameter_memory_mb': parameter_bytes / 1024**2,
        'gradient_memory_mb': gradient_bytes / 1024**2,
        'optimizer_memory_mb': optimizer_bytes / 1024**2,
    }


def measure_inference(model, sample, warmup=3, iterations=10):
    was_training = model.training
    model.eval()
    sample = sample[:1]
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        if sample.is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(iterations):
            model(sample)
        if sample.is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    model.train(was_training)
    latency_ms = elapsed / iterations * 1000.0
    return {
        'inference_latency_ms': latency_ms,
        'throughput_images_per_sec': 1000.0 / max(latency_ms, 1e-12),
        'inference_peak_vram_mb': (
            torch.cuda.max_memory_allocated() / 1024**2 if sample.is_cuda else 0.0
        ),
    }


def measure_module_memory(model, sample):
    rows = []
    hooks = []

    def register(name, layer):
        def hook(_, __, output):
            outputs = output if isinstance(output, (tuple, list)) else (output,)
            activation_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in outputs if torch.is_tensor(tensor)
            )
            rows.append({
                'layer': name,
                'stage': _stage_name(name),
                'parameter_memory_mb': (
                    sum(p.numel() * p.element_size() for p in layer.parameters(False))
                    / 1024**2
                ),
                'activation_volume_mb': activation_bytes / 1024**2,
            })
        hooks.append(layer.register_forward_hook(hook))

    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            register(name, layer)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(sample[:1])
    model.train(was_training)
    for hook in hooks:
        hook.remove()
    return rows


def measure_flops(model, sample):
    dense_flops = 0.0
    effective_flops = 0.0
    hooks = []

    def hook(layer, _, output):
        nonlocal dense_flops, effective_flops
        if isinstance(layer, nn.Conv2d):
            kernel_ops = (
                layer.kernel_size[0] * layer.kernel_size[1]
                * layer.in_channels / layer.groups
            )
            layer_flops = output.numel() * kernel_ops * 2.0
        else:
            layer_flops = output.numel() * layer.in_features * 2.0
        keep_ratio = (
            layer.mask.float().mean().item() if hasattr(layer, 'mask') else 1.0
        )
        dense_flops += layer_flops
        effective_flops += layer_flops * keep_ratio

    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            hooks.append(layer.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(sample[:1])
    model.train(was_training)
    for handle in hooks:
        handle.remove()
    return {
        'estimated_dense_flops': dense_flops,
        'estimated_effective_flops': effective_flops,
        'estimated_flops_reduction': (
            1.0 - effective_flops / dense_flops if dense_flops else 0.0
        ),
        'estimated_flops_speedup': dense_flops / max(effective_flops, 1.0),
    }


def measure_pruning_structure(model, topology_groups, selection_records):
    layer_rows = []
    for name, layer in model.named_modules():
        if not isinstance(layer, (nn.Conv2d, nn.Linear)) or not hasattr(layer, 'mask'):
            continue
        total_units = int(layer.mask.numel())
        alive_units = int(layer.mask.sum().item())
        keep_ratio = alive_units / total_units if total_units else 1.0
        param_count = int(layer.weight.numel())
        layer_rows.append({
            'layer': name,
            'stage': _stage_name(name),
            'total_units': total_units,
            'alive_units': alive_units,
            'sparsity_percent': (1.0 - keep_ratio) * 100.0,
            'param_count': param_count,
            'pruned_param_cost': param_count * (1.0 - keep_ratio),
        })

    selection_rows = sorted(
        list(selection_records or []), key=lambda row: int(row['rank'])
    )
    selection_by_id = {int(row['group_id']): row for row in selection_rows}
    modules = dict(model.named_modules())
    group_rows = []
    for group_id, group in enumerate(topology_groups or [], 1):
        names = _group_names(group)
        layers = [
            modules[name] for name in names
            if name in modules and hasattr(modules[name], 'mask')
        ]
        if not layers:
            continue
        total_cost = sum(layer.weight.numel() for layer in layers)
        alive_cost = sum(
            layer.weight.numel() * layer.mask.float().mean().item()
            for layer in layers
        )
        selection = selection_by_id.get(group_id, {})
        group_rows.append({
            'group_id': group_id,
            'layers': ', '.join(names),
            'selected': bool(selection.get('selected', False)),
            'selection_rank': selection.get('rank', ''),
            'grad_ema': selection.get('grad_ema', ''),
            'sparsity_percent': (
                (1.0 - alive_cost / total_cost) * 100.0 if total_cost else 0.0
            ),
            'param_count': int(total_cost),
        })

    totals = defaultdict(float)
    pruned = defaultdict(float)
    for row in layer_rows:
        totals[row['stage']] += row['param_count']
        pruned[row['stage']] += row['pruned_param_cost']
    stage_rows = [
        {
            'stage': stage,
            'param_count': int(totals[stage]),
            'sparsity_percent': pruned[stage] / totals[stage] * 100.0,
        }
        for stage in sorted(totals)
    ]
    return {
        'group_selection': selection_rows,
        'group_sparsity': group_rows,
        'layer_sparsity': layer_rows,
        'stage_sparsity': stage_rows,
    }


def save_pruning_tables(snapshot, run_dir, run_id, epoch):
    metrics_dir = os.path.join(run_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    paths = []
    for name, rows in snapshot.items():
        path = os.path.join(
            metrics_dir, f"{run_id}__{name}__epoch-{int(epoch):03d}.csv"
        )
        saved = _write_csv(path, rows)
        if saved:
            paths.append(saved)
    return paths


def save_epoch_metrics(history, run_dir, run_id):
    metrics_dir = os.path.join(run_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    csv_path = os.path.join(metrics_dir, f"{run_id}__epoch-metrics.csv")
    json_path = os.path.join(metrics_dir, f"{run_id}__epoch-metrics.json")
    _write_csv(csv_path, history)
    with open(json_path, 'w', encoding='utf-8') as stream:
        json.dump(history, stream, indent=2)
    return [csv_path, json_path]


def save_run_metadata(config, argv):
    run_dir = config['run_dir']
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, f"{config['run_id']}__metadata.json")
    with open(path, 'w', encoding='utf-8') as stream:
        json.dump({'command': list(argv), 'config': config}, stream, indent=2)
    return path
