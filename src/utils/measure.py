import csv
import json
import os
import re
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    total_model_params = sum(parameter.numel() for parameter in model.parameters())
    total_model_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    prunable_weight_params = 0
    remaining_prunable_weight_params = 0.0
    pruned_weight_bytes = 0.0
    for layer in model.modules():
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
            continue
        count = layer.weight.numel()
        prunable_weight_params += count
        keep_ratio = (
            layer.mask.float().mean().item() if hasattr(layer, 'mask') else 1.0
        )
        remaining_prunable_weight_params += count * keep_ratio
        pruned_weight_bytes += count * (1.0 - keep_ratio) * layer.weight.element_size()
    pruned_params = prunable_weight_params - remaining_prunable_weight_params
    remaining_model_params = total_model_params - pruned_params
    sparsity = (
        1.0 - remaining_prunable_weight_params / prunable_weight_params
        if prunable_weight_params else 0.0
    )
    return {
        'total_params': int(total_model_params),
        'remaining_params': int(round(remaining_model_params)),
        'prunable_weight_params': int(prunable_weight_params),
        'remaining_prunable_weight_params': int(round(remaining_prunable_weight_params)),
        'parameter_sparsity': sparsity,
        'model_parameter_reduction': (
            1.0 - remaining_model_params / total_model_params
            if total_model_params else 0.0
        ),
        'original_size_mb': total_model_bytes / 1024**2,
        'remaining_size_mb': (total_model_bytes - pruned_weight_bytes) / 1024**2,
        'theoretical_speedup': (
            prunable_weight_params / max(remaining_prunable_weight_params, 1.0)
        ),
    }


def reset_peak_memory(device):
    if torch.cuda.is_available() and torch.device(device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)


def synchronize_device(device):
    if torch.cuda.is_available() and torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def peak_memory_mb(device):
    if torch.cuda.is_available() and torch.device(device).type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / 1024**2
    return 0.0


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


def collect_epoch_metrics(model, optimizer, sample):
    return {
        **measure_model_resources(model),
        **measure_cuda_memory(),
        **measure_training_state_memory(model, optimizer),
        **measure_inference(model, sample),
        **measure_flops(model, sample),
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


def measure_activation_distribution(model, sample, max_samples=1):
    rows = []
    hooks = []

    def register(name, layer):
        def hook(_, __, output):
            outputs = output if isinstance(output, (tuple, list)) else (output,)
            tensors = [
                tensor.detach().float().flatten().cpu()
                for tensor in outputs if torch.is_tensor(tensor)
            ]
            if not tensors:
                return
            values = torch.cat(tensors)
            if values.numel() == 0:
                return
            abs_values = values.abs()
            quantiles = torch.quantile(
                values,
                torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]),
            )
            rows.append({
                'layer': name,
                'stage': _stage_name(name),
                'num_values': int(values.numel()),
                'mean': float(values.mean().item()),
                'std': float(values.std(unbiased=False).item()),
                'min': float(values.min().item()),
                'max': float(values.max().item()),
                'abs_mean': float(abs_values.mean().item()),
                'rms': float(torch.sqrt((values ** 2).mean()).item()),
                'zero_fraction': float((values == 0).float().mean().item()),
                'positive_fraction': float((values > 0).float().mean().item()),
                'q01': float(quantiles[0].item()),
                'q05': float(quantiles[1].item()),
                'q25': float(quantiles[2].item()),
                'q50': float(quantiles[3].item()),
                'q75': float(quantiles[4].item()),
                'q95': float(quantiles[5].item()),
                'q99': float(quantiles[6].item()),
            })
        hooks.append(layer.register_forward_hook(hook))

    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            register(name, layer)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(sample[:max_samples])
    model.train(was_training)
    for hook in hooks:
        hook.remove()
    return rows


def collect_hidden_representations(model, sample, max_samples=32):
    features = {}
    hooks = []

    def flatten_feature(output):
        if not torch.is_tensor(output):
            return None
        feature = output.detach().float()
        if feature.dim() > 2:
            feature = feature.mean(dim=tuple(range(2, feature.dim())))
        elif feature.dim() == 1:
            feature = feature.unsqueeze(0)
        else:
            feature = feature.flatten(start_dim=1)
        return feature.cpu()

    def register(name, layer):
        def hook(_, __, output):
            outputs = output if isinstance(output, (tuple, list)) else (output,)
            tensors = [flatten_feature(tensor) for tensor in outputs]
            tensors = [tensor for tensor in tensors if tensor is not None]
            if tensors:
                features[name] = torch.cat(tensors, dim=1)
        hooks.append(layer.register_forward_hook(hook))

    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            register(name, layer)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(sample[:max_samples])
    model.train(was_training)
    for hook in hooks:
        hook.remove()
    return features


def _linear_cka(x, y):
    if x.size(0) < 2:
        return None
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xy = torch.linalg.matrix_norm(x.T @ y).pow(2)
    xx = torch.linalg.matrix_norm(x.T @ x)
    yy = torch.linalg.matrix_norm(y.T @ y)
    denom = xx * yy
    if denom.item() <= 0:
        return None
    return float((xy / denom).item())


def _class_separation(feature, labels):
    if labels is None or feature.size(0) < 2:
        return None, None, None, 0, 0
    labels = labels[:feature.size(0)].detach().cpu()
    distances = torch.cdist(feature, feature, p=2)
    upper = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    same = (labels[:, None] == labels[None, :]) & upper
    different = (labels[:, None] != labels[None, :]) & upper
    same_count = int(same.sum().item())
    different_count = int(different.sum().item())
    intra = float(distances[same].mean().item()) if same_count else None
    inter = float(distances[different].mean().item()) if different_count else None
    ratio = (
        inter / max(intra, 1e-12)
        if intra is not None and inter is not None else None
    )
    return intra, inter, ratio, same_count, different_count


def compare_hidden_representations(before, after, labels=None):
    rows = []
    for name in before:
        if name not in after:
            continue
        before_feature = before[name]
        after_feature = after[name]
        if before_feature.shape != after_feature.shape or before_feature.numel() == 0:
            continue

        cosine = F.cosine_similarity(before_feature, after_feature, dim=1)
        l2 = torch.linalg.vector_norm(after_feature - before_feature, dim=1)
        before_intra, before_inter, before_ratio, same_pairs, diff_pairs = (
            _class_separation(before_feature, labels)
        )
        after_intra, after_inter, after_ratio, _, _ = (
            _class_separation(after_feature, labels)
        )
        rows.append({
            'layer': name,
            'stage': _stage_name(name),
            'num_samples': int(before_feature.size(0)),
            'feature_dim': int(before_feature.size(1)),
            'cosine_similarity': float(cosine.mean().item()),
            'l2_distance': float(l2.mean().item()),
            'cka_similarity': _linear_cka(before_feature, after_feature),
            'class_pairs_same': same_pairs,
            'class_pairs_different': diff_pairs,
            'class_intra_distance_before': before_intra,
            'class_intra_distance_after': after_intra,
            'class_inter_distance_before': before_inter,
            'class_inter_distance_after': after_inter,
            'class_separation_before': before_ratio,
            'class_separation_after': after_ratio,
            'class_separation_delta': (
                after_ratio - before_ratio
                if before_ratio is not None and after_ratio is not None else None
            ),
        })
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
