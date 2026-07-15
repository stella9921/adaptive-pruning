import os

from PIL import Image, ImageDraw


def _draw_bars(path, title, rows, label_key, value_key, selected_key=None):
    if not rows:
        return None
    width = 1200
    row_height = 26
    top = 58
    left = 300
    right = 170
    height = top + row_height * len(rows) + 30
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((20, 18), title, fill='black')
    values = [float(row[value_key] or 0.0) for row in rows]
    max_value = max(max(values), 1e-12)
    bar_width = width - left - right

    for index, (row, value) in enumerate(zip(rows, values)):
        y = top + index * row_height
        label = str(row[label_key])
        if len(label) > 42:
            label = label[:39] + '...'
        draw.text((20, y + 4), label, fill='black')
        color = '#2f80ed'
        if selected_key is not None:
            color = '#eb5757' if row.get(selected_key) else '#bdbdbd'
        length = int(bar_width * value / max_value)
        draw.rectangle((left, y + 3, left + length, y + 20), fill=color)
        if max_value < 0.01:
            text = f"{value:.3e}"
        elif value_key.endswith('percent'):
            text = f"{value:.2f}%"
        elif value_key.endswith('_mb'):
            text = f"{value:.2f} MB"
        else:
            text = f"{value:.4f}"
        draw.text((left + bar_width + 12, y + 4), text, fill='black')
    image.save(path)
    return path


def save_pruning_plots(snapshot, run_dir, run_id, epoch):
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    specs = [
        ('group_selection', 'MCPrune Group Selection', 'group_id', 'grad_ema', 'selected'),
        ('group_sparsity', 'Topology Group Parameter Sparsity', 'group_id', 'sparsity_percent', None),
        ('layer_sparsity', 'Layer Parameter Sparsity', 'layer', 'sparsity_percent', None),
        ('stage_sparsity', 'Model Stage Parameter Sparsity', 'stage', 'sparsity_percent', None),
        (
            'module_memory', 'Module Activation Volume',
            'layer', 'activation_volume_mb', None,
        ),
        (
            'activation_abs_mean_before',
            'Activation Abs Mean Before Pruning',
            'layer', 'abs_mean', None,
        ),
        (
            'activation_abs_mean_after',
            'Activation Abs Mean After Pruning',
            'layer', 'abs_mean', None,
        ),
        (
            'activation_zero_fraction_after',
            'Activation Zero Fraction After Pruning',
            'layer', 'zero_fraction', None,
        ),
    ]
    paths = []
    for name, title, label_key, value_key, selected_key in specs:
        path = os.path.join(
            plots_dir, f"{run_id}__{name}__epoch-{int(epoch):03d}.png"
        )
        saved = _draw_bars(
            path, title, snapshot.get(name, []), label_key, value_key, selected_key
        )
        if saved:
            paths.append(saved)
    return paths


def _draw_line_chart(path, title, rows, x_key, y_key):
    points = []
    for row in rows:
        if x_key not in row or y_key not in row:
            continue
        if row[x_key] is None or row[y_key] is None:
            continue
        try:
            points.append((float(row[x_key]), float(row[y_key])))
        except (TypeError, ValueError):
            continue
    if not points:
        return None
    width, height = 1000, 600
    left, right, top, bottom = 90, 40, 60, 70
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((20, 18), title, fill='black')
    draw.line((left, top, left, height - bottom), fill='black', width=2)
    draw.line(
        (left, height - bottom, width - right, height - bottom),
        fill='black', width=2,
    )
    xs, ys = zip(*points)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0

    pixels = []
    for x_value, y_value in points:
        x = left + (x_value - x_min) / (x_max - x_min) * (width - left - right)
        y = height - bottom - (y_value - y_min) / (y_max - y_min) * (
            height - top - bottom
        )
        pixels.append((x, y))
    if len(pixels) > 1:
        draw.line(pixels, fill='#2f80ed', width=3)
    for x, y in pixels:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill='#eb5757')
    draw.text((left, height - 45), f"{x_key}: {x_min:.3f} to {x_max:.3f}", fill='black')
    draw.text((left, top - 20), f"{y_key}: {y_min:.3f} to {y_max:.3f}", fill='black')
    image.save(path)
    return path


def save_history_plots(history, run_dir, run_id):
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    specs = [
        ('validation-accuracy', 'Validation Accuracy by Epoch', 'epoch', 'val_accuracy'),
        ('training-loss', 'Training Loss by Epoch', 'epoch', 'train_loss'),
        (
            'accuracy-sparsity', 'Accuracy vs Parameter Sparsity',
            'parameter_sparsity', 'val_accuracy',
        ),
        ('peak-vram', 'Peak Allocated VRAM by Epoch', 'epoch', 'cuda_peak_allocated_mb'),
        ('inference-latency', 'Inference Latency by Epoch', 'epoch', 'inference_latency_ms'),
    ]
    paths = []
    for name, title, x_key, y_key in specs:
        path = os.path.join(plots_dir, f"{run_id}__{name}.png")
        saved = _draw_line_chart(path, title, history, x_key, y_key)
        if saved:
            paths.append(saved)
    return paths
