# pruning/sensitivity.py
import os
import copy
import json
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models import build_model, find_prunable_blocks
from pruning.resnet_utils import prune_resnet_blockwise
from pruning.vgg_utils import prune_vgg_blockwise


NUM_CLASSES = 100  # CIFAR-100 기준


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def _test(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def _train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ------------------------------------------------------------
#   ResNet18 민감도
# ------------------------------------------------------------
def compute_sensitivity_resnet18(
    base_ckpt_path,
    trainloader,
    testloader,
    device,
    block_ratios=None,
    finetune_epochs=3
):
    if block_ratios is None:
        block_ratios = [0.0, 0.2, 0.4, 0.6, 0.8]

    model = build_model("resnet18", NUM_CLASSES).to(device)
    state = torch.load(base_ckpt_path, map_location=device)
    model.load_state_dict(state)

    base_acc = _test(model, testloader, device)
    print(f"[Sensitivity-ResNet18] Base Test Accuracy: {base_acc:.2f}%")

    blocks = find_prunable_blocks(model, "resnet18")
    block_names = sorted(blocks.keys())
    sensitivity_si = {}

    for blk_name in block_names:
        print(f"\n[Sensitivity-ResNet18] Block {blk_name}")
        ratio_to_acc = {0.0: base_acc}

        for ratio in block_ratios[1:]:
            print(f"  - Pruning {blk_name} @ {ratio*100:.0f}%")

            tmp = build_model("resnet18").to(device)
            tmp.load_state_dict(state)

            blocks_imp = find_prunable_blocks(tmp, "resnet18")
            block = blocks_imp[blk_name]
            imp = block.bn1.weight.data.abs().cpu()

            C = imp.numel()
            order = imp.argsort()
            k = min(int(C * ratio), C - 1)
            prune = set(order[:k].tolist())
            keep = sorted(list(set(range(C)) - prune))

            block_keep_indices = {
                n: (keep if n == blk_name else list(range(b.conv1.out_channels)))
                for n, b in blocks_imp.items()
            }

            pruned = prune_resnet_blockwise(tmp, block_keep_indices, device)

            optimizer = optim.SGD(pruned.parameters(), lr=0.01, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            for ep in range(finetune_epochs):
                _train_one_epoch(pruned, trainloader, optimizer, criterion, device)

            acc = _test(pruned, testloader, device)
            ratio_to_acc[ratio] = acc
            print(f"    => acc {acc:.2f}")

        slopes = []
        for r1, r2 in zip(block_ratios[:-1], block_ratios[1:]):
            slopes.append(abs((ratio_to_acc[r2] - ratio_to_acc[r1]) / ((r2-r1)*100)))
        sensitivity_si[blk_name] = float(np.mean(slopes))

    return sensitivity_si


# ------------------------------------------------------------
#   VGG16 민감도
# ------------------------------------------------------------
def compute_sensitivity_vgg16(
    base_ckpt_path,
    trainloader,
    testloader,
    device,
    block_ratios=None,
    finetune_epochs=3
):
    if block_ratios is None:
        block_ratios = [0.0, 0.2, 0.4, 0.6, 0.8]

    model = build_model("vgg16", NUM_CLASSES).to(device)
    state = torch.load(base_ckpt_path, map_location=device)
    model.load_state_dict(state)

    base_acc = _test(model, testloader, device)
    print(f"[Sensitivity-VGG16] Base Test Accuracy: {base_acc:.2f}%")

    blocks = find_prunable_blocks(model, "vgg16")
    block_names = sorted(blocks.keys())
    sensitivity_si = {}

    for blk_name in block_names:
        print(f"\n[Sensitivity-VGG16] Block {blk_name}")
        ratio_to_acc = {0.0: base_acc}

        for ratio in block_ratios[1:]:
            print(f"  - Pruning {blk_name} @ {ratio*100:.0f}%")

            tmp = build_model("vgg16").to(device)
            tmp.load_state_dict(state)

            blocks_imp = find_prunable_blocks(tmp, "vgg16")
            block = blocks_imp[blk_name]
            w = block.weight.data
            imp = w.view(w.size(0), -1).abs().sum(dim=1).cpu()

            C = imp.numel()
            order = imp.argsort()
            k = min(int(C * ratio), C - 1)
            prune = set(order[:k].tolist())
            keep = sorted(list(set(range(C)) - prune))

            block_keep_indices = {
                n: (keep if n == blk_name else list(range(b.out_channels)))
                for n, b in blocks_imp.items()
            }

            pruned = prune_vgg_blockwise(tmp, block_keep_indices, device, NUM_CLASSES)

            optimizer = optim.SGD(pruned.parameters(), lr=0.01, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            for ep in range(finetune_epochs):
                _train_one_epoch(pruned, trainloader, optimizer, criterion, device)

            acc = _test(pruned, testloader, device)
            ratio_to_acc[ratio] = acc
            print(f"    => acc {acc:.2f}")

        slopes = []
        for r1, r2 in zip(block_ratios[:-1], block_ratios[1:]):
            slopes.append(abs((ratio_to_acc[r2] - ratio_to_acc[r1]) / ((r2-r1)*100)))
        sensitivity_si[blk_name] = float(np.mean(slopes))

    return sensitivity_si


# ------------------------------------------------------------
#   ResNet152 민감도 (추가된 부분)
# ------------------------------------------------------------
def compute_sensitivity_resnet152(
    base_ckpt_path,
    trainloader,
    testloader,
    device,
    block_ratios=None,
    finetune_epochs=3
):
    """
    Bottleneck 구조(conv2 기준)로 민감도 계산
    """
    if block_ratios is None:
        block_ratios = [0.0, 0.2, 0.4, 0.6, 0.8]

    model = build_model("resnet152", NUM_CLASSES).to(device)
    state = torch.load(base_ckpt_path, map_location=device)
    model.load_state_dict(state)

    base_acc = _test(model, testloader, device)
    print(f"[Sensitivity-ResNet152] Base Test Accuracy: {base_acc:.2f}%")

    blocks = find_prunable_blocks(model, "resnet152")  # Bottleneck 블록 dict
    block_names = sorted(blocks.keys())
    sensitivity_si = {}

    for blk_name in block_names:
        print(f"\n[Sensitivity-ResNet152] Block {blk_name}")
        ratio_to_acc = {0.0: base_acc}

        for ratio in block_ratios[1:]:
            print(f"  - Pruning {blk_name} @ {ratio*100:.0f}%")

            tmp = build_model("resnet152").to(device)
            tmp.load_state_dict(state)

            blocks_imp = find_prunable_blocks(tmp, "resnet152")
            block = blocks_imp[blk_name]

            # Bottleneck 기준: conv2 필터 중요도
            conv_w = block.conv2.weight.data
            imp = conv_w.view(conv_w.size(0), -1).abs().sum(dim=1).cpu()

            C = imp.numel()
            order = imp.argsort()
            k = min(int(C * ratio), C - 1)
            prune = set(order[:k].tolist())
            keep = sorted(list(set(range(C)) - prune))

            block_keep_indices = {}
            for n, b in blocks_imp.items():
                if hasattr(b, "conv2"):
                    block_keep_indices[n] = keep if n == blk_name else list(range(b.conv2.out_channels))
                else:
                    # 혹시 BasicBlock이 섞이면 conv1 사용
                    block_keep_indices[n] = list(range(b.conv1.out_channels))

            pruned = prune_resnet_blockwise(tmp, block_keep_indices, device)

            optimizer = optim.SGD(pruned.parameters(), lr=0.01, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            for ep in range(finetune_epochs):
                _train_one_epoch(pruned, trainloader, optimizer, criterion, device)

            acc = _test(pruned, testloader, device)
            ratio_to_acc[ratio] = acc
            print(f"    => acc {acc:.2f}")

        slopes = []
        for r1, r2 in zip(block_ratios[:-1], block_ratios[1:]):
            slopes.append(abs((ratio_to_acc[r2] - ratio_to_acc[r1]) / ((r2-r1)*100)))
        sensitivity_si[blk_name] = float(np.mean(slopes))

    return sensitivity_si


# ------------------------------------------------------------
#   JSON 로드 + 계산
# ------------------------------------------------------------
def maybe_load_or_compute_sensitivity(
    model_id,
    checkpoint_dir,
    trainloader,
    testloader,
    device,
    block_ratios=None,
    finetune_epochs=3
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    json_path = os.path.join(checkpoint_dir, f"{model_id}_sensitivity.json")
    base_ckpt_path = os.path.join(checkpoint_dir, f"{model_id}_base.pth")

    if os.path.exists(json_path):
        print(f"[Sensitivity] Load from {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 모델별 분기
    if model_id == "resnet18":
        si = compute_sensitivity_resnet18(
            base_ckpt_path, trainloader, testloader, device,
            block_ratios, finetune_epochs
        )

    elif model_id == "vgg16":
        si = compute_sensitivity_vgg16(
            base_ckpt_path, trainloader, testloader, device,
            block_ratios, finetune_epochs
        )

    elif model_id == "resnet152":
        si = compute_sensitivity_resnet152(
            base_ckpt_path, trainloader, testloader, device,
            block_ratios, finetune_epochs
        )

    else:
        raise NotImplementedError(
            f"Sensitivity calculation not implemented for {model_id}"
        )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(si, f, ensure_ascii=False, indent=2)

    print(f"[Sensitivity] Saved sensitivity to {json_path}")
    return si
