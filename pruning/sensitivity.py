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
from pruning.efficientnet_utils import (
    prune_efficientnet_blockwise,
    find_prunable_blocks_efficientnet,
)

NUM_CLASSES = 100


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def _test(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def _train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

##############################################################################
# EfficientNet-B0 민감도
##############################################################################
def compute_sensitivity_efficientnet_b0(
    base_ckpt_path,
    trainloader,
    testloader,
    device,
    block_ratios=None,
    finetune_epochs=3
):

    if block_ratios is None:
        block_ratios = [0.0, 0.2, 0.4, 0.6, 0.8]

    model = build_model("efficientnet_b0", NUM_CLASSES).to(device)
    state = torch.load(base_ckpt_path, map_location=device)
    model.load_state_dict(state)

    base_acc = _test(model, testloader, device)
    print(f"[Sensitivity-EfficientNet-B0] Base Test Accuracy: {base_acc:.2f}%")

    blocks = find_prunable_blocks_efficientnet(model)
    block_names = sorted(blocks.keys())
    sensitivity_si = {}

    for blk_name in block_names:
        print(f"\n[Sensitivity-EfficientNet-B0] Block {blk_name}")
        ratio_to_acc = {0.0: base_acc}

        for ratio in block_ratios[1:]:
            print(f"  - Pruning {blk_name} @ {int(ratio*100)}%")

            tmp = build_model("efficientnet_b0", NUM_CLASSES).to(device)
            tmp.load_state_dict(state)

            blocks_imp = find_prunable_blocks_efficientnet(tmp)
            block = blocks_imp[blk_name]

            conv_w = block.weight.data
            imp = conv_w.view(conv_w.size(0), -1).abs().sum(dim=1).cpu()

            C = imp.numel()
            order = imp.argsort()
            k = min(int(C * ratio), C - 1)
            prune_idx = set(order[:k].tolist())
            keep_idx = sorted(list(set(range(C)) - prune_idx))

            block_keep_indices = {
                n: (keep_idx if n == blk_name else list(range(b.weight.size(0))))
                for n, b in blocks_imp.items()
            }

            pruned = prune_efficientnet_blockwise(tmp, block_keep_indices, device)

            optimizer = optim.SGD(pruned.parameters(), lr=0.01, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            for ep in range(finetune_epochs):
                _train_one_epoch(pruned, trainloader, optimizer, criterion, device)

            acc = _test(pruned, testloader, device)
            ratio_to_acc[ratio] = acc
            print(f"    => acc {acc:.2f}")

        slopes = []
        for r1, r2 in zip(block_ratios[:-1], block_ratios[1:]):
            slopes.append(abs((ratio_to_acc[r2] - ratio_to_acc[r1]) / ((r2 - r1) * 100)))

        sensitivity_si[blk_name] = float(np.mean(slopes))

    return sensitivity_si


##############################################################################
# 기존 ResNet18 / VGG16 / ResNet152 그대로 유지
##############################################################################
# … (너의 원래 코드 그대로 유지 — 생략)

##############################################################################
# maybe_load_or_compute_sensitivity 에 EfficientNet 추가
##############################################################################
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
        with open(json_path, "r") as f:
            return json.load(f)

    if model_id == "resnet18":
        si = compute_sensitivity_resnet18(base_ckpt_path, trainloader, testloader, device, block_ratios, finetune_epochs)
    elif model_id == "vgg16":
        si = compute_sensitivity_vgg16(base_ckpt_path, trainloader, testloader, device, block_ratios, finetune_epochs)
    elif model_id == "resnet152":
        si = compute_sensitivity_resnet152(base_ckpt_path, trainloader, testloader, device, block_ratios, finetune_epochs)

    elif model_id == "efficientnet_b0":
        si = compute_sensitivity_efficientnet_b0(
            base_ckpt_path, trainloader, testloader, device,
            block_ratios, finetune_epochs
        )

    else:
        raise NotImplementedError(f"Sensitivity not implemented for {model_id}")

    with open(json_path, "w") as f:
        json.dump(si, f, indent=2)

    print(f"[Sensitivity] Saved sensitivity to {json_path}")
    return si
