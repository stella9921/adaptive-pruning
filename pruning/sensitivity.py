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
from pruning.vgg_utils import prune_vgg_blockwise  # VGG 프루닝 유틸

NUM_CLASSES = 100  # 현재 CIFAR-100 기준. 나중에 다른 데이터면 바꿀 수 있음.


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


def compute_sensitivity_resnet18(
    base_ckpt_path: str,
    trainloader,
    testloader,
    device,
    block_ratios: List[float] = None,
    finetune_epochs: int = 3,
) -> Dict[str, float]:
    """
    ResNet-18에 대해 '블록별 민감도'를 자동 계산해서 dict로 반환.
      - 각 블록을 0, 20, 40, 60, 80% 프루닝
      - 각 점에서의 정확도 측정
      - [0-20], [20-40], [40-60], [60-80] 구간별 기울기(정확도 변화율) 평균 → 민감도
    """
    if block_ratios is None:
        block_ratios = [0.0, 0.2, 0.4, 0.6, 0.8]

    model = build_model("resnet18", num_classes=NUM_CLASSES).to(device)
    state = torch.load(base_ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    base_acc = _test(model, testloader, device)
    print(f"[Sensitivity-ResNet18] Base Test Accuracy: {base_acc:.2f}%")

    block_modules = find_prunable_blocks(model, "resnet18")
    block_names = sorted(block_modules.keys())

    sensitivity_si = {}

    for blk_name in block_names:
        print(f"\n[Sensitivity-ResNet18] Block: {blk_name}")
        ratio_to_acc = {0.0: base_acc}

        for ratio in block_ratios[1:]:
            print(f"  - Pruning {blk_name} @ {int(ratio * 100)}%")

            tmp_model = build_model("resnet18", num_classes=NUM_CLASSES).to(device)
            tmp_model.load_state_dict(state)

            blocks_for_imp = find_prunable_blocks(tmp_model, "resnet18")
            block = blocks_for_imp[blk_name]
            imp = block.bn1.weight.data.abs().cpu()
            C = imp.numel()

            order = imp.argsort()  # ascending
            k = int(C * ratio)
            k = min(k, max(C - 1, 0))
            prune_idx = set(order[:k].tolist())
            keep_idx = sorted(list(set(range(C)) - prune_idx))
            if len(keep_idx) == 0:
                keep_idx = [order[-1].item()]

            block_keep_indices = {}
            for name2, blk2 in blocks_for_imp.items():
                if name2 == blk_name:
                    block_keep_indices[name2] = keep_idx
                else:
                    block_keep_indices[name2] = list(range(blk2.conv1.out_channels))

            pruned = prune_resnet_blockwise(tmp_model, block_keep_indices, device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(
                pruned.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
            )
            for ep in range(1, finetune_epochs + 1):
                loss = _train_one_epoch(pruned, trainloader, optimizer, criterion, device)
                print(f"      epoch {ep}/{finetune_epochs}, loss={loss:.4f}")

            acc = _test(pruned, testloader, device)
            print(f"      => acc={acc:.2f}%")
            ratio_to_acc[ratio] = acc

        slopes = []
        for r1, r2 in zip(block_ratios[:-1], block_ratios[1:]):
            a1 = ratio_to_acc[r1]
            a2 = ratio_to_acc[r2]
            dr = (r2 - r1) * 100.0
            slope = (a2 - a1) / dr
            slopes.append(abs(slope))

        si = float(np.mean(slopes))
        sensitivity_si[blk_name] = si
        print(f"  => sensitivity[{blk_name}] = {si:.6f}")

    return sensitivity_si


def compute_sensitivity_vgg16(
    base_ckpt_path: str,
    trainloader,
    testloader,
    device,
    block_ratios: List[float] = None,
    finetune_epochs: int = 3,
) -> Dict[str, float]:
    """
    VGG16에 대해 '블록별 민감도'를 자동 계산하는 버전.
    블록 단위 정의는 models.find_prunable_blocks("vgg16")에 따르고,
    중요도는 Conv2d weight의 L1-norm(out_channel 기준)을 사용.
    """
    if block_ratios is None:
        block_ratios = [0.0, 0.2, 0.4, 0.6, 0.8]

    model = build_model("vgg16", num_classes=NUM_CLASSES).to(device)
    state = torch.load(base_ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    base_acc = _test(model, testloader, device)
    print(f"[Sensitivity-VGG16] Base Test Accuracy: {base_acc:.2f}%")

    block_modules = find_prunable_blocks(model, "vgg16")  # 예: conv 블록 dict
    block_names = sorted(block_modules.keys())

    sensitivity_si = {}

    for blk_name in block_names:
        print(f"\n[Sensitivity-VGG16] Block: {blk_name}")
        ratio_to_acc = {0.0: base_acc}

        for ratio in block_ratios[1:]:
            print(f"  - Pruning {blk_name} @ {int(ratio * 100)}%")

            tmp_model = build_model("vgg16", num_classes=NUM_CLASSES).to(device)
            tmp_model.load_state_dict(state)

            blocks_for_imp = find_prunable_blocks(tmp_model, "vgg16")
            block = blocks_for_imp[blk_name]  # nn.Conv2d 라고 가정
            conv_w = block.weight.data
            imp = conv_w.view(conv_w.size(0), -1).abs().sum(dim=1).cpu()
            C = imp.numel()

            order = imp.argsort()  # ascending
            k = int(C * ratio)
            k = min(k, max(C - 1, 0))
            prune_idx = set(order[:k].tolist())
            keep_idx = sorted(list(set(range(C)) - prune_idx))
            if len(keep_idx) == 0:
                keep_idx = [order[-1].item()]

            block_keep_indices = {}
            for name2, blk2 in blocks_for_imp.items():
                if name2 == blk_name:
                    block_keep_indices[name2] = keep_idx
                else:
                    # VGG에서는 blk2가 Conv2d 모듈이라고 가정하고 out_channels 사용
                    block_keep_indices[name2] = list(range(blk2.out_channels))

            pruned = prune_vgg_blockwise(
                tmp_model,
                block_keep_indices,
                device,
                num_classes=NUM_CLASSES,
            )

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(
                pruned.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
            )
            for ep in range(1, finetune_epochs + 1):
                loss = _train_one_epoch(pruned, trainloader, optimizer, criterion, device)
                print(f"      epoch {ep}/{finetune_epochs}, loss={loss:.4f}")

            acc = _test(pruned, testloader, device)
            print(f"      => acc={acc:.2f}%")
            ratio_to_acc[ratio] = acc

        slopes = []
        for r1, r2 in zip(block_ratios[:-1], block_ratios[1:]):
            a1 = ratio_to_acc[r1]
            a2 = ratio_to_acc[r2]
            dr = (r2 - r1) * 100.0
            slope = (a2 - a1) / dr
            slopes.append(abs(slope))

        si = float(np.mean(slopes))
        sensitivity_si[blk_name] = si
        print(f"  => sensitivity[{blk_name}] = {si:.6f}")

    return sensitivity_si


def maybe_load_or_compute_sensitivity(
    model_id: str,
    checkpoint_dir: str,
    trainloader,
    testloader,
    device,
    block_ratios: List[float] = None,
    finetune_epochs: int = 3,
) -> Dict[str, float]:
    """
    1) sensitivity JSON이 있으면 읽어오고
    2) 없으면 계산해서 JSON 저장 후 반환
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    json_path = os.path.join(checkpoint_dir, f"{model_id}_sensitivity.json")
    base_ckpt_path = os.path.join(checkpoint_dir, f"{model_id}_base.pth")

    if os.path.exists(json_path):
        print(f"[Sensitivity] Load from {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if model_id == "resnet18":
        print(f"[Sensitivity] No JSON found. Compute sensitivity for {model_id}...")
        si = compute_sensitivity_resnet18(
            base_ckpt_path=base_ckpt_path,
            trainloader=trainloader,
            testloader=testloader,
            device=device,
            block_ratios=block_ratios,
            finetune_epochs=finetune_epochs,
        )
    elif model_id == "vgg16":
        print(f"[Sensitivity] No JSON found. Compute sensitivity for {model_id}...")
        si = compute_sensitivity_vgg16(
            base_ckpt_path=base_ckpt_path,
            trainloader=trainloader,
            testloader=testloader,
            device=device,
            block_ratios=block_ratios,
            finetune_epochs=finetune_epochs,
        )
    else:
        raise NotImplementedError(
            f"model_id={model_id}에 대한 sensitivity 계산 함수는 아직 구현되지 않음"
        )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(si, f, ensure_ascii=False, indent=2)
    print(f"[Sensitivity] Saved sensitivity to {json_path}")

    return si
