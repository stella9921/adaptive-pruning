# train.py

import os
import copy

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

# 전략 함수들
from pruning.strategies import (
    compute_dynamic_ratios_vanilla,
    compute_dynamic_ratios_p,
    compute_dynamic_ratios_beta,
)

# 모델/블록/필터 정보
from models import build_model, find_prunable_blocks, get_filter_counts

# 프루닝 유틸
from pruning.resnet_utils import prune_resnet_blockwise
from pruning.vgg_utils import prune_vgg_blockwise
from pruning.efficientnet_utils import prune_efficientnet_blockwise

# 민감도 로드/계산
from pruning.sensitivity import maybe_load_or_compute_sensitivity


# ------------------------------------------------
# 0) 설정
# ------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 사용할 모델 설정
MODEL_ID = "efficientnet_b0"

from google.colab import drive
drive.mount("/content/drive")

CHECKPOINT_DIR = "/content/drive/MyDrive/ckpt_block_sweep"
BASE_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f"{MODEL_ID}_base.pth")

USE_AMP = torch.cuda.is_available()
BATCH_SIZE = 128


# ------------------------------------------------
# 1) 데이터 준비 (CIFAR-100)
# ------------------------------------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

full_trainset = torchvision.datasets.CIFAR100(
    root="./data", train=True, download=True, transform=transform_train
)
train_size = int(0.9 * len(full_trainset))
val_size = len(full_trainset) - train_size
trainset, validationset = random_split(full_trainset, [train_size, val_size])

testset = torchvision.datasets.CIFAR100(
    root="./data", train=False, download=True, transform=transform_test
)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
validationloader = DataLoader(validationset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


# ------------------------------------------------
# 2) 유틸 함수
# ------------------------------------------------
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if USE_AMP:
    scaler = torch.amp.GradScaler(device.type)
else:
    scaler = None


def train_one_epoch(model, loader, optimizer, criterion, epoch: int):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        if USE_AMP:
            with torch.amp.autocast(device_type=device.type):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} | Loss: {total_loss / len(loader):.4f}")


@torch.no_grad()
def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


# ------------------------------------------------
# 3) 기존 반복(Post-training) 프루닝 함수 ← 그대로 유지
# ------------------------------------------------
def main_adaptive_iterative_pruning(
    n_rounds=10,
    finetune_epochs_per_round=5,
    final_finetune_epochs=10,
    alpha=0.1,
    strategy="vanilla",
    p=2.5,
    beta=0.5,
    GLOBAL_PRUNING_TARGET_RATIO=60.0,
):
    print(f"Strategy: {strategy} | Model: {MODEL_ID}")

    base_model = build_model(MODEL_ID, num_classes=100).to(device)
    base_state = torch.load(BASE_MODEL_PATH, map_location=device)
    base_model.load_state_dict(base_state)

    base_params = count_parameters(base_model)
    print(f"Base parameter count: {base_params:,}")

    last_accuracy = test(base_model, testloader)
    print(f"Base Test Accuracy: {last_accuracy:.2f}%")

    pruned_model = copy.deepcopy(base_model)

    sensitivity_si = maybe_load_or_compute_sensitivity(
        model_id=MODEL_ID,
        checkpoint_dir=CHECKPOINT_DIR,
        trainloader=trainloader,
        testloader=testloader,
        device=device,
        block_ratios=[0.0, 0.2, 0.4, 0.6, 0.8],
        finetune_epochs=3,
    )

    filter_counts_Ni_original = get_filter_counts(MODEL_ID)
    blocks_base = find_prunable_blocks(base_model, MODEL_ID)
    param_counts_Ni = {name: count_parameters(block) for name, block in blocks_base.items()}
    total_block_params = sum(param_counts_Ni.values())

    for round_idx in range(1, n_rounds + 1):
        print(f"\n===== Round {round_idx}/{n_rounds} =====")

        if strategy == "vanilla":
            dynamic_ratios = compute_dynamic_ratios_vanilla(
                round_idx, n_rounds, sensitivity_si,
                filter_counts_Ni_original,
                GLOBAL_PRUNING_TARGET_RATIO,
            )
        elif strategy == "p":
            dynamic_ratios = compute_dynamic_ratios_p(
                round_idx, n_rounds, sensitivity_si,
                param_counts_Ni, total_block_params,
                GLOBAL_PRUNING_TARGET_RATIO, p
            )
        elif strategy == "beta":
            dynamic_ratios = compute_dynamic_ratios_beta(
                round_idx, n_rounds, sensitivity_si,
                param_counts_Ni, total_block_params,
                GLOBAL_PRUNING_TARGET_RATIO, beta
            )

        blocks_current = find_prunable_blocks(pruned_model, MODEL_ID)
        global_keep_indices = {}

        for block_name in sorted(filter_counts_Ni_original.keys()):
            original_C = filter_counts_Ni_original[block_name]
            ratio_to_prune = dynamic_ratios.get(block_name, 0.0)
            num_to_keep = max(1, int(original_C * (1 - ratio_to_prune / 100.0)))

            block = blocks_current[block_name]

            if MODEL_ID == "efficientnet_b0":
                w = block.weight.data
                importance = w.view(w.size(0), -1).abs().sum(dim=1)
            else:
                raise ValueError("Other models omitted for brevity")

            order = importance.argsort(descending=True)
            keep_idx = sorted(order[:num_to_keep].tolist())
            global_keep_indices[block_name] = keep_idx

        if MODEL_ID == "efficientnet_b0":
            pruned_model = prune_efficientnet_blockwise(pruned_model, global_keep_indices, device)

        optimizer = optim.SGD(pruned_model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for ep in range(finetune_epochs_per_round):
            train_one_epoch(pruned_model, trainloader, optimizer, criterion, ep+1)

        last_accuracy = test(pruned_model, validationloader)
        print(f"Val Acc: {last_accuracy:.2f}%")


# ------------------------------------------------
# ⭐ NEW ⭐ 4) In-training EMA Gradient Pruning
# ------------------------------------------------
def main_intraining_ema_pruning(
    total_epochs=60,
    prune_epochs=[20, 40],
    prune_ratio=0.3,
    ema_decay=0.9,
):
    print(f"\n===== In-Training EMA Gradient Pruning | Model={MODEL_ID} =====")

    model = build_model(MODEL_ID, num_classes=100).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    blocks = find_prunable_blocks(model, MODEL_ID)
    grad_importance = {name: 0.0 for name in blocks.keys()}

    for epoch in range(1, total_epochs + 1):
        print(f"\n--- Epoch {epoch}/{total_epochs} ---")
        model.train()

        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()

            # EMA 업데이트
            for blk_name, blk in blocks.items():
                if hasattr(blk, "weight") and blk.weight.grad is not None:
                    g = blk.weight.grad.abs().mean().item()
                    grad_importance[blk_name] = (
                        ema_decay * grad_importance[blk_name] +
                        (1 - ema_decay) * g
                    )

            optimizer.step()

        # prune 시점이면 pruning 수행
        if epoch in prune_epochs:
            print(f">>> Pruning at epoch {epoch}")

            sorted_blocks = sorted(grad_importance.items(), key=lambda x: x[1])
            num_prune = int(len(sorted_blocks) * prune_ratio)
            prune_targets = [b[0] for b in sorted_blocks[:num_prune]]

            keep_indices = {}
            for blk_name, blk in blocks.items():
                w = blk.weight.data
                C = w.size(0)
                k = max(1, int(C * (1 - prune_ratio)))
                importance = w.view(C, -1).abs().mean(dim=1)
                order = importance.argsort(descending=True)
                keep_indices[blk_name] = sorted(order[:k].tolist())

            model = prune_efficientnet_blockwise(model, keep_indices, device)
            blocks = find_prunable_blocks(model, MODEL_ID)

        val_acc = test(model, validationloader)
        print(f"Validation Acc: {val_acc:.2f}%")

    test_acc = test(model, testloader)
    print(f"\n===== Final Test Accuracy: {test_acc:.2f}% =====")


# ------------------------------------------------
# 5) 실행부
# ------------------------------------------------
if __name__ == "__main__":

    # 기존 방식 사용하려면
    # main_adaptive_iterative_pruning()

    # ⭐ 새 방식 사용하려면 이거!
    main_intraining_ema_pruning(
        total_epochs=50,
        prune_epochs=[15, 30, 45],
        prune_ratio=0.3,
        ema_decay=0.9,
    )
