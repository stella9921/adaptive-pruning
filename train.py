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

# "resnet18" 또는 "vgg16"
MODEL_ID = "vgg16"

from google.colab import drive
drive.mount("/content/drive")

CHECKPOINT_DIR = "/content/drive/MyDrive/ckpt_block_sweep"
BASE_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f"{MODEL_ID}_base.pth")

USE_AMP = torch.cuda.is_available()
BATCH_SIZE = 128


# ------------------------------------------------
# 1) 데이터: CIFAR-100 + train/val 분리
# ------------------------------------------------
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ]
)
transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ]
)

full_trainset = torchvision.datasets.CIFAR100(
    root="./data", train=True, download=True, transform=transform_train
)
train_size = int(0.9 * len(full_trainset))
val_size = len(full_trainset) - train_size
trainset, validationset = random_split(full_trainset, [train_size, val_size])

testset = torchvision.datasets.CIFAR100(
    root="./data", train=False, download=True, transform=transform_test
)

trainloader = DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
validationloader = DataLoader(
    validationset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
testloader = DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)


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
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
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

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")


@torch.no_grad()
def test(model, loader) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


# ------------------------------------------------
# 3) 메인 반복 프루닝 함수
# ------------------------------------------------
def main_adaptive_iterative_pruning(
    n_rounds: int = 10,
    finetune_epochs_per_round: int = 5,
    final_finetune_epochs: int = 10,
    alpha: float = 0.1,
    strategy: str = "vanilla",  # "vanilla", "p", "beta"
    p: float = 2.5,
    beta: float = 0.5,
    GLOBAL_PRUNING_TARGET_RATIO: float = 60.0,
):
    print(
        f"Strategy: {strategy} | Model: {MODEL_ID} | "
        f"Rounds: {n_rounds}, alpha={alpha}"
    )

    # 1) 기준 모델 로드
    base_model = build_model(MODEL_ID, num_classes=100).to(device)
    base_state = torch.load(BASE_MODEL_PATH, map_location=device)
    base_model.load_state_dict(base_state)

    base_params = count_parameters(base_model)
    print(f"Base parameter count: {base_params:,}")

    print("--- Base Test Accuracy ---")
    last_accuracy = test(base_model, testloader)
    print(f"Test Accuracy: {last_accuracy:.2f}%")

    pruned_model = copy.deepcopy(base_model)

    # 2) 민감도 로드 또는 계산
    sensitivity_si = maybe_load_or_compute_sensitivity(
        model_id=MODEL_ID,
        checkpoint_dir=CHECKPOINT_DIR,
        trainloader=trainloader,
        testloader=testloader,
        device=device,
        block_ratios=[0.0, 0.2, 0.4, 0.6, 0.8],
        finetune_epochs=3,
    )

    # 3) 블록별 원본 필터 수 / 파라미터 수
    filter_counts_Ni_original = get_filter_counts(MODEL_ID)
    blocks_base = find_prunable_blocks(base_model, MODEL_ID)
    param_counts_Ni = {
        name: count_parameters(block) for name, block in blocks_base.items()
    }
    total_block_params = sum(param_counts_Ni.values())

    # ------------------------------------------------
    # 라운드 반복
    # ------------------------------------------------
    for round_idx in range(1, n_rounds + 1):
        print(f"\n===== Round {round_idx}/{n_rounds} =====")

        # 3가지 전략 중 하나로 블록별 프루닝 비율 계산
        if strategy == "vanilla":
            dynamic_ratios = compute_dynamic_ratios_vanilla(
                round_idx=round_idx,
                n_rounds=n_rounds,
                sensitivity_si=sensitivity_si,
                filter_counts_Ni_original=filter_counts_Ni_original,
                global_target_ratio=GLOBAL_PRUNING_TARGET_RATIO,
            )
        elif strategy == "p":
            dynamic_ratios = compute_dynamic_ratios_p(
                round_idx=round_idx,
                n_rounds=n_rounds,
                sensitivity_si=sensitivity_si,
                param_counts_Ni=param_counts_Ni,
                total_block_params=total_block_params,
                global_target_ratio=GLOBAL_PRUNING_TARGET_RATIO,
                p=p,
            )
        elif strategy == "beta":
            dynamic_ratios = compute_dynamic_ratios_beta(
                round_idx=round_idx,
                n_rounds=n_rounds,
                sensitivity_si=sensitivity_si,
                param_counts_Ni=param_counts_Ni,
                total_block_params=total_block_params,
                global_target_ratio=GLOBAL_PRUNING_TARGET_RATIO,
                beta=beta,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # 현재 프루닝된 모델 기준으로 블록 모듈 가져오기
        blocks_current = find_prunable_blocks(pruned_model, MODEL_ID)
        global_keep_indices = {}

        print("\n[Round pruning plan]")
        for block_name in sorted(filter_counts_Ni_original.keys()):
            original_C = filter_counts_Ni_original[block_name]
            ratio_to_prune = dynamic_ratios.get(block_name, 0.0)

            num_to_keep = int(original_C * (1.0 - ratio_to_prune / 100.0))
            num_to_keep = max(1, num_to_keep)

            block = blocks_current[block_name]

            # 중요도 계산: 모델 종류별로 다르게
            if MODEL_ID == "resnet18":
                # BasicBlock: bn1 기준
                importance = block.bn1.weight.data.abs().cpu()
            elif MODEL_ID == "vgg16":
                # Conv2d: weight L1-norm (out_channel별)
                w = block.weight.data  # [C_out, C_in, k, k]
                importance = w.abs().mean(dim=(1, 2, 3)).cpu()
            else:
                raise ValueError(f"Unknown MODEL_ID: {MODEL_ID}")

            order = importance.argsort(descending=True)
            keep_idx = sorted(order[:num_to_keep].tolist())
            global_keep_indices[block_name] = keep_idx

            print(
                f"Block {block_name:>10} | original {original_C:4d} -> "
                f"keep {len(keep_idx):4d} (prune {ratio_to_prune:.1f}%)"
            )

        # 실제 프루닝 적용
        if MODEL_ID == "resnet18":
            pruned_model = prune_resnet_blockwise(
                pruned_model, global_keep_indices, device
            )
        elif MODEL_ID == "vgg16":
            pruned_model = prune_vgg_blockwise(
                pruned_model, global_keep_indices, device, num_classes=100
            )
        else:
            raise ValueError(f"Unknown MODEL_ID: {MODEL_ID}")

        # 라운드별 파인튜닝
        optimizer = optim.SGD(
            pruned_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
        )
        criterion = nn.CrossEntropyLoss()

        print(
            f"\nFine-tuning pruned model for "
            f"{finetune_epochs_per_round} epochs (round {round_idx})"
        )
        for ep in range(1, finetune_epochs_per_round + 1):
            train_one_epoch(pruned_model, trainloader, optimizer, criterion, ep)

        # 검증 정확도 측정
        current_val_acc = test(pruned_model, validationloader)
        print(f"Validation Accuracy after round {round_idx}: {current_val_acc:.2f}%")

        accuracy_drop = max(0.0, last_accuracy - current_val_acc)
        print(
            f"Prev acc: {last_accuracy:.2f} -> "
            f"current: {current_val_acc:.2f} (drop {accuracy_drop:.2f})"
        )

        # 민감도 업데이트
        update_value = accuracy_drop * alpha
        if update_value > 0:
            for name in sensitivity_si.keys():
                sensitivity_si[name] += update_value

        last_accuracy = current_val_acc

    # ------------------------------------------------
    # 최종 재학습
    # ------------------------------------------------
    print("\n===== Final retraining on pruned model =====")
    optimizer = optim.SGD(
        pruned_model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4
    )
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, final_finetune_epochs + 1):
        train_one_epoch(pruned_model, trainloader, optimizer, criterion, ep)

    final_params = count_parameters(pruned_model)
    compression_rate = (1.0 - final_params / base_params) * 100.0

    print("\n--- Final model summary ---")
    print(f"Final parameter count: {final_params:,}")
    print(f"Compression rate: {compression_rate:.2f}%")

    final_val_acc = test(pruned_model, validationloader)
    final_test_acc = test(pruned_model, testloader)
    print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")


# ------------------------------------------------
# 4) 스크립트 실행
# ------------------------------------------------
if __name__ == "__main__":
    main_adaptive_iterative_pruning(
        n_rounds=10,
        finetune_epochs_per_round=5,
        final_finetune_epochs=10,
        alpha=0.1,
        strategy="vanilla",
        GLOBAL_PRUNING_TARGET_RATIO=80.0,
    )
