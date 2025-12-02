import os, json, time, copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import matplotlib.pyplot as plt

# 모델/블록 정보
from models import (
    build_model,            # build_model(model_id, num_classes)
    find_prunable_blocks,   # find_prunable_blocks(model, model_id)
    get_filter_counts,      # get_filter_counts(model_id)
)

# ResNet 전용 프루닝 유틸
from pruning.resnet_utils import prune_resnet_blockwise

# 전략 함수들 (전략1/2/3)
from pruning.strategies import (
    compute_dynamic_ratios_vanilla,
    compute_dynamic_ratios_p,
    compute_dynamic_ratios_beta,
)

# -------------------------------
# 0) 설정
# -------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

MODEL_ID = "resnet18"  # 나중에 "vgg16", "efficientnet_b0" 등으로 확장 예정

from google.colab import drive; drive.mount('/content/drive')
CHECKPOINT_DIR = "/content/drive/MyDrive/ckpt_block_sweep"
RESULTS_JSON = os.path.join(CHECKPOINT_DIR, f"{MODEL_ID}_per_block_results.json")
BASE_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f"{MODEL_ID}_base.pth")

USE_AMP = torch.cuda.is_available()
BATCH_SIZE = 128

# -------------------------------
# 1) 데이터: CIFAR-100 (Validation Set 분리)
# -------------------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616),
    ),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616),
    ),
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

trainloader = DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=True
)
validationloader = DataLoader(
    validationset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True
)
testloader = DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True
)

# -------------------------------
# 2) 유틸
# -------------------------------
def count_parameters(model):
    """모델의 학습 가능한 파라미터 수를 계산합니다."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -------------------------------
# 3) 학습/평가 루프
# -------------------------------
if USE_AMP:
    scaler = torch.amp.GradScaler(device.type)
else:
    scaler = None


def train_one_epoch(model, loader, optimizer, criterion, epoch, scheduler=None):
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
    if scheduler:
        scheduler.step()
    print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f}")


@torch.no_grad()
def test(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    acc = 100.0 * correct / total
    return acc


# -------------------------------
# 6) 메인 함수 (전략 선택 가능 버전)
# -------------------------------
def main_adaptive_iterative_pruning(
    n_rounds=10,
    finetune_epochs_per_round=5,
    final_finetune_epochs=10,
    alpha=0.1,
    strategy="vanilla",          # "vanilla", "p", "beta"
    p=2.5,                       # 전략 2에서 사용하는 p
    beta=0.5,                    # 전략 3에서 사용하는 beta
    GLOBAL_PRUNING_TARGET_RATIO=60.0,  # 기본은 전략1에서 네가 쓴 60%
):
    """
    전략 1/2/3을 선택해서 반복 프루닝을 수행하는 함수.
    모델 종류는 MODEL_ID로 제어.
    """
    print(f"===== 전략: {strategy} | 모델: {MODEL_ID} | 총 {n_rounds} 라운드, alpha={alpha} =====")

    # 1) 기준 모델 생성 및 로드
    base_model = build_model(MODEL_ID, num_classes=100).to(device)
    base_model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device))
    base_params = count_parameters(base_model)
    print(f"기준 모델 파라미터 수: {base_params:,}")

    print("--- 기준 모델 Test Accuracy ---")
    last_accuracy = test(base_model, testloader)
    print(f"Test Accuracy: {last_accuracy:.2f}%")

    pruned_model = copy.deepcopy(base_model)

    # 2) 민감도 / 필터 정보 (ResNet18 기준, MODEL_ID별로 달라질 수 있음)
    #    민감도는 현재 ResNet18에 맞춰 수동으로 넣어둔 값
    sensitivity_si = {
        'layer1.0': 0.0221, 'layer1.1': 0.0551,
        'layer2.0': 0.1250, 'layer2.1': 0.0319,
        'layer3.0': 0.1564, 'layer3.1': 0.1011,
        'layer4.0': 0.0768, 'layer4.1': 0.0488
    }

    # 블록별 필터 개수 (MODEL_ID에 따라 가져오기)
    filter_counts_Ni_original = get_filter_counts(MODEL_ID)

    # 블록별 파라미터 수 (전략 2, 3에서 사용)
    all_blocks = find_prunable_blocks(base_model, MODEL_ID)
    param_counts_Ni = {
        name: count_parameters(block) for name, block in all_blocks.items()
    }
    total_block_params = sum(param_counts_Ni.values())

    for i in range(1, n_rounds + 1):
        print(f"\n===== 라운드 {i}/{n_rounds} =====")

        # 3) 전략에 따라 dynamic_ratios 계산
        if strategy == "vanilla":
            dynamic_ratios = compute_dynamic_ratios_vanilla(
                round_idx=i,
                n_rounds=n_rounds,
                sensitivity_si=sensitivity_si,
                filter_counts_Ni_original=filter_counts_Ni_original,
                global_target_ratio=GLOBAL_PRUNING_TARGET_RATIO,
            )
        elif strategy == "p":
            dynamic_ratios = compute_dynamic_ratios_p(
                round_idx=i,
                n_rounds=n_rounds,
                sensitivity_si=sensitivity_si,
                param_counts_Ni=param_counts_Ni,
                total_block_params=total_block_params,
                global_target_ratio=GLOBAL_PRUNING_TARGET_RATIO,
                p=p,
            )
        elif strategy == "beta":
            dynamic_ratios = compute_dynamic_ratios_beta(
                round_idx=i,
                n_rounds=n_rounds,
                sensitivity_si=sensitivity_si,
                param_counts_Ni=param_counts_Ni,
                total_block_params=total_block_params,
                global_target_ratio=GLOBAL_PRUNING_TARGET_RATIO,
                beta=beta,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # 4) dynamic_ratios → 남길 필터 인덱스 계산
        global_keep_indices = {}
        print("\n[이번 라운드에 적용할 프루닝 인덱스 계산]")
        for block_name in sorted(filter_counts_Ni_original.keys()):
            ratio_to_prune = dynamic_ratios.get(block_name, 0.0)
            original_C = filter_counts_Ni_original[block_name]
            num_to_keep = int(original_C * (1.0 - ratio_to_prune / 100.0))
            num_to_keep = max(1, num_to_keep)

            # 현재 pruned_model 기준으로 중요도 계산
            all_modules_current = find_prunable_blocks(pruned_model, MODEL_ID)
            block = all_modules_current[block_name]
            imp = block.bn1.weight.data.abs().cpu()

            order = imp.argsort(descending=True)
            keep_idx_target = sorted(order[:num_to_keep].tolist())
            global_keep_indices[block_name] = keep_idx_target
            print(f"블록: {block_name:<15} | 원본 필터: {original_C} -> 남길 필터: {len(keep_idx_target)}")

        # 5) 실제 프루닝 적용
        if MODEL_ID == "resnet18":
            pruned_model = prune_resnet_blockwise(pruned_model, global_keep_indices, device)
        else:
            raise NotImplementedError("현재 프루닝은 resnet18만 구현되어 있음")

        # 6) 라운드별 파인튜닝
        optimizer = optim.SGD(
            pruned_model.parameters(), lr=0.01,
            momentum=0.9, weight_decay=5e-4
        )
        criterion = nn.CrossEntropyLoss()

        print(f"\n라운드 {i} Fine-tune 시작 (총 {finetune_epochs_per_round} 에포크)")
        for ep in range(1, finetune_epochs_per_round + 1):
            train_one_epoch(pruned_model, trainloader, optimizer, criterion, ep)

        # 7) 중간 평가 및 민감도 업데이트
        print(f"\n--- 라운드 {i} 중간 평가 (Validation Set) ---")
        current_accuracy = test(pruned_model, validationloader)
        print(f"Validation Accuracy: {current_accuracy:.2f}%")
        accuracy_drop = max(0, last_accuracy - current_accuracy)
        print(f"이전 정확도: {last_accuracy:.2f}% -> 현재 정확도: {current_accuracy:.2f}% (하락폭: {accuracy_drop:.2f}%)")

        update_value = accuracy_drop * alpha
        if update_value > 0:
            print(f"민감도 스코어를 {update_value:.4f} 만큼 업데이트합니다.")
            for block_name in sensitivity_si:
                sensitivity_si[block_name] += update_value

        last_accuracy = current_accuracy

    # 8) 최종 재학습
    print("\n모든 프루닝 완료. 최종 재학습을 시작합니다.")
    FINAL_RETRAIN_EPOCHS = final_finetune_epochs

    optimizer = optim.SGD(
        pruned_model.parameters(), lr=0.005,
        momentum=0.9, weight_decay=5e-4
    )
    criterion = nn.CrossEntropyLoss()
    for ep in range(1, FINAL_RETRAIN_EPOCHS + 1):
        train_one_epoch(pruned_model, trainloader, optimizer, criterion, ep)

    # 9) 최종 결과 출력
    print("\n--- 최종 경량화 모델 성능 요약 ---")
    final_pruned_params = count_parameters(pruned_model)
    final_compression_rate = (1 - final_pruned_params / base_params) * 100
    print(f"최종 압축률: {final_compression_rate:.2f}%")

    print("\n--- 최종 Validation Accuracy ---")
    final_val_acc = test(pruned_model, validationloader)
    print(f"Validation Accuracy: {final_val_acc:.2f}%")

    print("\n--- 최종 Test Accuracy ---")
    final_test_acc = test(pruned_model, testloader)
    print(f"Test Accuracy: {final_test_acc:.2f}%")


# -------------------------------
# 7) 스크립트 실행
# -------------------------------
if __name__ == "__main__":
    # 전략 1: p / beta 없는 기본 버전
    main_adaptive_iterative_pruning(
        n_rounds=10,
        finetune_epochs_per_round=5,
        final_finetune_epochs=10,
        alpha=0.1,
        strategy="vanilla",
        GLOBAL_PRUNING_TARGET_RATIO=60.0,
    )

    # 전략 2 예시: p 전략 실험
    # main_adaptive_iterative_pruning(
    #     n_rounds=10,
    #     finetune_epochs_per_round=5,
    #     final_finetune_epochs=20,
    #     alpha=0.1,
    #     strategy="p",
    #     p=2.5,
    #     GLOBAL_PRUNING_TARGET_RATIO=80.0,
    # )

    # 전략 3 예시: beta 전략 실험
    # main_adaptive_iterative_pruning(
    #     n_rounds=10,
    #     finetune_epochs_per_round=5,
    #     final_finetune_epochs=20,
    #     alpha=0.1,
    #     strategy="beta",
    #     beta=0.8,
    #     GLOBAL_PRUNING_TARGET_RATIO=80.0,
    # )
