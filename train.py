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

# 전략 함수들 (Post-training)
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

# 실행할 모델 ID 설정 ("resnet152", "resnet18", "vgg16", "efficientnet_b0")
MODEL_ID = "resnet152"

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
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

full_trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
train_size = int(0.9 * len(full_trainset))
val_size = len(full_trainset) - train_size
trainset, validationset = random_split(full_trainset, [train_size, val_size])
testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

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
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if USE_AMP:
            with torch.amp.autocast(device_type=device.type):
                out = model(x); loss = criterion(out, y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            out = model(x); loss = criterion(out, y); loss.backward(); optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch} | Loss: {total_loss / len(loader):.4f}")

@torch.no_grad()
def test(model, loader) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x); pred = out.argmax(1)
        correct += (pred == y).sum().item(); total += y.size(0)
    return 100.0 * correct / total

# ------------------------------------------------
# 3) Post-training Iterative Pruning (민감도 기반)
# ------------------------------------------------
def main_adaptive_iterative_pruning(
    n_rounds=10, finetune_epochs_per_round=5, final_finetune_epochs=10,
    alpha=0.1, strategy="vanilla", p=2.5, beta=0.5, GLOBAL_PRUNING_TARGET_RATIO=60.0,
):
    print(f"\n[Mode] Post-training Iterative Pruning | Strategy: {strategy}")
    base_model = build_model(MODEL_ID, num_classes=100).to(device)
    base_model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device))
    
    last_accuracy = test(base_model, testloader)
    pruned_model = copy.deepcopy(base_model)
    
    sensitivity_si = maybe_load_or_compute_sensitivity(
        MODEL_ID, CHECKPOINT_DIR, trainloader, testloader, device, [0.0, 0.2, 0.4, 0.6, 0.8], 3
    )

    filter_counts_Ni_original = get_filter_counts(MODEL_ID)
    blocks_base = find_prunable_blocks(base_model, MODEL_ID)
    param_counts_Ni = {n: count_parameters(b) for n, b in blocks_base.items()}
    total_block_params = sum(param_counts_Ni.values())

    for round_idx in range(1, n_rounds + 1):
        print(f"\n===== Round {round_idx}/{n_rounds} =====")
        if strategy == "vanilla":
            dynamic_ratios = compute_dynamic_ratios_vanilla(round_idx, n_rounds, sensitivity_si, filter_counts_Ni_original, GLOBAL_PRUNING_TARGET_RATIO)
        elif strategy == "p":
            dynamic_ratios = compute_dynamic_ratios_p(round_idx, n_rounds, sensitivity_si, param_counts_Ni, total_block_params, GLOBAL_PRUNING_TARGET_RATIO, p)
        elif strategy == "beta":
            dynamic_ratios = compute_dynamic_ratios_beta(round_idx, n_rounds, sensitivity_si, param_counts_Ni, total_block_params, GLOBAL_PRUNING_TARGET_RATIO, beta)

        blocks_current = find_prunable_blocks(pruned_model, MODEL_ID)
        global_keep_indices = {}

        for block_name, original_C in filter_counts_Ni_original.items():
            ratio = dynamic_ratios.get(block_name, 0.0)
            num_keep = max(1, int(original_C * (1 - ratio / 100.0)))
            block = blocks_current[block_name]
            
            if MODEL_ID == "resnet18": importance = block.bn1.weight.data.abs().cpu()
            elif MODEL_ID == "resnet152":
                w = block.conv2.weight.data; importance = w.view(w.size(0), -1).abs().sum(dim=1).cpu()
            elif MODEL_ID == "vgg16":
                w = block.weight.data; importance = w.abs().mean(dim=(1, 2, 3)).cpu()
            elif "efficientnet" in MODEL_ID:
                w = block.weight.data; importance = w.view(w.size(0), -1).abs().sum(dim=1).cpu()

            order = importance.argsort(descending=True)
            global_keep_indices[block_name] = sorted(order[:num_keep].tolist())

        if "resnet" in MODEL_ID:
            pruned_model = prune_resnet_blockwise(pruned_model, global_keep_indices, device)
        elif "vgg" in MODEL_ID:
            pruned_model = prune_vgg_blockwise(pruned_model, global_keep_indices, device, num_classes=100)
        elif "efficientnet" in MODEL_ID:
            pruned_model = prune_efficientnet_blockwise(pruned_model, global_keep_indices, device)

        optimizer = optim.SGD(pruned_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        for ep in range(1, finetune_epochs_per_round + 1):
            train_one_epoch(pruned_model, trainloader, optimizer, nn.CrossEntropyLoss(), ep)
        
        current_val_acc = test(pruned_model, validationloader)
        accuracy_drop = max(0.0, last_accuracy - current_val_acc)
        update_value = accuracy_drop * alpha
        if update_value > 0:
            for name in sensitivity_si.keys(): sensitivity_si[name] += update_value
        last_accuracy = current_val_acc
        print(f"Val Acc: {current_val_acc:.2f}%")

# ------------------------------------------------
# 4) In-training Adaptive History Pruning (Adagrad 스타일)
# ------------------------------------------------
def main_intraining_history_adaptive_pruning(
    total_epochs=60, 
    prune_at_epochs=[40], 
    GLOBAL_TARGET_PRUNE_RATIO=0.6 
):
    print(f"\n[Mode] Adagrad-style Adaptive History Pruning | Global Target: {GLOBAL_TARGET_PRUNE_RATIO*100}%")
    
    model = build_model(MODEL_ID, num_classes=100).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    blocks = find_prunable_blocks(model, MODEL_ID)
    
    # 기울기 제곱 누적 히스토리 (학습 영향력 판단)
    grad_history = {}
    for name, blk in blocks.items():
        if "152" in MODEL_ID: n_f = blk.conv2.weight.shape[0]
        elif "resnet" in MODEL_ID: n_f = blk.bn1.weight.shape[0]
        else: n_f = blk.weight.shape[0] # VGG, EfficientNet
        grad_history[name] = torch.zeros(n_f).to(device)

    for epoch in range(1, total_epochs + 1):
        model.train()
        for x, y in trainloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if USE_AMP:
                with torch.amp.autocast(device_type=device.type):
                    out = model(x); loss = criterion(out, y)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                out = model(x); loss = criterion(out, y); loss.backward(); optimizer.step()

            # 필터별 기울기 제곱(g^2) 누적
            with torch.no_grad():
                for name, blk in blocks.items():
                    target_grad = blk.conv2.weight.grad if "152" in MODEL_ID else (blk.weight.grad if hasattr(blk, 'weight') else None)
                    if target_grad is not None:
                        g_sq = target_grad.pow(2).view(target_grad.shape[0], -1).mean(dim=1)
                        grad_history[name] += g_sq

        print(f"Epoch [{epoch}/{total_epochs}] Val Acc: {test(model, validationloader):.2f}%")

        if epoch in prune_at_epochs:
            print(f"\n>>> [Adaptive Pruning Event] Global Thresholding 적용")
            
            all_scores = torch.cat([s for s in grad_history.values()])
            threshold = torch.sort(all_scores)[0][int(len(all_scores) * GLOBAL_TARGET_PRUNE_RATIO)].item()
            
            keep_indices = {}
            for name, score in grad_history.items():
                keep_idx = (score > threshold).nonzero(as_tuple=True)[0].tolist()
                if not keep_idx: keep_idx = [score.argmax().item()]
                keep_indices[name] = sorted(keep_idx)
                print(f"Block: {name:20s} | Ratio: {(1 - len(keep_idx)/score.numel())*100:.1f}%")

            if "resnet" in MODEL_ID: model = prune_resnet_blockwise(model, keep_indices, device)
            elif "vgg" in MODEL_ID: model = prune_vgg_blockwise(model, keep_indices, device, 100)
            elif "efficientnet" in MODEL_ID: model = prune_efficientnet_blockwise(model, keep_indices, device)
            
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            blocks = find_prunable_blocks(model, MODEL_ID)
            print(f"Pruning 완료. 파라미터 수: {count_parameters(model):,}\n")

    print(f"Final Test Acc: {test(model, testloader):.2f}%")

# ------------------------------------------------
# 5) 실행부
# ------------------------------------------------
if __name__ == "__main__":
    # main_adaptive_iterative_pruning(strategy="p", p=2.0, GLOBAL_PRUNING_TARGET_RATIO=80.0)
    
    main_intraining_history_adaptive_pruning(total_epochs=60, prune_at_epochs=[40], GLOBAL_TARGET_PRUNE_RATIO=0.6)