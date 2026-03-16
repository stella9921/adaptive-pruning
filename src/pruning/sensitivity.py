import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.models import get_model, find_prunable_blocks, get_prune_fn

# ------------------------------------------------------------
#  공통 유틸리티 함수
# ------------------------------------------------------------
@torch.no_grad()
def _test(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total

def _train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

# ------------------------------------------------------------
#  민감도 계산 엔진 (ResNet18, VGG16, ResNet152 통합)
# ------------------------------------------------------------
def compute_sensitivity(model_id, base_ckpt_path, trainloader, testloader, device, block_ratios=[0.0, 0.2, 0.4, 0.6, 0.8], finetune_epochs=3):
    """
    논문 단계 1: 각 레이어별로 Probing을 수행하여 민감도(Slope) 산출
    """
    # 모델 빌드 및 가중치 로드
    model = get_model({"name": model_id}).to(device)
    state = torch.load(base_ckpt_path, map_location=device)
    # model.load_state_dict(state)
    model.load_state_dict(state, strict=False)

    base_acc = _test(model, testloader, device)
    print(f"[{model_id}] Base Accuracy: {base_acc:.2f}%")

    blocks = find_prunable_blocks(model, model_id)
    block_names = sorted(blocks.keys())
    prune_fn = get_prune_fn(model_id)
    sensitivity_si = {}

    for blk_name in block_names:
        print(f"\n>> Analyzing Block: {blk_name}")
        ratio_to_acc = {0.0: base_acc}

        for ratio in block_ratios[1:]:
            print(f"   - Pruning {ratio*100:.0f}% ...", end=" ")
            
            # 임시 모델 복구
            tmp = get_model({"name": model_id}).to(device)
            tmp.load_state_dict(state)
            
            # 중요도 계산 및 인덱스 추출 (L1-norm)
            blocks_imp = find_prunable_blocks(tmp, model_id)
            target_blk = blocks_imp[blk_name]
            
            # 레이어 특성에 따른 가중치 추출 (ResNet18:bn1, VGG/ResNet152:weight)
            if model_id == "resnet18":
                imp = target_blk.bn1.weight.data.abs().cpu()
            else:
                w = target_blk.conv2.weight.data if hasattr(target_blk, 'conv2') else target_blk.weight.data
                imp = w.view(w.size(0), -1).abs().sum(dim=1).cpu()

            C = imp.numel()
            k = min(int(C * ratio), C - 1)
            keep = imp.argsort(descending=True)[:C-k].tolist()

            # 물리적 프루닝 (구조 변경)
            pruned = prune_fn(tmp, {blk_name: keep}, device)

            # 에폭 여러 번 돌려서 정확도 회복 (Fine-tuning)
            optimizer = optim.SGD(pruned.parameters(), lr=0.01, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            for _ in range(finetune_epochs):
                _train_one_epoch(pruned, trainloader, optimizer, criterion, device)

            acc = _test(pruned, testloader, device)
            ratio_to_acc[ratio] = acc
            print(f"Acc: {acc:.2f}%")

        # 기울기(Slope) 평균 계산 (수식의 기초 지표)
        slopes = [abs((ratio_to_acc[r2] - ratio_to_acc[r1]) / ((r2-r1)*100)) 
                  for r1, r2 in zip(block_ratios[:-1], block_ratios[1:])]
        sensitivity_si[blk_name] = float(np.mean(slopes))

    return sensitivity_si

# ------------------------------------------------------------
#  메인 호출 함수 (JSON 저장 로직 포함)
# ------------------------------------------------------------
def maybe_load_or_compute_sensitivity(config, trainloader, testloader, device):
    model_id = config['model']['name']
    checkpoint_dir = config['save_dir']
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    json_path = os.path.join(checkpoint_dir, f"{model_id}_sensitivity.json")
    base_ckpt_path = os.path.join(checkpoint_dir, f"{model_id}_base.pth")

    if os.path.exists(json_path):
        print(f"[Sensitivity] Load from {json_path}")
        with open(json_path, "r") as f: return json.load(f)

    # 민감도 계산 실행
    si = compute_sensitivity(
        model_id, base_ckpt_path, trainloader, testloader, device,
        block_ratios=[0.0, 0.2, 0.4, 0.6, 0.8],
        finetune_epochs=3
    )

    with open(json_path, "w") as f:
        json.dump(si, f, indent=2)
    
    print(f"[Sensitivity] Saved to {json_path}")
    return si