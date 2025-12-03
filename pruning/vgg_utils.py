# pruning/vgg_utils.py
import copy
import torch.nn as nn


def prune_conv_vgg(conv: nn.Conv2d, keep_idx, in_keep_idx=None):
    """
    VGG Conv2d용 프루닝 함수.
    - keep_idx: 출력 채널 중 남길 인덱스 리스트
    - in_keep_idx: 입력 채널 중 남길 인덱스 (이전 conv에서의 keep_idx)
    """
    if not keep_idx:
        raise ValueError("Pruning produced zero output channels.")

    W = conv.weight.data.clone()
    B = conv.bias.data.clone() if conv.bias is not None else None

    # 입력 채널 먼저 줄이기 (이전 레이어에서 채널을 줄였을 경우)
    if in_keep_idx is not None:
        W = W[:, in_keep_idx, :, :]

    # 출력 채널 줄이기
    W = W[keep_idx, :, :, :]
    if B is not None:
        B = B[keep_idx]

    new_conv = nn.Conv2d(
        in_channels=W.shape[1],
        out_channels=W.shape[0],
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None),
    )
    new_conv.weight.data = W.clone()
    if B is not None:
        new_conv.bias.data = B.clone()

    return new_conv


def prune_vgg_blockwise(model, block_keep_indices, device, num_classes=100):
    """
    VGG16용 블록 단위 프루닝 함수.
    - block_keep_indices: { "features.0": [남길 채널 idx ...], ... }
    - conv 레이어들을 순서대로 돌면서:
        * 현재 conv의 out_channels를 prune
        * 다음 conv의 in_channels를 이전 conv keep_idx 기준으로 prune
    - 마지막 conv의 out_channels에 맞춰 classifier[0]을 재구성
    """
    model = copy.deepcopy(model).cpu()

    # 1) features 안의 Conv2d 레이어들 순서를 보장하기 위해 정렬
    conv_items = []
    for name, module in model.named_modules():
        # name 예시: "features.0", "features.2", ...
        if isinstance(module, nn.Conv2d) and name.startswith("features."):
            idx = int(name.split(".")[1])
            conv_items.append((idx, name, module))
    conv_items.sort(key=lambda x: x[0])  # idx 기준 오름차순 정렬

    prev_out_indices = None  # 이전 conv의 keep_idx (입력 채널 prune 용)
    last_conv_name = None

    for idx, full_name, conv in conv_items:
        # full_name: "features.0" 같은 형태
        keep_idx = block_keep_indices.get(full_name, list(range(conv.out_channels)))
        in_keep_idx = prev_out_indices

        new_conv = prune_conv_vgg(conv, keep_idx, in_keep_idx)

        # 실제 모델에 반영
        parent_name, child_name = full_name.split(".")  # "features", "0"
        parent = getattr(model, parent_name)
        parent[int(child_name)] = new_conv

        #prev_out_indices = list(range(len(keep_idx)))
        prev_out_indices = keep_idx

        last_conv_name = full_name

    # 2) 마지막 conv의 출력 채널 수에 맞춰 classifier[0] 재구성
    if last_conv_name is not None:
        parent_name, child_name = last_conv_name.split(".")
        last_conv = getattr(model, parent_name)[int(child_name)]
        last_out_channels = last_conv.out_channels
    else:
        # conv가 없을 리는 없지만, 방어 코드
        last_out_channels = 512

    old_classifier = model.classifier
    # old_classifier = [Linear, ReLU, Dropout, Linear, ReLU, Dropout, Linear] 가정
    if isinstance(old_classifier, nn.Sequential) and len(old_classifier) >= 7:
        hidden1 = old_classifier[0].out_features
        hidden2 = old_classifier[3].out_features
        out_features = old_classifier[-1].out_features
    else:
        # 혹시 구조가 다르면 기본 값 사용
        hidden1 = 4096
        hidden2 = 4096
        out_features = num_classes

    new_classifier = nn.Sequential(
        nn.Linear(last_out_channels, hidden1),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(hidden2, out_features),
    )
    model.classifier = new_classifier

    return model.to(device)
