# pruning/resnet_utils.py
import copy
import torch.nn as nn
import torchvision
from torchvision.models import resnet as _tv_resnet


def prune_conv_and_bn(conv, bn, keep_idx, in_keep_idx=None):
    """
    conv의 out_channel(=keep_idx)과, 필요하면 in_channel(in_keep_idx)까지 줄여서
    Conv2d + BatchNorm2d 쌍을 다시 만들어 주는 유틸.
    """
    if not keep_idx:
        raise ValueError("Pruning produced zero output channels.")

    W = conv.weight.data.clone()
    B = conv.bias.data.clone() if conv.bias is not None else None

    # out_channel 먼저 슬라이스
    W = W[keep_idx, :, :, :]
    if B is not None:
        B = B[keep_idx]

    # in_channel도 줄여야 할 때
    if in_keep_idx is not None:
        W = W[:, in_keep_idx, :, :]

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

    new_bn = nn.BatchNorm2d(len(keep_idx))
    new_bn.weight.data = bn.weight.data[keep_idx].clone()
    new_bn.bias.data = bn.bias.data[keep_idx].clone()
    new_bn.running_mean = bn.running_mean[keep_idx].clone()
    new_bn.running_var = bn.running_var[keep_idx].clone()

    return new_conv, new_bn


def prune_conv_in_channels(conv, in_keep_idx):
    """
    Bottleneck의 conv3처럼 out_channel은 그대로 두고,
    in_channel만 줄이고 싶을 때 쓰는 유틸.
    """
    if in_keep_idx is None:
        return conv

    W = conv.weight.data.clone()
    B = conv.bias.data.clone() if conv.bias is not None else None

    # in_channel 방향만 슬라이스
    W = W[:, in_keep_idx, :, :]

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


def prune_resnet_blockwise(model, block_keep_indices, device):
    """
    ResNet 계열 블록 단위 프루닝.
    - ResNet18: BasicBlock 기준, 기존 코드와 동일하게 block 출력 채널 자체를 줄임
    - ResNet152: Bottleneck 기준, conv2(3x3) 채널을 줄이고,
                 conv1/conv3 및 BN을 그에 맞게 재구성 (stage 출력 채널 수는 유지)
    """
    model = copy.deepcopy(model).cpu()

    # BasicBlock / Bottleneck 둘 다 잡아오도록 수정
    block_items = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, (_tv_resnet.BasicBlock, _tv_resnet.Bottleneck))
    ]

    # ResNet18(BasicBlock)에서만 사용되는 입력 채널 인덱스
    prev_out_indices = list(range(model.conv1.out_channels))

    for name, block in block_items:
        # ------------------------------------------------
        # 1) BasicBlock (ResNet18)
        # ------------------------------------------------
        if isinstance(block, _tv_resnet.BasicBlock):
            in_keep_idx = prev_out_indices
            keep_idx = block_keep_indices.get(
                name, list(range(block.conv1.out_channels))
            )

            # conv1 / bn1 : in = 이전 블록 출력, out = keep_idx
            new_conv1, new_bn1 = prune_conv_and_bn(
                block.conv1, block.bn1, keep_idx, in_keep_idx
            )
            block.conv1 = new_conv1
            block.bn1 = new_bn1

            # conv2 / bn2 : in/out 모두 keep_idx 기준
            new_conv2, new_bn2 = prune_conv_and_bn(
                block.conv2, block.bn2, keep_idx, keep_idx
            )
            block.conv2 = new_conv2
            block.bn2 = new_bn2

            # downsample도 같이 줄여줌
            if block.downsample is not None:
                ds_conv, ds_bn = block.downsample[0], block.downsample[1]
                new_ds_conv, new_ds_bn = prune_conv_and_bn(
                    ds_conv, ds_bn, keep_idx, in_keep_idx
                )
                block.downsample = nn.Sequential(new_ds_conv, new_ds_bn)
            elif len(in_keep_idx) != len(keep_idx):
                # 기존엔 identity였는데 채널 수가 바뀐 경우, 1x1 conv로 보정
                block.downsample = nn.Sequential(
                    nn.Conv2d(
                        len(in_keep_idx),
                        len(keep_idx),
                        kernel_size=1,
                        stride=block.conv1.stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(len(keep_idx)),
                )

            prev_out_indices = keep_idx

        # ------------------------------------------------
        # 2) Bottleneck (ResNet152)
        # ------------------------------------------------
        elif isinstance(block, _tv_resnet.Bottleneck):
            # 여기서는 각 블록의 "중간 채널"(conv2 out_channels)을 줄이는 전략 사용
            C_mid = block.conv2.out_channels
            keep_idx = block_keep_indices.get(name, list(range(C_mid)))

            # conv1/bn1 : out_channel = C_mid 기준으로 pruning
            # 입력 채널(=stage의 출력 채널)은 그대로 두고 out만 줄임
            new_conv1, new_bn1 = prune_conv_and_bn(
                block.conv1, block.bn1, keep_idx, in_keep_idx=None
            )
            block.conv1 = new_conv1
            block.bn1 = new_bn1

            # conv2/bn2 : in/out 모두 keep_idx 기준으로 pruning
            new_conv2, new_bn2 = prune_conv_and_bn(
                block.conv2, block.bn2, keep_idx, keep_idx
            )
            block.conv2 = new_conv2
            block.bn2 = new_bn2

            # conv3 : out_channel(=stage 출력 채널 수)는 유지하고,
            # in_channel만 keep_idx 기준으로 줄임
            block.conv3 = prune_conv_in_channels(block.conv3, keep_idx)

            # Bottleneck에서는 stage 출력 채널 수(=bn3.num_features)를
            # 유지하므로 prev_out_indices는 건드리지 않음.

        else:
            # 혹시 다른 타입이 들어오면 그냥 넘어감 (현재는 없음)
            continue

    # 마지막 fc 재설정 (BasicBlock/ Bottleneck 모두 지원)
    last_block = block_items[-1][1]
    if isinstance(last_block, _tv_resnet.BasicBlock):
        final_in_features = last_block.bn2.num_features
    elif isinstance(last_block, _tv_resnet.Bottleneck):
        final_in_features = last_block.bn3.num_features
    else:
        final_in_features = model.fc.in_features

    model.fc = nn.Linear(final_in_features, 100)

    return model.to(device)
