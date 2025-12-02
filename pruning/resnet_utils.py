# pruning/resnet_utils.py
import copy
import torch.nn as nn
import torchvision


def prune_conv_and_bn(conv, bn, keep_idx, in_keep_idx=None):
    if not keep_idx:
        raise ValueError("Pruning produced zero output channels.")

    W = conv.weight.data.clone()
    B = conv.bias.data.clone() if conv.bias is not None else None

    W = W[keep_idx, :, :, :]
    if B is not None:
        B = B[keep_idx]
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


def prune_resnet_blockwise(model, block_keep_indices, device):
    """
    ResNet18 전용 프루닝 함수 (기존 train.py 내용 그대로 옮긴 것 + device 인자 추가)
    """
    model = copy.deepcopy(model).cpu()
    block_items = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, torchvision.models.resnet.BasicBlock)
    ]

    prev_out_indices = list(range(model.conv1.out_channels))

    for name, block in block_items:
        in_keep_idx = prev_out_indices
        keep_idx = block_keep_indices.get(
            name, list(range(block.conv1.out_channels))
        )

        new_conv1, new_bn1 = prune_conv_and_bn(block.conv1, block.bn1, keep_idx, in_keep_idx)
        block.conv1 = new_conv1
        block.bn1 = new_bn1

        new_conv2, new_bn2 = prune_conv_and_bn(block.conv2, block.bn2, keep_idx, keep_idx)
        block.conv2 = new_conv2
        block.bn2 = new_bn2

        if block.downsample is not None:
            ds_conv, ds_bn = block.downsample[0], block.downsample[1]
            new_ds_conv, new_ds_bn = prune_conv_and_bn(ds_conv, ds_bn, keep_idx, in_keep_idx)
            block.downsample = nn.Sequential(new_ds_conv, new_ds_bn)
        elif len(in_keep_idx) != len(keep_idx):
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

    # 마지막 fc 재설정
    last_block = block_items[-1][1]
    final_in_features = last_block.bn2.num_features
    model.fc = nn.Linear(final_in_features, 100)

    return model.to(device)
