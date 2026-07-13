import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights


# ----------------------------
# Model Loader
# ----------------------------
def get_mobilenet(name="mobilenet_v2", num_classes=100, pretrained=False):

    if "v2" in name:
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
    else:
        raise ValueError(f"Unsupported MobileNet variant: {name}")

    # CIFAR-100 대응
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    return model


# ----------------------------
# Blockwise Physical Pruning
# ----------------------------
def prune_mobilenet_blockwise(model, keep_indices_dict, device):

    import torch_pruning as tp

    model.eval()
    example_inputs = torch.randn(1, 3, 32, 32).to(device)

    DG = tp.DependencyGraph().build_dependency(
        model,
        example_inputs=example_inputs
    )

    for name, module in model.named_modules():

        if name in keep_indices_dict:

            keep_idx = keep_indices_dict[name]

            if isinstance(module, nn.Conv2d):

                pruning_group = DG.get_pruning_group(
                    module,
                    tp.prune_conv_out_channels,
                    idxs=keep_idx
                )
                pruning_group.prune()

            elif isinstance(module, nn.Linear):

                pruning_group = DG.get_pruning_group(
                    module,
                    tp.prune_linear_out_channels,
                    idxs=keep_idx
                )
                pruning_group.prune()

    return model
