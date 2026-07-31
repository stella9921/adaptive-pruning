from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"Could not find pattern for {label}")
    return text.replace(old, new, 1)


def patch_main(repo: Path) -> None:
    path = repo / "main.py"
    text = path.read_text()

    if '"boundary_compensation_selection": "resource_aware"' not in text:
        text = replace_once(
            text,
            '    "boundary_compensation_channel_ratio": 1.0,\n'
            '    "boundary_compensation_eps": 1.0e-6,\n',
            '    "boundary_compensation_channel_ratio": 1.0,\n'
            '    "boundary_compensation_selection": "resource_aware",\n'
            '    "boundary_compensation_eps": 1.0e-6,\n',
            "DEFAULT_CONFIG boundary_compensation_selection",
        )

    old_choices = 'choices=["resource_aware", "random"],'
    new_choices = 'choices=["resource_aware", "random", "mismatch", "outlier", "mismatch_outlier"],'
    if old_choices in text:
        text = text.replace(old_choices, new_choices, 1)
    elif 'dest="boundary_compensation_selection"' not in text:
        text = replace_once(
            text,
            '    parser.add_argument("--boundary-compensation-channel-ratio", dest="boundary_compensation_channel_ratio", type=float, default=None)\n'
            '    parser.add_argument("--boundary-compensation-eps", dest="boundary_compensation_eps", type=float, default=None)\n',
            '    parser.add_argument("--boundary-compensation-channel-ratio", dest="boundary_compensation_channel_ratio", type=float, default=None)\n'
            '    parser.add_argument(\n'
            '        "--boundary-compensation-selection",\n'
            '        dest="boundary_compensation_selection",\n'
            '        choices=["resource_aware", "random", "mismatch", "outlier", "mismatch_outlier"],\n'
            '        default=None,\n'
            '    )\n'
            '    parser.add_argument("--boundary-compensation-eps", dest="boundary_compensation_eps", type=float, default=None)\n',
            "argparse boundary_compensation_selection",
        )

    if 'config["boundary_compensation_selection"] = args.boundary_compensation_selection' not in text:
        anchors = [
            (
                '    if args.boundary_compensation_eps is not None:\n'
                '        config["boundary_compensation_eps"] = args.boundary_compensation_eps\n',
                '    if args.boundary_compensation_selection is not None:\n'
                '        config["boundary_compensation_selection"] = args.boundary_compensation_selection\n'
                '    if args.boundary_compensation_eps is not None:\n'
                '        config["boundary_compensation_eps"] = args.boundary_compensation_eps\n',
            ),
            (
                '    if args.boundary_compensation_channel_ratio is not None:\n'
                '        config["boundary_compensation_channel_ratio"] = args.boundary_compensation_channel_ratio\n',
                '    if args.boundary_compensation_channel_ratio is not None:\n'
                '        config["boundary_compensation_channel_ratio"] = args.boundary_compensation_channel_ratio\n'
                '    if args.boundary_compensation_selection is not None:\n'
                '        config["boundary_compensation_selection"] = args.boundary_compensation_selection\n',
            ),
        ]
        for old, new in anchors:
            if old in text:
                text = text.replace(old, new, 1)
                break
        else:
            raise SystemExit("Could not find config override block for boundary compensation args")

    if "selection_strategy=config.get(" not in text:
        text = replace_once(
            text,
            '                    memory_weight=float(config["memory_weight"]),\n'
            '                    eps=float(config["boundary_compensation_eps"]),\n',
            '                    memory_weight=float(config["memory_weight"]),\n'
            '                    selection_strategy=config.get("boundary_compensation_selection", "resource_aware"),\n'
            '                    eps=float(config["boundary_compensation_eps"]),\n',
            "estimate_boundary_affine_compensation call",
        )

    path.write_text(text)


def patch_compensation(repo: Path) -> None:
    path = repo / "amcprune" / "compensation.py"
    text = path.read_text()

    if 'selection_strategy="resource_aware"' not in text.split("):", 1)[0]:
        text = replace_once(
            text,
            '    memory_weight=0.25,\n'
            '    eps=1.0e-6,\n'
            '):\n',
            '    memory_weight=0.25,\n'
            '    selection_strategy="resource_aware",\n'
            '    eps=1.0e-6,\n'
            '):\n',
            "estimate_boundary_affine_compensation signature",
        )

    old_block = (
        '        if selection_strategy == "random":\n'
        '            generator = torch.Generator(device=objective.device)\n'
        '            generator.manual_seed(42)\n'
        '            selected_indices = torch.randperm(\n'
        '                objective.numel(),\n'
        '                device=objective.device,\n'
        '                generator=generator,\n'
        '            )[:selected_count]\n'
        '            selection_name = "random"\n'
        '        else:\n'
        '            selected_indices = torch.topk(objective, k=selected_count, largest=True).indices\n'
        '            selection_name = "lagrangian_boundary_mismatch_outlier_resource"\n'
    )
    new_block = (
        '        if selection_strategy == "random":\n'
        '            generator = torch.Generator(device=objective.device)\n'
        '            generator.manual_seed(42)\n'
        '            selected_indices = torch.randperm(\n'
        '                objective.numel(),\n'
        '                device=objective.device,\n'
        '                generator=generator,\n'
        '            )[:selected_count]\n'
        '            selection_name = "random"\n'
        '        elif selection_strategy == "mismatch":\n'
        '            selected_indices = torch.topk(mismatch, k=selected_count, largest=True).indices\n'
        '            selection_name = "mismatch_only"\n'
        '        elif selection_strategy == "outlier":\n'
        '            selected_indices = torch.topk(outlier_risk, k=selected_count, largest=True).indices\n'
        '            selection_name = "outlier_only"\n'
        '        elif selection_strategy == "mismatch_outlier":\n'
        '            mismatch_z = (mismatch - mismatch.mean()) / (mismatch.std(unbiased=False) + eps)\n'
        '            outlier_z = (outlier_risk - outlier_risk.mean()) / (outlier_risk.std(unbiased=False) + eps)\n'
        '            ablation_objective = mismatch_z + float(outlier_weight) * outlier_z\n'
        '            selected_indices = torch.topk(ablation_objective, k=selected_count, largest=True).indices\n'
        '            objective = ablation_objective\n'
        '            selection_name = "mismatch_outlier"\n'
        '        else:\n'
        '            selected_indices = torch.topk(objective, k=selected_count, largest=True).indices\n'
        '            selection_name = "lagrangian_boundary_mismatch_outlier_resource"\n'
    )

    if old_block in text:
        text = text.replace(old_block, new_block, 1)
    elif 'selection_strategy == "mismatch_outlier"' not in text:
        old_unpatched = (
            '    ratio = min(max(float(channel_ratio), 0.0), 1.0)\n'
            '    if ratio >= 1.0:\n'
            '        channel_mask = torch.ones_like(mismatch, dtype=torch.bool)\n'
            '    elif ratio <= 0.0:\n'
            '        channel_mask = torch.zeros_like(mismatch, dtype=torch.bool)\n'
            '    else:\n'
            '        selected_count = max(1, int(round(objective.numel() * ratio)))\n'
            '        selected_indices = torch.topk(objective, k=selected_count, largest=True).indices\n'
            '        channel_mask = torch.zeros_like(mismatch, dtype=torch.bool)\n'
            '        channel_mask[selected_indices] = True\n'
        )
        new_unpatched = (
            '    ratio = min(max(float(channel_ratio), 0.0), 1.0)\n'
            '    selection_strategy = str(selection_strategy or "resource_aware")\n'
            '    if ratio >= 1.0:\n'
            '        channel_mask = torch.ones_like(mismatch, dtype=torch.bool)\n'
            '        selection_name = "all_channels"\n'
            '    elif ratio <= 0.0:\n'
            '        channel_mask = torch.zeros_like(mismatch, dtype=torch.bool)\n'
            '        selection_name = "no_channels"\n'
            '    else:\n'
            '        selected_count = max(1, int(round(objective.numel() * ratio)))\n'
            + new_block +
            '        channel_mask = torch.zeros_like(mismatch, dtype=torch.bool)\n'
            '        channel_mask[selected_indices] = True\n'
        )
        text = replace_once(text, old_unpatched, new_unpatched, "channel selection block")

    text = text.replace(
        '"selection_objective": "lagrangian_boundary_mismatch_outlier_resource",',
        '"selection_objective": selection_name,',
        1,
    )

    path.write_text(text)


def main() -> None:
    repo = Path.cwd()
    if not (repo / "main.py").exists() or not (repo / "amcprune" / "compensation.py").exists():
        raise SystemExit("Run this script from the AMCPrune_rescomp repo root.")
    patch_main(repo)
    patch_compensation(repo)
    print("Patched boundary compensation ablation selections.")
    print("Run: python -m py_compile main.py amcprune/compensation.py")


if __name__ == "__main__":
    main()
