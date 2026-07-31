from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"Could not find pattern for {label}")
    return text.replace(old, new, 1)


def ensure_default_config(text: str) -> str:
    if '"boundary_compensation_selection":' in text:
        text = text.replace(
            '"boundary_compensation_selection": "resource_aware"',
            '"boundary_compensation_selection": "mismatch_outlier"',
        )
        return text

    return replace_once(
        text,
        '    "boundary_compensation_channel_ratio": 1.0,\n'
        '    "boundary_compensation_eps": 1.0e-6,\n',
        '    "boundary_compensation_channel_ratio": 1.0,\n'
        '    "boundary_compensation_selection": "mismatch_outlier",\n'
        '    "boundary_compensation_eps": 1.0e-6,\n',
        "DEFAULT_CONFIG boundary_compensation_selection",
    )


def patch_main(repo: Path) -> None:
    path = repo / "main.py"
    text = path.read_text()
    text = ensure_default_config(text)

    old_choices = [
        'choices=["resource_aware", "random"],',
        'choices=["resource_aware", "random", "mismatch", "outlier", "mismatch_outlier"],',
    ]
    for old in old_choices:
        if old in text:
            text = text.replace(
                old,
                'choices=["mismatch_outlier", "mismatch", "outlier", "random", "all"],',
                1,
            )
            break
    else:
        if 'dest="boundary_compensation_selection"' not in text:
            text = replace_once(
                text,
                '    parser.add_argument("--boundary-compensation-channel-ratio", dest="boundary_compensation_channel_ratio", type=float, default=None)\n'
                '    parser.add_argument("--boundary-compensation-eps", dest="boundary_compensation_eps", type=float, default=None)\n',
                '    parser.add_argument("--boundary-compensation-channel-ratio", dest="boundary_compensation_channel_ratio", type=float, default=None)\n'
                '    parser.add_argument(\n'
                '        "--boundary-compensation-selection",\n'
                '        dest="boundary_compensation_selection",\n'
                '        choices=["mismatch_outlier", "mismatch", "outlier", "random", "all"],\n'
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
            '                    selection_strategy=config.get("boundary_compensation_selection", "mismatch_outlier"),\n'
            '                    eps=float(config["boundary_compensation_eps"]),\n',
            "estimate_boundary_affine_compensation call",
        )
    else:
        text = text.replace(
            'selection_strategy=config.get("boundary_compensation_selection", "resource_aware")',
            'selection_strategy=config.get("boundary_compensation_selection", "mismatch_outlier")',
        )

    path.write_text(text)


def patch_compensation(repo: Path) -> None:
    path = repo / "amcprune" / "compensation.py"
    text = path.read_text()

    if 'selection_strategy=' not in text.split("):", 1)[0]:
        text = replace_once(
            text,
            '    memory_weight=0.25,\n'
            '    eps=1.0e-6,\n'
            '):\n',
            '    memory_weight=0.25,\n'
            '    selection_strategy="mismatch_outlier",\n'
            '    eps=1.0e-6,\n'
            '):\n',
            "estimate_boundary_affine_compensation signature",
        )
    else:
        text = text.replace(
            'selection_strategy="resource_aware"',
            'selection_strategy="mismatch_outlier"',
        )

    selection_block_candidates = [
        (
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
            '        channel_mask = torch.zeros_like(mismatch, dtype=torch.bool)\n'
            '        channel_mask[selected_indices] = True\n'
        ),
        (
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
        ),
    ]

    new_selection_block = (
        '    ratio = min(max(float(channel_ratio), 0.0), 1.0)\n'
        '    selection_strategy = str(selection_strategy or "mismatch_outlier")\n'
        '    mismatch_score = (mismatch - mismatch.mean()) / (mismatch.std(unbiased=False) + eps)\n'
        '    outlier_score = (outlier_risk - outlier_risk.mean()) / (outlier_risk.std(unbiased=False) + eps)\n'
        '    mismatch_outlier_score = mismatch_score + float(outlier_weight) * outlier_score\n'
        '    if ratio >= 1.0 or selection_strategy == "all":\n'
        '        channel_mask = torch.ones_like(mismatch, dtype=torch.bool)\n'
        '        selected_score = mismatch_outlier_score\n'
        '        selection_name = "all_channels"\n'
        '    elif ratio <= 0.0:\n'
        '        channel_mask = torch.zeros_like(mismatch, dtype=torch.bool)\n'
        '        selected_score = mismatch_outlier_score\n'
        '        selection_name = "no_channels"\n'
        '    else:\n'
        '        selected_count = max(1, int(round(mismatch.numel() * ratio)))\n'
        '        if selection_strategy == "random":\n'
        '            generator = torch.Generator(device=mismatch.device)\n'
        '            generator.manual_seed(42)\n'
        '            selected_indices = torch.randperm(\n'
        '                mismatch.numel(),\n'
        '                device=mismatch.device,\n'
        '                generator=generator,\n'
        '            )[:selected_count]\n'
        '            selected_score = mismatch_outlier_score\n'
        '            selection_name = "random"\n'
        '        elif selection_strategy == "mismatch":\n'
        '            selected_score = mismatch_score\n'
        '            selected_indices = torch.topk(selected_score, k=selected_count, largest=True).indices\n'
        '            selection_name = "mismatch_only"\n'
        '        elif selection_strategy == "outlier":\n'
        '            selected_score = outlier_score\n'
        '            selected_indices = torch.topk(selected_score, k=selected_count, largest=True).indices\n'
        '            selection_name = "outlier_only"\n'
        '        elif selection_strategy == "mismatch_outlier":\n'
        '            selected_score = mismatch_outlier_score\n'
        '            selected_indices = torch.topk(selected_score, k=selected_count, largest=True).indices\n'
        '            selection_name = "mismatch_outlier"\n'
        '        else:\n'
        '            raise ValueError(f"Unknown boundary compensation selection: {selection_strategy}")\n'
        '        channel_mask = torch.zeros_like(mismatch, dtype=torch.bool)\n'
        '        channel_mask[selected_indices] = True\n'
        '    objective = selected_score\n'
    )

    if 'selection_strategy == "mismatch_outlier"' not in text:
        for old_block in selection_block_candidates:
            if old_block in text:
                text = text.replace(old_block, new_selection_block, 1)
                break
        else:
            raise SystemExit("Could not find channel selection block in compensation.py")

    replacements = {
        '"selection_objective": "lagrangian_boundary_mismatch_outlier_resource",': '"selection_objective": selection_name,',
        '"resource_cost_per_channel": resource_cost_per_channel,': '"resource_cost_per_channel": None,',
        '"resource_cost_total": resource_cost_total,': '"resource_cost_total": None,',
    }
    for old, new in replacements.items():
        if old in text:
            text = text.replace(old, new)

    path.write_text(text)


def main() -> None:
    repo = Path.cwd()
    if not (repo / "main.py").exists() or not (repo / "amcprune" / "compensation.py").exists():
        raise SystemExit("Run this script from the AMCPrune_rescomp repo root.")
    patch_main(repo)
    patch_compensation(repo)
    print("Patched mismatch/outlier channel-selective compensation.")
    print("Run: python -m py_compile main.py amcprune/compensation.py")


if __name__ == "__main__":
    main()
