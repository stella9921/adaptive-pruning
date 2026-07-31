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

    if 'dest="boundary_compensation_selection"' not in text:
        text = replace_once(
            text,
            '    parser.add_argument("--boundary-compensation-channel-ratio", dest="boundary_compensation_channel_ratio", type=float, default=None)\n'
            '    parser.add_argument("--boundary-compensation-eps", dest="boundary_compensation_eps", type=float, default=None)\n',
            '    parser.add_argument("--boundary-compensation-channel-ratio", dest="boundary_compensation_channel_ratio", type=float, default=None)\n'
            '    parser.add_argument(\n'
            '        "--boundary-compensation-selection",\n'
            '        dest="boundary_compensation_selection",\n'
            '        choices=["resource_aware", "random"],\n'
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

    old_selection = (
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
    new_selection = (
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
    )
    if 'selection_name = "random"' not in text:
        text = replace_once(text, old_selection, new_selection, "channel selection block")

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
    print("Patched boundary compensation random channel selection.")
    print("Run: python -m py_compile main.py amcprune/compensation.py")


if __name__ == "__main__":
    main()
