from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"Could not find pattern for {label}")
    return text.replace(old, new, 1)


def insert_once(text: str, marker: str, insertion: str, label: str) -> str:
    if insertion in text:
        return text
    if marker not in text:
        raise SystemExit(f"Could not find marker for {label}")
    return text.replace(marker, marker + insertion, 1)


def ensure_default_key(text: str, anchor: str, key_line: str, label: str) -> str:
    key = key_line.split(":", 1)[0].strip()
    if key in text:
        return text
    return replace_once(text, anchor, anchor + key_line, label)


def patch_main(repo: Path) -> None:
    path = repo / "main.py"
    text = path.read_text()

    anchor = '    "boundary_compensation_channel_ratio": 1.0,\n'
    text = ensure_default_key(text, anchor, '    "boundary_compensation_type": "affine",\n', "default compensation type")
    text = ensure_default_key(text, anchor, '    "boundary_compensation_rank": 64,\n', "default low-rank rank")
    text = ensure_default_key(text, anchor, '    "boundary_compensation_train_steps": 200,\n', "default low-rank steps")
    text = ensure_default_key(text, anchor, '    "boundary_compensation_lr": 1.0e-3,\n', "default low-rank lr")
    text = ensure_default_key(text, anchor, '    "boundary_compensation_gamma": 1.0,\n', "default compensation gamma")
    text = ensure_default_key(text, anchor, '    "boundary_outlier_protect_ratio": 0.2,\n', "default outlier protect ratio")

    arg_anchor = '    parser.add_argument("--boundary-compensation-channel-ratio", dest="boundary_compensation_channel_ratio", type=float, default=None)\n'
    arg_lines = [
        (
            "--boundary-compensation-type",
            '    parser.add_argument("--boundary-compensation-type", dest="boundary_compensation_type", '
            'choices=["affine", "low_rank_residual"], default=None)\n',
        ),
        (
            "--boundary-compensation-rank",
            '    parser.add_argument("--boundary-compensation-rank", dest="boundary_compensation_rank", type=int, default=None)\n',
        ),
        (
            "--boundary-compensation-train-steps",
            '    parser.add_argument("--boundary-compensation-train-steps", dest="boundary_compensation_train_steps", type=int, default=None)\n',
        ),
        (
            "--boundary-compensation-lr",
            '    parser.add_argument("--boundary-compensation-lr", dest="boundary_compensation_lr", type=float, default=None)\n',
        ),
        (
            "--boundary-compensation-gamma",
            '    parser.add_argument("--boundary-compensation-gamma", dest="boundary_compensation_gamma", type=float, default=None)\n',
        ),
        (
            "--boundary-outlier-protect-ratio",
            '    parser.add_argument("--boundary-outlier-protect-ratio", dest="boundary_outlier_protect_ratio", type=float, default=None)\n',
        ),
    ]
    for option, line in reversed(arg_lines):
        if option not in text:
            text = insert_once(text, arg_anchor, line, f"{option} arg")

    override_marker = '    config = resolve_config(args, DEFAULT_CONFIG)\n'
    override_block = (
        '    for _key in [\n'
        '        "boundary_compensation_type",\n'
        '        "boundary_compensation_rank",\n'
        '        "boundary_compensation_train_steps",\n'
        '        "boundary_compensation_lr",\n'
        '        "boundary_compensation_gamma",\n'
        '        "boundary_outlier_protect_ratio",\n'
        '    ]:\n'
        '        _value = getattr(args, _key, None)\n'
        '        if _value is not None:\n'
        '            config[_key] = _value\n'
    )
    text = insert_once(text, override_marker, override_block, "low-rank compensation config overrides")

    call_anchor = '                    eps=float(config["boundary_compensation_eps"]),\n'
    call_block = (
        '                    compensation_type=config.get("boundary_compensation_type", "affine"),\n'
        '                    rank=int(config.get("boundary_compensation_rank", 64)),\n'
        '                    train_steps=int(config.get("boundary_compensation_train_steps", 200)),\n'
        '                    lr=float(config.get("boundary_compensation_lr", 1.0e-3)),\n'
        '                    gamma=float(config.get("boundary_compensation_gamma", 1.0)),\n'
        '                    outlier_protect_ratio=float(config.get("boundary_outlier_protect_ratio", 0.2)),\n'
    )
    if "compensation_type=config.get(" not in text:
        text = insert_once(text, call_anchor, call_block, "low-rank estimate kwargs")
    else:
        if "gamma=float(config.get(\"boundary_compensation_gamma\"" not in text:
            text = insert_once(text, call_anchor, '                    gamma=float(config.get("boundary_compensation_gamma", 1.0)),\n', "gamma estimate kwarg")
        if "outlier_protect_ratio=float(config.get(" not in text:
            text = insert_once(
                text,
                '                    gamma=float(config.get("boundary_compensation_gamma", 1.0)),\n',
                '                    outlier_protect_ratio=float(config.get("boundary_outlier_protect_ratio", 0.2)),\n',
                "outlier protect ratio estimate kwarg",
            )

    if '"repairable"' not in text:
        choice_updates = [
            (
                'choices=["mismatch_outlier", "mismatch", "outlier", "random", "all"]',
                'choices=["mismatch_outlier", "mismatch", "outlier", "repairable", "balanced", "protect_outlier_mismatch", "random", "all"]',
            ),
            (
                'choices=["mismatch_outlier", "mismatch", "mismatch_low", "outlier", "outlier_low", "random", "all"]',
                'choices=["mismatch_outlier", "mismatch", "mismatch_low", "outlier", "outlier_low", "repairable", "balanced", "protect_outlier_mismatch", "random", "all"]',
            ),
        ]
        for old, new in choice_updates:
            if old in text:
                text = text.replace(old, new, 1)
                break
        else:
            raise SystemExit("Could not find boundary compensation selection choices")

    # Physical pruning used to export only a custom state_dict file, which cannot
    # be consumed by lm_eval/AutoModel. Keep that diagnostic artifact, but also
    # emit normal Hugging Face weights and config from the current pruned model.
    physical_export = '''                if config.get("pruning_mode") in {"unit_physical", "depth_width_physical"}:
                    import torch
                    torch.save(
                        model.state_dict(),
                        os.path.join(export_dir, "unit_physical_state_dict.pt"),
                    )
                else:
                    model.save_pretrained(export_dir)
'''
    hf_physical_export = '''                if config.get("pruning_mode") in {"unit_physical", "depth_width_physical"}:
                    import torch
                    torch.save(
                        model.state_dict(),
                        os.path.join(export_dir, "unit_physical_state_dict.pt"),
                    )
                    model.save_pretrained(export_dir)
                else:
                    model.save_pretrained(export_dir)
'''
    if physical_export in text:
        text = text.replace(physical_export, hf_physical_export, 1)
    elif (
        'os.path.join(export_dir, "unit_physical_state_dict.pt"),\n'
        in text
        and "                    model.save_pretrained(export_dir)\n                else:\n                    model.save_pretrained(export_dir)\n" not in text
    ):
        text = insert_once(
            text,
            '                        os.path.join(export_dir, "unit_physical_state_dict.pt"),\n                    )\n',
            '                    model.save_pretrained(export_dir)\n',
            "HF export after physical state dict",
        )

    path.write_text(text)


LOW_RANK_CLASS = r'''

class BoundaryLowRankResidualWrapper(nn.Module):
    def __init__(self, block, down_weight, up_weight, up_bias=None, gamma=1.0):
        super().__init__()
        self.block = block
        self.gamma = float(gamma)
        device = down_weight.device
        self.down = nn.Linear(down_weight.shape[1], down_weight.shape[0], bias=False, device=device)
        self.up = nn.Linear(up_weight.shape[1], up_weight.shape[0], bias=up_bias is not None, device=device)
        self.down.weight.data.copy_(down_weight.detach().float())
        self.up.weight.data.copy_(up_weight.detach().float())
        if up_bias is not None:
            self.up.bias.data.copy_(up_bias.detach().float())

    def forward(self, hidden_states, *args, **kwargs):
        residual = self.up(torch.nn.functional.gelu(self.down(hidden_states.float()))).to(dtype=hidden_states.dtype)
        hidden_states = hidden_states + self.gamma * residual
        return self.block(hidden_states, *args, **kwargs)
'''


LOW_RANK_HELPER = r'''

def _fit_low_rank_residual(source_hidden, target_hidden, channel_mask, rank=64, train_steps=200, lr=1.0e-3, gamma=1.0):
    x = source_hidden.detach().float().reshape(-1, source_hidden.shape[-1])
    y = target_hidden.detach().float().reshape(-1, target_hidden.shape[-1])
    hidden_size = x.shape[-1]
    rank = max(1, min(int(rank), hidden_size))
    train_steps = max(1, int(train_steps))
    device = x.device

    down = nn.Linear(hidden_size, rank, bias=False, device=device, dtype=torch.float32)
    up = nn.Linear(rank, hidden_size, bias=True, device=device, dtype=torch.float32)
    torch.nn.init.normal_(down.weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(up.weight)
    torch.nn.init.zeros_(up.bias)

    mask = channel_mask.detach().to(device=device, dtype=torch.bool)
    if not bool(mask.any()):
        mask = torch.ones(hidden_size, device=device, dtype=torch.bool)

    opt = torch.optim.AdamW(list(down.parameters()) + list(up.parameters()), lr=float(lr), weight_decay=0.0)
    with torch.enable_grad():
        for _ in range(train_steps):
            opt.zero_grad(set_to_none=True)
            residual = up(torch.nn.functional.gelu(down(x)))
            pred = x + float(gamma) * residual
            loss = (pred[:, mask] - y[:, mask]).pow(2).mean()
            loss.backward()
            opt.step()

        residual = up(torch.nn.functional.gelu(down(x)))
        pred = x + float(gamma) * residual
        final_loss = (pred[:, mask] - y[:, mask]).pow(2).mean()

    return {
        "down_weight": down.weight.detach().cpu(),
        "up_weight": up.weight.detach().cpu(),
        "up_bias": up.bias.detach().cpu(),
        "adapter_loss": float(final_loss.item()),
    }
'''


LOW_RANK_ESTIMATE_BLOCK = r'''
    if str(compensation_type) == "low_rank_residual":
        if not source_samples or not target_samples:
            raise RuntimeError("Low-rank residual compensation needs captured source/target calibration activations.")
        adapter = _fit_low_rank_residual(
            torch.cat(source_samples, dim=0),
            torch.cat(target_samples, dim=0),
            channel_mask,
            rank=rank,
            train_steps=train_steps,
            lr=lr,
            gamma=gamma,
        )
        result.update(adapter)
        result["mode"] = "boundary_low_rank_residual"
        result["compensation_type"] = "low_rank_residual"
        result["rank"] = int(rank)
        result["train_steps"] = int(train_steps)
        result["lr"] = float(lr)
    else:
        result["compensation_type"] = "affine"
'''


def patch_compensation(repo: Path) -> None:
    path = repo / "amcprune" / "compensation.py"
    text = path.read_text()

    if "class BoundaryLowRankResidualWrapper" not in text:
        marker = "\n\ndef _hidden_from_block_output"
        text = replace_once(text, marker, LOW_RANK_CLASS + marker, "low-rank wrapper insertion")
    elif "device = down_weight.device" not in text:
        old_device_block = '''        self.down = nn.Linear(down_weight.shape[1], down_weight.shape[0], bias=False)
        self.up = nn.Linear(up_weight.shape[1], up_weight.shape[0], bias=up_bias is not None)
'''
        new_device_block = '''        device = down_weight.device
        self.down = nn.Linear(down_weight.shape[1], down_weight.shape[0], bias=False, device=device)
        self.up = nn.Linear(up_weight.shape[1], up_weight.shape[0], bias=up_bias is not None, device=device)
'''
        text = replace_once(text, old_device_block, new_device_block, "low-rank wrapper device update")

    if "def _fit_low_rank_residual" not in text:
        marker = "\n\ndef apply_boundary_affine_compensation"
        text = replace_once(text, marker, LOW_RANK_HELPER + marker, "low-rank helper insertion")
    elif "with torch.enable_grad():" not in text:
        old_loop = '''    opt = torch.optim.AdamW(list(down.parameters()) + list(up.parameters()), lr=float(lr), weight_decay=0.0)
    for _ in range(train_steps):
        opt.zero_grad(set_to_none=True)
        residual = up(torch.nn.functional.gelu(down(x)))
        pred = x + float(gamma) * residual
        loss = (pred[:, mask] - y[:, mask]).pow(2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        residual = up(torch.nn.functional.gelu(down(x)))
        pred = x + float(gamma) * residual
        final_loss = (pred[:, mask] - y[:, mask]).pow(2).mean()
'''
        new_loop = '''    opt = torch.optim.AdamW(list(down.parameters()) + list(up.parameters()), lr=float(lr), weight_decay=0.0)
    with torch.enable_grad():
        for _ in range(train_steps):
            opt.zero_grad(set_to_none=True)
            residual = up(torch.nn.functional.gelu(down(x)))
            pred = x + float(gamma) * residual
            loss = (pred[:, mask] - y[:, mask]).pow(2).mean()
            loss.backward()
            opt.step()

        residual = up(torch.nn.functional.gelu(down(x)))
        pred = x + float(gamma) * residual
        final_loss = (pred[:, mask] - y[:, mask]).pow(2).mean()
'''
        text = replace_once(text, old_loop, new_loop, "low-rank helper enable_grad update")

    if "compensation_type=\"affine\"" not in text:
        text = replace_once(
            text,
            '    gamma=1.0,\n):\n',
            '    gamma=1.0,\n'
            '    compensation_type="affine",\n'
            '    rank=64,\n'
            '    train_steps=200,\n'
            '    lr=1.0e-3,\n'
            '    outlier_protect_ratio=0.2,\n'
            '):\n',
            "estimate low-rank signature",
        )
    elif "outlier_protect_ratio=0.2" not in text:
        text = insert_once(
            text,
            '    lr=1.0e-3,\n',
            '    outlier_protect_ratio=0.2,\n',
            "estimate outlier protect ratio signature",
        )

    if "source_samples = []" not in text:
        text = replace_once(
            text,
            "    captured = {}\n",
            "    captured = {}\n    source_samples = []\n    target_samples = []\n",
            "low-rank calibration sample buffers",
        )

    if "source_samples.append(source.detach())" not in text:
        text = replace_once(
            text,
            "            if source.shape != target.shape:\n                continue\n",
            "            if source.shape != target.shape:\n                continue\n"
            "            if str(compensation_type) == \"low_rank_residual\":\n"
            "                source_samples.append(source.detach())\n"
            "                target_samples.append(target.detach())\n",
            "low-rank calibration sample capture",
        )

    # Normalize the successful estimate return through a local result variable before returning.
    success_return = '    return {\n        "enabled": True,\n'
    if success_return in text:
        text = text.replace(success_return, '    result = {\n        "enabled": True,\n', 1)

    if LOW_RANK_ESTIMATE_BLOCK.strip() not in text:
        marker = '        "beta_std": float(beta.std(unbiased=False).item()),\n    }\n'
        text = replace_once(
            text,
            marker,
            marker + LOW_RANK_ESTIMATE_BLOCK + "\n    return result\n",
            "reachable low-rank estimate branch",
        )

    # Older broken patches placed the low-rank branch after a return. Remove that unreachable duplicate.
    broken = (
        '\n    if str(compensation_type) == "low_rank_residual":\n'
        '        hidden_candidates = [\n'
    )
    if broken in text:
        start = text.find(broken)
        end = text.find("\n\ndef _fit_low_rank_residual", start)
        if end == -1:
            raise SystemExit("Could not remove unreachable low-rank branch")
        text = text[:start] + text[end:]

    if '"gamma": float(gamma),' not in text:
        text = replace_once(
            text,
            '        "channel_ratio": ratio,\n',
            '        "channel_ratio": ratio,\n        "gamma": float(gamma),\n',
            "estimate gamma field",
        )

    if 'selection_name = "repairable_mismatch"' not in text:
        marker = '''        elif selection_strategy == "mismatch_outlier":
            selected_score = mismatch_outlier_score
            selected_indices = torch.topk(selected_score, k=selected_count, largest=True).indices
            selection_name = "mismatch_outlier"
'''
        insertion = '''        elif selection_strategy == "repairable":
            selected_score = mismatch_score - float(outlier_weight) * outlier_score
            selected_indices = torch.topk(selected_score, k=selected_count, largest=True).indices
            selection_name = "repairable_mismatch"
        elif selection_strategy == "balanced":
            selected_score = mismatch_score - float(outlier_weight) * torch.abs(outlier_score)
            selected_indices = torch.topk(selected_score, k=selected_count, largest=True).indices
            selection_name = "balanced_mismatch"
        elif selection_strategy == "protect_outlier_mismatch":
            selected_score = mismatch_score
            protect_ratio = min(max(float(outlier_protect_ratio), 0.0), 1.0)
            protected_count = min(mismatch.numel() - 1, max(1, int(round(mismatch.numel() * protect_ratio))))
            protected_indices = torch.topk(outlier_score, k=protected_count, largest=True).indices
            candidate_mask = torch.ones_like(mismatch, dtype=torch.bool)
            candidate_mask[protected_indices] = False
            candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
            if candidate_indices.numel() <= selected_count:
                selected_indices = candidate_indices
            else:
                candidate_scores = selected_score[candidate_indices]
                local_indices = torch.topk(candidate_scores, k=selected_count, largest=True).indices
                selected_indices = candidate_indices[local_indices]
            selection_name = "protect_outlier_mismatch"
'''
        text = replace_once(text, marker, insertion + marker, "repairable selection strategies")

    old_protect_line = "            protected_count = min(selected_count, max(1, int(round(mismatch.numel() * float(channel_ratio)))))\n"
    if old_protect_line in text:
        text = text.replace(
            old_protect_line,
            "            protect_ratio = min(max(float(outlier_protect_ratio), 0.0), 1.0)\n"
            "            protected_count = min(mismatch.numel() - 1, max(1, int(round(mismatch.numel() * protect_ratio))))\n",
            1,
        )

    if '"outlier_protect_ratio": float(outlier_protect_ratio),' not in text:
        text = replace_once(
            text,
            '        "outlier_weight": float(outlier_weight),\n',
            '        "outlier_weight": float(outlier_weight),\n        "outlier_protect_ratio": float(outlier_protect_ratio),\n',
            "estimate outlier protect ratio metadata",
        )

    if "blocks[target_new_index] = BoundaryLowRankResidualWrapper" not in text:
        old = '    blocks[target_new_index] = BoundaryAffineWrapper(blocks[target_new_index], alpha, beta, gamma=float(compensation.get("gamma", 1.0)))\n'
        if old not in text:
            old = '    blocks[target_new_index] = BoundaryAffineWrapper(blocks[target_new_index], alpha, beta)\n'
        new = (
            '    if compensation.get("compensation_type") == "low_rank_residual":\n'
            '        down_weight = compensation["down_weight"].to(device=target_device)\n'
            '        up_weight = compensation["up_weight"].to(device=target_device)\n'
            '        up_bias = compensation.get("up_bias")\n'
            '        if up_bias is not None:\n'
            '            up_bias = up_bias.to(device=target_device)\n'
            '        blocks[target_new_index] = BoundaryLowRankResidualWrapper(\n'
            '            blocks[target_new_index],\n'
            '            down_weight,\n'
            '            up_weight,\n'
            '            up_bias=up_bias,\n'
            '            gamma=float(compensation.get("gamma", 1.0)),\n'
            '        )\n'
            '    else:\n'
            '        blocks[target_new_index] = BoundaryAffineWrapper(\n'
            '            blocks[target_new_index],\n'
            '            alpha,\n'
            '            beta,\n'
            '            gamma=float(compensation.get("gamma", 1.0)),\n'
            '        )\n'
        )
        if old in text:
            text = replace_once(text, old, new, "apply low-rank wrapper branch")
        elif 'if compensation.get("compensation_type") == "low_rank_residual":' in text:
            pass
        else:
            raise SystemExit("Could not find pattern for apply low-rank wrapper branch")

    if '"compensation_type": compensation.get("compensation_type", "affine"),' not in text:
        text = replace_once(
            text,
            '        "mode": compensation.get("mode", "boundary_affine_channelwise"),\n',
            '        "mode": compensation.get("mode", "boundary_affine_channelwise"),\n'
            '        "compensation_type": compensation.get("compensation_type", "affine"),\n',
            "applied compensation type metadata",
        )
    if '"rank": int(compensation.get("rank", 0)),' not in text:
        text = replace_once(
            text,
            '        "gamma": float(compensation.get("gamma", 1.0)),\n',
            '        "gamma": float(compensation.get("gamma", 1.0)),\n'
            '        "rank": int(compensation.get("rank", 0)),\n'
            '        "adapter_loss": float(compensation.get("adapter_loss", 0.0)),\n',
            "applied low-rank metadata",
        )

    path.write_text(text)


def main() -> None:
    repo = Path.cwd()
    if not (repo / "main.py").exists() or not (repo / "amcprune" / "compensation.py").exists():
        raise SystemExit("Run this script from the AMCPrune_rescomp repo root.")
    patch_main(repo)
    patch_compensation(repo)
    print("Patched low-rank residual boundary compensation.")
    print("Run: python -m py_compile main.py amcprune/compensation.py")


if __name__ == "__main__":
    main()
