#!/usr/bin/env python3
"""Patch AMCPrune_rescomp for TAPER block-pruning experiments.

This script is intended to be run from the remote ``AMCPrune_rescomp`` root.
It fixes three experiment blockers/features:

1. Infer a boundary automatically for pure block pruning, so boundary
   compensation works when ``--pruning-mode block`` removes a contiguous layer
   interval.
2. Add soft mismatch/outlier-aware channel weighting strategies. These keep all
   channels in the representation loss, but weight high-risk channels more.
3. Keep the patch idempotent enough to re-run on each machine after ``git pull``.
"""

from __future__ import annotations

import re
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"Could not find pattern for {label}")
    return text.replace(old, new, 1)


def insert_after(text: str, marker: str, insertion: str, label: str) -> str:
    if insertion.strip() in text:
        return text
    if marker not in text:
        raise SystemExit(f"Could not find marker for {label}")
    return text.replace(marker, marker + insertion, 1)


def patch_main(repo: Path) -> None:
    path = repo / "main.py"
    text = path.read_text()

    # Expose soft channel strategies in argparse if choices are present.
    choice_patterns = [
        (
            '"mismatch_outlier", "mismatch", "mismatch_low", "outlier", "outlier_low", '
            '"repairable", "balanced", "protect_outlier_mismatch", "random", "all"'
        ),
        (
            '"mismatch_outlier", "mismatch", "outlier", "random", "all"'
        ),
    ]
    for old_choices in choice_patterns:
        if old_choices in text and "mismatch_outlier_soft" not in text:
            new_choices = old_choices.replace(
                '"random", "all"',
                '"mismatch_outlier_soft", "mismatch_soft", "outlier_soft", "random", "all"',
            )
            text = text.replace(old_choices, new_choices, 1)
            break

    # In pure block-pruning mode, the removed contiguous interval is stored in
    # selected_blocks, while depth_pruned_blocks is only populated by
    # depth-width modes. Boundary compensation needs the removed interval.
    if "boundary_depth_pruned_blocks = depth_pruned_blocks" not in text:
        marker = "        if config.get(\"boundary_compensation\"):\n"
        insertion = (
            "            boundary_depth_pruned_blocks = depth_pruned_blocks\n"
            "            if config.get(\"pruning_mode\") == \"block\" and not boundary_depth_pruned_blocks:\n"
            "                boundary_depth_pruned_blocks = selected_blocks\n"
            "            print(f\"[Boundary] using depth_pruned_blocks={boundary_depth_pruned_blocks}\", flush=True)\n"
        )
        text = insert_after(
            text,
            marker,
            insertion,
            "block pruning boundary depth block fallback",
        )

    text = text.replace(
        "                    depth_pruned_blocks=depth_pruned_blocks,\n",
        "                    depth_pruned_blocks=boundary_depth_pruned_blocks,\n",
    )

    path.write_text(text)


def patch_boundary_in_compensation(repo: Path) -> None:
    path = repo / "amcprune" / "compensation.py"
    text = path.read_text()

    boundary_patch = '''    if boundary is None:
        removed = []
        try:
            removed = [int(i) for i in depth_pruned_blocks]
        except Exception:
            removed = []
        removed = sorted(set(removed))
        if removed:
            boundary = (max(min(removed) - 1, 0), max(removed) + 1)
            print(f"[Boundary] inferred boundary={boundary} from depth_pruned_blocks={removed}", flush=True)
        else:
            raise RuntimeError(
                "Boundary compensation requested, but boundary=None and "
                "depth_pruned_blocks is empty. For block pruning, pass the "
                "removed contiguous block indices."
            )
'''

    if boundary_patch.strip() not in text:
        text = replace_once(
            text,
            "    source_index, target_index = boundary\n",
            boundary_patch + "    source_index, target_index = boundary\n",
            "boundary inference before unpack",
        )

    path.write_text(text)


def patch_low_rank_weighted_loss(repo: Path) -> None:
    path = repo / "amcprune" / "compensation.py"
    text = path.read_text()

    # Add an optional channel_weight argument to the simple low-rank trainer.
    text = text.replace(
        "def _fit_low_rank_residual(source_hidden, target_hidden, channel_mask, rank=64, train_steps=200, lr=1.0e-3, gamma=1.0):",
        "def _fit_low_rank_residual(source_hidden, target_hidden, channel_mask, rank=64, train_steps=200, lr=1.0e-3, gamma=1.0, channel_weight=None):",
    )

    weight_setup = '''    weight = None
    if channel_weight is not None:
        weight = channel_weight.detach().to(device=device, dtype=torch.float32)
        if weight.numel() != hidden_size:
            raise RuntimeError(f"channel_weight has {weight.numel()} entries, expected {hidden_size}")
        weight = weight[mask].view(1, -1)
        weight = weight / (weight.mean() + 1.0e-8)
'''
    if "if channel_weight is not None:" not in text and "def _fit_low_rank_residual" in text:
        text = insert_after(
            text,
            "    if not bool(mask.any()):\n        mask = torch.ones(hidden_size, device=device, dtype=torch.bool)\n",
            "\n" + weight_setup,
            "low-rank channel weight setup",
        )

    text = text.replace(
        "            loss = (pred[:, mask] - y[:, mask]).pow(2).mean()\n",
        "            diff = (pred[:, mask] - y[:, mask]).pow(2)\n"
        "            loss = (diff * weight).mean() if weight is not None else diff.mean()\n",
    )
    text = text.replace(
        "        final_loss = (pred[:, mask] - y[:, mask]).pow(2).mean()\n",
        "        diff = (pred[:, mask] - y[:, mask]).pow(2)\n"
        "        final_loss = (diff * weight).mean() if weight is not None else diff.mean()\n",
    )

    # If the teacher-student low-rank helper exists, make rep loss weighted too.
    text = text.replace(
        "def _fit_low_rank_residual_teacher_student(\n    *,\n    samples,\n    channel_mask,\n    target_block,",
        "def _fit_low_rank_residual_teacher_student(\n    *,\n    samples,\n    channel_mask,\n    target_block,\n    channel_weight=None,",
    )
    ts_weight_setup = '''    weight = None
    if channel_weight is not None:
        weight = channel_weight.detach().to(device=device, dtype=torch.float32)
        if weight.numel() != hidden_size:
            raise RuntimeError(f"channel_weight has {weight.numel()} entries, expected {hidden_size}")
        weight = weight[mask].view(1, 1, -1)
        weight = weight / (weight.mean() + 1.0e-8)
'''
    if "def _fit_low_rank_residual_teacher_student" in text and "weight = weight[mask].view(1, 1, -1)" not in text:
        text = insert_after(
            text,
            "    if not bool(mask.any()):\n        mask = torch.ones(hidden_size, device=device, dtype=torch.bool)\n",
            "\n" + ts_weight_setup,
            "teacher-student low-rank channel weight setup",
        )
    text = text.replace(
        "                loss_rep = (student_in[..., mask] - teacher_in[..., mask]).pow(2).mean()\n",
        "                diff_rep = (student_in[..., mask] - teacher_in[..., mask]).pow(2)\n"
        "                loss_rep = (diff_rep * weight).mean() if weight is not None else diff_rep.mean()\n",
    )

    path.write_text(text)


def patch_soft_selection(repo: Path) -> None:
    path = repo / "amcprune" / "compensation.py"
    text = path.read_text()

    # Define a default per-channel loss weight after the normalized scores exist.
    marker = (
        "    mismatch_outlier_score = mismatch_score + float(outlier_weight) * outlier_score\n"
    )
    if marker in text and "channel_weight = torch.ones_like(mismatch, dtype=torch.float32)" not in text:
        text = insert_after(
            text,
            marker,
            "    channel_weight = torch.ones_like(mismatch, dtype=torch.float32)\n",
            "default channel weight",
        )

    soft_branch = '''    if selection_strategy in {"mismatch_outlier_soft", "mismatch_soft", "outlier_soft"}:
        if selection_strategy == "mismatch_soft":
            selected_score = mismatch_score
        elif selection_strategy == "outlier_soft":
            selected_score = outlier_score
        else:
            selected_score = mismatch_outlier_score
        centered = selected_score - selected_score.mean()
        scaled = centered / (selected_score.std(unbiased=False) + eps)
        channel_weight = torch.nn.functional.softplus(scaled) + eps
        channel_weight = channel_weight / (channel_weight.mean() + eps)
        channel_mask = torch.ones_like(mismatch, dtype=torch.bool)
        selection_name = selection_strategy
    elif ratio >= 1.0 or selection_strategy == "all":
'''
    if soft_branch.strip() not in text and "elif ratio >= 1.0 or selection_strategy == \"all\":" in text:
        text = text.replace(
            "    if ratio >= 1.0 or selection_strategy == \"all\":\n",
            soft_branch,
            1,
        )

    # Pass channel_weight into low-rank trainers when present.
    text = re.sub(
        r"(            channel_mask,\n            rank=rank,\n)(?!            channel_weight=channel_weight,\n)",
        r"\1            channel_weight=channel_weight,\n",
        text,
        count=1,
    )
    text = re.sub(
        r"(            channel_mask=channel_mask,\n            target_block=blocks\[target_index\],\n)(?!            channel_weight=channel_weight,\n)",
        r"\1            channel_weight=channel_weight,\n",
        text,
        count=1,
    )

    # Clean up accidental duplicate keyword insertions from older patch versions.
    text = re.sub(
        r"(            channel_weight=channel_weight,\n)(?:            channel_weight=channel_weight,\n)+",
        r"\1",
        text,
    )

    if '"channel_weight_mean":' not in text and '"selection_objective": selection_name,' in text:
        text = text.replace(
            '"selection_objective": selection_name,',
            '"selection_objective": selection_name,\n'
            '        "channel_weight_mean": float(channel_weight.mean().item()),\n'
            '        "channel_weight_std": float(channel_weight.std(unbiased=False).item()),',
            1,
        )

    path.write_text(text)


def main() -> None:
    repo = Path.cwd()
    if not (repo / "main.py").exists() or not (repo / "amcprune" / "compensation.py").exists():
        raise SystemExit("Run this script from the AMCPrune_rescomp repo root.")

    patch_main(repo)
    patch_boundary_in_compensation(repo)
    patch_low_rank_weighted_loss(repo)
    patch_soft_selection(repo)

    print("Patched TAPER block boundary inference and soft channel weighting.")
    print("Run: python -m py_compile main.py amcprune/compensation.py")


if __name__ == "__main__":
    main()
