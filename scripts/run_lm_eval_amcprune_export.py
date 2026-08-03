#!/usr/bin/env python3
"""Evaluate AMCPrune exported models with boundary wrapper weights intact.

Hugging Face AutoModel cannot reconstruct custom boundary wrappers from an
exported state dict whose keys look like ``model.layers.19.block.*`` and
``model.layers.19.up/down.*``. This script rebuilds those wrappers before
loading the saved weights, then passes the initialized model to lm-eval.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class BoundaryLowRankResidualWrapper(nn.Module):
    def __init__(self, block: nn.Module, down_weight: torch.Tensor, up_weight: torch.Tensor, up_bias: torch.Tensor | None, gamma: float = 1.0):
        super().__init__()
        self.block = block
        self.gamma = float(gamma)
        self.down = nn.Linear(down_weight.shape[1], down_weight.shape[0], bias=False)
        self.up = nn.Linear(up_weight.shape[1], up_weight.shape[0], bias=up_bias is not None)

    def forward(self, hidden_states, *args, **kwargs):
        residual = self.up(torch.nn.functional.gelu(self.down(hidden_states.float()))).to(dtype=hidden_states.dtype)
        hidden_states = hidden_states + self.gamma * residual
        return self.block(hidden_states, *args, **kwargs)


class BoundaryMLPResidualWrapper(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        fc1_weight: torch.Tensor,
        fc1_bias: torch.Tensor | None,
        fc2_weight: torch.Tensor,
        fc2_bias: torch.Tensor | None,
        channel_mask: torch.Tensor | None = None,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.block = block
        self.gamma = float(gamma)
        self.fc1 = nn.Linear(fc1_weight.shape[1], fc1_weight.shape[0], bias=fc1_bias is not None)
        self.fc2 = nn.Linear(fc2_weight.shape[1], fc2_weight.shape[0], bias=fc2_bias is not None)
        if channel_mask is None:
            channel_mask = torch.ones(fc2_weight.shape[0], dtype=torch.float32)
        self.register_buffer("channel_mask", channel_mask.float().view(1, 1, -1), persistent=True)

    def forward(self, hidden_states, *args, **kwargs):
        residual = self.fc2(torch.nn.functional.gelu(self.fc1(hidden_states.float()))).to(dtype=hidden_states.dtype)
        residual = residual * self.channel_mask.to(device=residual.device, dtype=residual.dtype)
        hidden_states = hidden_states + self.gamma * residual
        return self.block(hidden_states, *args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "--pretrained", dest="model_path", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", default="1")
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--num-fewshot", dest="num_fewshot", type=int, default=None)
    parser.add_argument("--confirm-run-unsafe-code", "--confirm_run_unsafe_code", action="store_true")
    return parser.parse_args()


def torch_dtype(name: str):
    return {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_state_dict(model_path: Path) -> dict[str, torch.Tensor]:
    index = model_path / "model.safetensors.index.json"
    files: list[Path]
    if index.exists():
        data = json.load(open(index))
        files = sorted({model_path / name for name in data["weight_map"].values()})
    else:
        files = sorted(model_path.glob("*.safetensors"))
    if files:
        state: dict[str, torch.Tensor] = {}
        for file in files:
            state.update(load_safetensors(str(file), device="cpu"))
        return state

    bins = sorted(model_path.glob("*.bin"))
    if not bins:
        raise FileNotFoundError(f"No safetensors/bin weights found in {model_path}")
    state = {}
    for file in bins:
        obj = torch.load(file, map_location="cpu")
        if isinstance(obj, dict) and "state_dict" in obj:
            obj = obj["state_dict"]
        state.update(obj)
    return state


def wrapper_layer_indices(state: dict[str, torch.Tensor], suffix: str) -> list[int]:
    prefix = "model.layers."
    out = []
    for key in state:
        if not key.startswith(prefix) or suffix not in key:
            continue
        rest = key[len(prefix) :]
        idx = rest.split(".", 1)[0]
        if idx.isdigit():
            out.append(int(idx))
    return sorted(set(out))


def read_gamma(model_path: Path, default: float = 1.0) -> float:
    cfg_path = model_path / "amcprune_pruning_config.json"
    if not cfg_path.exists():
        return default
    try:
        cfg = json.load(open(cfg_path))
    except Exception:
        return default
    for node in (cfg.get("boundary_compensation"), cfg.get("physical_pruning", {}).get("boundary_compensation")):
        if isinstance(node, dict) and "gamma" in node:
            return float(node["gamma"])
    return default


def rebuild_wrappers(model: nn.Module, state: dict[str, torch.Tensor], gamma: float) -> None:
    layers = model.model.layers
    for idx in wrapper_layer_indices(state, ".down.weight"):
        prefix = f"model.layers.{idx}."
        if prefix + "fc1.weight" in state:
            layers[idx] = BoundaryMLPResidualWrapper(
                layers[idx],
                state[prefix + "fc1.weight"],
                state.get(prefix + "fc1.bias"),
                state[prefix + "fc2.weight"],
                state.get(prefix + "fc2.bias"),
                state.get(prefix + "channel_mask"),
                gamma=gamma,
            )
        elif prefix + "up.weight" in state:
            layers[idx] = BoundaryLowRankResidualWrapper(
                layers[idx],
                state[prefix + "down.weight"],
                state[prefix + "up.weight"],
                state.get(prefix + "up.bias"),
                gamma=gamma,
            )


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    state = load_state_dict(model_path)

    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype(args.dtype))
    rebuild_wrappers(model, state, gamma=read_gamma(model_path))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("loaded", model_path)
    print("missing", len(missing), missing[:20])
    print("unexpected", len(unexpected), unexpected[:20])
    allowed_missing = {"lm_head.weight"}
    bad_missing = [key for key in missing if key not in allowed_missing]
    print("bad_missing", len(bad_missing), bad_missing[:20])
    if bad_missing or unexpected:
        raise RuntimeError("AMCPrune wrapper load was not clean; refusing to evaluate a partially initialized model.")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from lm_eval.models.huggingface import HFLM
    try:
        from lm_eval import simple_evaluate
    except ImportError:
        from lm_eval.evaluator import simple_evaluate

    model = model.to(args.device)
    model.eval()
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size, device=args.device)

    kwargs = {
        "model": lm,
        "tasks": [t.strip() for t in args.tasks.split(",") if t.strip()],
        "batch_size": args.batch_size,
        "limit": args.limit,
        "num_fewshot": args.num_fewshot,
    }
    sig = inspect.signature(simple_evaluate)
    if "confirm_run_unsafe_code" in sig.parameters:
        kwargs["confirm_run_unsafe_code"] = args.confirm_run_unsafe_code

    results = simple_evaluate(**kwargs)
    output = Path(args.output_path).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    out_file = output / "results.json"
    with open(out_file, "w") as f:
        json.dump(json_safe(results), f, indent=2)
    print("saved", out_file)


if __name__ == "__main__":
    main()
