import re
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"Could not find pattern for {label}")
    return text.replace(old, new, 1)


def insert_after(text: str, marker: str, insertion: str, label: str) -> str:
    if insertion in text:
        return text
    if marker not in text:
        raise SystemExit(f"Could not find marker for {label}")
    return text.replace(marker, marker + insertion, 1)


def ensure_default(text: str, anchor: str, key_line: str, label: str) -> str:
    key = key_line.split(":", 1)[0].strip()
    if key in text:
        return text
    return replace_once(text, anchor, anchor + key_line, label)


def patch_main(repo: Path) -> None:
    path = repo / "main.py"
    text = path.read_text()

    anchor = '    "boundary_compensation_channel_ratio": 1.0,\n'
    text = ensure_default(text, anchor, '    "boundary_compensation_rep_weight": 1.0,\n', "default representation weight")
    text = ensure_default(text, anchor, '    "boundary_compensation_function_weight": 0.5,\n', "default function weight")
    text = ensure_default(text, anchor, '    "boundary_compensation_kd_weight": 0.0,\n', "default KD weight")
    text = ensure_default(text, anchor, '    "boundary_compensation_kd_temperature": 2.0,\n', "default KD temperature")

    if 'choices=["affine", "low_rank_residual"]' in text:
        text = text.replace(
            'choices=["affine", "low_rank_residual"]',
            'choices=["affine", "low_rank_residual", "mlp_residual"]',
            1,
        )

    arg_anchor = '    parser.add_argument("--boundary-compensation-gamma", dest="boundary_compensation_gamma", type=float, default=None)\n'
    arg_block = (
        '    parser.add_argument("--boundary-compensation-rep-weight", dest="boundary_compensation_rep_weight", type=float, default=None)\n'
        '    parser.add_argument("--boundary-compensation-function-weight", dest="boundary_compensation_function_weight", type=float, default=None)\n'
        '    parser.add_argument("--boundary-compensation-kd-weight", dest="boundary_compensation_kd_weight", type=float, default=None)\n'
        '    parser.add_argument("--boundary-compensation-kd-temperature", dest="boundary_compensation_kd_temperature", type=float, default=None)\n'
    )
    if "--boundary-compensation-rep-weight" not in text:
        text = insert_after(text, arg_anchor, arg_block, "teacher-student compensation args")

    override_marker = '        "boundary_compensation_gamma",\n'
    override_block = (
        '        "boundary_compensation_rep_weight",\n'
        '        "boundary_compensation_function_weight",\n'
        '        "boundary_compensation_kd_weight",\n'
        '        "boundary_compensation_kd_temperature",\n'
    )
    if '"boundary_compensation_rep_weight",' not in text:
        text = insert_after(text, override_marker, override_block, "teacher-student config overrides")

    call_anchor = '                    gamma=float(config.get("boundary_compensation_gamma", 1.0)),\n'
    call_block = (
        '                    rep_weight=float(config.get("boundary_compensation_rep_weight", 1.0)),\n'
        '                    function_weight=float(config.get("boundary_compensation_function_weight", 0.5)),\n'
        '                    kd_weight=float(config.get("boundary_compensation_kd_weight", 0.0)),\n'
        '                    kd_temperature=float(config.get("boundary_compensation_kd_temperature", 2.0)),\n'
    )
    if "rep_weight=float(config.get(" not in text:
        text = insert_after(text, call_anchor, call_block, "teacher-student estimate kwargs")

    path.write_text(text)


MLP_CLASS = r'''

class BoundaryMLPResidualWrapper(nn.Module):
    def __init__(self, block, fc1_weight, fc1_bias, fc2_weight, fc2_bias, gamma=1.0):
        super().__init__()
        self.block = block
        self.gamma = float(gamma)
        device = fc1_weight.device
        self.fc1 = nn.Linear(fc1_weight.shape[1], fc1_weight.shape[0], bias=fc1_bias is not None, device=device)
        self.fc2 = nn.Linear(fc2_weight.shape[1], fc2_weight.shape[0], bias=fc2_bias is not None, device=device)
        self.fc1.weight.data.copy_(fc1_weight.detach().float())
        if fc1_bias is not None:
            self.fc1.bias.data.copy_(fc1_bias.detach().float())
        self.fc2.weight.data.copy_(fc2_weight.detach().float())
        if fc2_bias is not None:
            self.fc2.bias.data.copy_(fc2_bias.detach().float())

    def forward(self, hidden_states, *args, **kwargs):
        residual = self.fc2(torch.nn.functional.gelu(self.fc1(hidden_states.float()))).to(dtype=hidden_states.dtype)
        hidden_states = hidden_states + self.gamma * residual
        return self.block(hidden_states, *args, **kwargs)
'''


HELPER = r'''

def _detach_tree(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach_tree(item) for key, item in value.items()}
    return value


def _move_tree(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_tree(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tree(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_tree(item, device) for key, item in value.items()}
    return value


def _replace_hidden_arg(inputs, kwargs, hidden_states):
    inputs = tuple(inputs or ())
    kwargs = dict(kwargs or {})
    if inputs:
        return (hidden_states,) + inputs[1:], kwargs
    kwargs["hidden_states"] = hidden_states
    return (), kwargs


def _fit_mlp_residual_teacher_student(
    *,
    samples,
    target_block,
    channel_mask,
    rank=512,
    train_steps=400,
    lr=5.0e-4,
    gamma=1.0,
    rep_weight=1.0,
    function_weight=0.5,
    kd_weight=0.0,
    kd_temperature=2.0,
):
    if not samples:
        raise RuntimeError("Teacher-student compensation needs captured calibration samples.")

    first_source = samples[0]["source"]
    hidden_size = first_source.shape[-1]
    device = first_source.device
    rank = max(1, min(int(rank), hidden_size))
    train_steps = max(1, int(train_steps))
    rep_weight = float(rep_weight)
    function_weight = float(function_weight)
    kd_weight = float(kd_weight)
    kd_temperature = float(kd_temperature)

    if kd_weight != 0.0:
        raise NotImplementedError(
            "Final-logit KD requires a full dense/student model forward path. "
            "Set --boundary-compensation-kd-weight 0.0 for the boundary module training stage."
        )

    mask = channel_mask.detach().to(device=device, dtype=torch.bool)
    if not bool(mask.any()):
        mask = torch.ones(hidden_size, device=device, dtype=torch.bool)

    fc1 = nn.Linear(hidden_size, rank, bias=True, device=device, dtype=torch.float32)
    fc2 = nn.Linear(rank, hidden_size, bias=True, device=device, dtype=torch.float32)
    torch.nn.init.normal_(fc1.weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(fc1.bias)
    torch.nn.init.zeros_(fc2.weight)
    torch.nn.init.zeros_(fc2.bias)

    target_block_was_training = target_block.training
    target_block.eval()
    for param in target_block.parameters():
        param.requires_grad_(False)

    opt = torch.optim.AdamW(list(fc1.parameters()) + list(fc2.parameters()), lr=float(lr), weight_decay=0.0)
    final_rep = None
    final_function = None
    final_total = None

    with torch.enable_grad():
        for step in range(train_steps):
            sample = samples[step % len(samples)]
            source = sample["source"].detach().float()
            teacher_in = sample["target_in"].detach().float()
            teacher_out = sample["target_out"].detach().float()
            block_inputs = _move_tree(sample.get("target_inputs", ()), device)
            block_kwargs = _move_tree(sample.get("target_kwargs", {}), device)

            opt.zero_grad(set_to_none=True)
            residual = fc2(torch.nn.functional.gelu(fc1(source)))
            student_in = source + float(gamma) * residual
            loss_rep = (student_in[..., mask] - teacher_in[..., mask]).pow(2).mean()

            if function_weight != 0.0:
                call_inputs, call_kwargs = _replace_hidden_arg(block_inputs, block_kwargs, student_in.to(dtype=teacher_in.dtype))
                student_out = _hidden_from_block_output(target_block(*call_inputs, **call_kwargs)).float()
                loss_function = (student_out - teacher_out).pow(2).mean()
            else:
                loss_function = torch.zeros((), device=device, dtype=torch.float32)

            loss_kd = torch.zeros((), device=device, dtype=torch.float32)
            loss = rep_weight * loss_rep + function_weight * loss_function + kd_weight * loss_kd
            loss.backward()
            opt.step()

            final_rep = loss_rep.detach()
            final_function = loss_function.detach()
            final_total = loss.detach()

    if target_block_was_training:
        target_block.train()

    return {
        "fc1_weight": fc1.weight.detach().cpu(),
        "fc1_bias": fc1.bias.detach().cpu(),
        "fc2_weight": fc2.weight.detach().cpu(),
        "fc2_bias": fc2.bias.detach().cpu(),
        "adapter_loss": float(final_total.item()),
        "adapter_rep_loss": float(final_rep.item()),
        "adapter_function_loss": float(final_function.item()),
        "adapter_kd_loss": 0.0,
    }
'''


def patch_compensation(repo: Path) -> None:
    path = repo / "amcprune" / "compensation.py"
    text = path.read_text()

    if "class BoundaryMLPResidualWrapper" not in text:
        text = replace_once(text, "\n\ndef _hidden_from_block_output", MLP_CLASS + "\n\ndef _hidden_from_block_output", "MLP wrapper insertion")

    if "def _fit_mlp_residual_teacher_student" not in text:
        marker = "\n\ndef apply_boundary_affine_compensation"
        if "def _fit_mlp_residual(" in text:
            start = text.find("\ndef _fit_mlp_residual(")
            end = text.find(marker, start)
            if end == -1:
                raise SystemExit("Could not find end of existing _fit_mlp_residual")
            text = text[:start] + HELPER + text[end:]
        else:
            text = replace_once(text, marker, HELPER + marker, "teacher-student helper insertion")

    if "rep_weight=1.0" not in text.split("):", 1)[0]:
        text = replace_once(
            text,
            '    outlier_protect_ratio=0.2,\n):\n',
            '    outlier_protect_ratio=0.2,\n'
            '    rep_weight=1.0,\n'
            '    function_weight=0.5,\n'
            '    kd_weight=0.0,\n'
            '    kd_temperature=2.0,\n'
            '):\n',
            "estimate signature loss weights",
        )

    if "function_samples = []" not in text:
        text = replace_once(
            text,
            "    source_samples = []\n    target_samples = []\n",
            "    source_samples = []\n    target_samples = []\n    function_samples = []\n",
            "function sample buffer",
        )

    pre_hook_old = '''    def target_pre_hook(_module, inputs):
        if inputs:
            captured["target"] = inputs[0].detach().float()
'''
    pre_hook_new = '''    def target_pre_hook(_module, inputs, kwargs=None):
        kwargs = kwargs or {}
        if inputs:
            captured["target"] = inputs[0].detach().float()
        elif "hidden_states" in kwargs:
            captured["target"] = kwargs["hidden_states"].detach().float()
        captured["target_inputs"] = _detach_tree(inputs)
        captured["target_kwargs"] = _detach_tree(kwargs)
'''
    if "captured[\"target_inputs\"]" not in text:
        text = replace_once(text, pre_hook_old, pre_hook_new, "target pre-hook kwargs capture")

    hook_old = "    target_handle = blocks[target_index].register_forward_pre_hook(target_pre_hook)\n"
    hook_new = '''    try:
        target_handle = blocks[target_index].register_forward_pre_hook(target_pre_hook, with_kwargs=True)
    except TypeError:
        target_handle = blocks[target_index].register_forward_pre_hook(target_pre_hook)

    def target_output_hook(_module, _inputs, output):
        captured["target_output"] = _hidden_from_block_output(output).detach().float()

    target_output_handle = blocks[target_index].register_forward_hook(target_output_hook)
'''
    if "target_output_handle" not in text:
        text = replace_once(text, hook_old, hook_new, "target output hook")

    remove_old = '''    finally:
        source_handle.remove()
        target_handle.remove()
'''
    remove_new = '''    finally:
        source_handle.remove()
        target_handle.remove()
        target_output_handle.remove()
'''
    if "target_output_handle.remove()" not in text:
        text = replace_once(text, remove_old, remove_new, "target output hook removal")

    text = text.replace(
        'if str(compensation_type) == "low_rank_residual":\n                source_samples.append(source.detach())',
        'if str(compensation_type) in {"low_rank_residual", "mlp_residual"}:\n                source_samples.append(source.detach())',
    )
    text = text.replace(
        'if str(compensation_type) == "low_rank_residual":\n                target_samples.append(target.detach())',
        'if str(compensation_type) in {"low_rank_residual", "mlp_residual"}:\n                target_samples.append(target.detach())',
    )

    sample_anchor = '''            if source.shape != target.shape:
                continue
'''
    sample_block = '''            if str(compensation_type) == "mlp_residual" and "target_output" in captured:
                function_samples.append({
                    "source": captured["source"].detach(),
                    "target_in": captured["target"].detach(),
                    "target_out": captured["target_output"].detach(),
                    "target_inputs": captured.get("target_inputs", ()),
                    "target_kwargs": captured.get("target_kwargs", {}),
                })
'''
    if "function_samples.append" not in text:
        text = insert_after(text, sample_anchor, sample_block, "function sample capture")

    branch_marker = '''        result["lr"] = float(lr)
    else:
        result["compensation_type"] = "affine"
'''
    branch_new = '''        result["lr"] = float(lr)
    elif str(compensation_type) == "mlp_residual":
        adapter = _fit_mlp_residual_teacher_student(
            samples=function_samples,
            target_block=blocks[target_index],
            channel_mask=channel_mask,
            rank=rank,
            train_steps=train_steps,
            lr=lr,
            gamma=gamma,
            rep_weight=rep_weight,
            function_weight=function_weight,
            kd_weight=kd_weight,
            kd_temperature=kd_temperature,
        )
        result.update(adapter)
        result["mode"] = "boundary_mlp_residual_teacher_student"
        result["compensation_type"] = "mlp_residual"
        result["rank"] = int(rank)
        result["train_steps"] = int(train_steps)
        result["lr"] = float(lr)
        result["rep_weight"] = float(rep_weight)
        result["function_weight"] = float(function_weight)
        result["kd_weight"] = float(kd_weight)
        result["kd_temperature"] = float(kd_temperature)
    else:
        result["compensation_type"] = "affine"
'''
    if "boundary_mlp_residual_teacher_student" not in text:
        existing_mlp_branch = re.compile(
            r'    elif str\(compensation_type\) == "mlp_residual":\n'
            r'(?:        .*\n)+?'
            r'    else:\n'
            r'        result\["compensation_type"\] = "affine"\n',
        )
        if existing_mlp_branch.search(text):
            text = existing_mlp_branch.sub(branch_new, text, count=1)
        else:
            text = replace_once(text, branch_marker, branch_new, "teacher-student MLP branch")

    apply_old = '''    if compensation.get("compensation_type") == "low_rank_residual":
        down_weight = compensation["down_weight"].to(device=target_device)
        up_weight = compensation["up_weight"].to(device=target_device)
        up_bias = compensation.get("up_bias")
        if up_bias is not None:
            up_bias = up_bias.to(device=target_device)
        blocks[target_new_index] = BoundaryLowRankResidualWrapper(
            blocks[target_new_index],
            down_weight,
            up_weight,
            up_bias=up_bias,
            gamma=float(compensation.get("gamma", 1.0)),
        )
    else:
'''
    apply_new = '''    if compensation.get("compensation_type") == "low_rank_residual":
        down_weight = compensation["down_weight"].to(device=target_device)
        up_weight = compensation["up_weight"].to(device=target_device)
        up_bias = compensation.get("up_bias")
        if up_bias is not None:
            up_bias = up_bias.to(device=target_device)
        blocks[target_new_index] = BoundaryLowRankResidualWrapper(
            blocks[target_new_index],
            down_weight,
            up_weight,
            up_bias=up_bias,
            gamma=float(compensation.get("gamma", 1.0)),
        )
    elif compensation.get("compensation_type") == "mlp_residual":
        fc1_weight = compensation["fc1_weight"].to(device=target_device)
        fc1_bias = compensation.get("fc1_bias")
        fc2_weight = compensation["fc2_weight"].to(device=target_device)
        fc2_bias = compensation.get("fc2_bias")
        if fc1_bias is not None:
            fc1_bias = fc1_bias.to(device=target_device)
        if fc2_bias is not None:
            fc2_bias = fc2_bias.to(device=target_device)
        blocks[target_new_index] = BoundaryMLPResidualWrapper(
            blocks[target_new_index],
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
            gamma=float(compensation.get("gamma", 1.0)),
        )
    else:
'''
    if "BoundaryMLPResidualWrapper(" not in text:
        text = replace_once(text, apply_old, apply_new, "apply teacher-student MLP wrapper")

    if '"adapter_rep_loss":' not in text and '"adapter_loss": float(compensation.get("adapter_loss", 0.0)),' in text:
        text = text.replace(
            '"adapter_loss": float(compensation.get("adapter_loss", 0.0)),',
            '"adapter_loss": float(compensation.get("adapter_loss", 0.0)),\n'
            '        "adapter_rep_loss": float(compensation.get("adapter_rep_loss", 0.0)),\n'
            '        "adapter_function_loss": float(compensation.get("adapter_function_loss", 0.0)),\n'
            '        "adapter_kd_loss": float(compensation.get("adapter_kd_loss", 0.0)),',
            1,
        )

    path.write_text(text)


def main() -> None:
    repo = Path.cwd()
    if not (repo / "main.py").exists() or not (repo / "amcprune" / "compensation.py").exists():
        raise SystemExit("Run this script from the AMCPrune_rescomp repo root.")
    patch_main(repo)
    patch_compensation(repo)
    print("Patched teacher-student boundary compensation losses.")
    print("Run: python -m py_compile main.py amcprune/compensation.py")


if __name__ == "__main__":
    main()
