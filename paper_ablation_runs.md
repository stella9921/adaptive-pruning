# AMCPrune Ablation Runs

Goal: validate the channel-selection part of the method while keeping the depth-pruned backbone and pruning ratio fixed.

Main ablation table:

| Variant | Selection score | Purpose |
|---|---|---|
| random | random channels | checks whether affine capacity alone helps |
| mismatch-only | `m_c` | tests boundary distribution shift |
| outlier-only | `o_c` | tests activation-risk selection |
| mismatch+outlier | `Norm(m_c) + lambda_o Norm(o_c)` | proposed main variant |
| low-rank residual | selected channels + `H + U GELU(VH)` | tests whether compensation capacity, not only channel choice, recovers the pruned boundary |

Recommended first runs on 29/30:

```bash
# Random channel compensation
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python main.py \
  --strategy qwen_hidden_cosine_outlier \
  --model /home/stella/work/hf_models/Llama-3.2-3B \
  --dataset Salesforce/wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --dtype bf16 \
  --score hidden_cosine_streamline \
  --pruning-mode depth_width_physical \
  --pruning-ratio 0.25 \
  --unit-score none \
  --unit-pruning-ratio 0.0 \
  --width-candidate-count 0 \
  --max-samples 128 \
  --seq-len 1024 \
  --post-depth-recalibration \
  --exclude-depth-boundary-blocks \
  --boundary-compensation \
  --boundary-compensation-channel-ratio 0.5 \
  --boundary-compensation-max-batches 32 \
  --boundary-compensation-selection random \
  --outlier-weight 0.0 \
  --output-dir exp/runs/llama3p2_3b_d025_boundary_random_ch050_s1024 \
  > run_llama3p2_3b_d025_boundary_random_ch050_s1024.log 2>&1 &

# Mismatch-only channel compensation
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python main.py \
  --strategy qwen_hidden_cosine_outlier \
  --model /home/stella/work/hf_models/Llama-3.2-3B \
  --dataset Salesforce/wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --dtype bf16 \
  --score hidden_cosine_streamline \
  --pruning-mode depth_width_physical \
  --pruning-ratio 0.25 \
  --unit-score none \
  --unit-pruning-ratio 0.0 \
  --width-candidate-count 0 \
  --max-samples 128 \
  --seq-len 1024 \
  --post-depth-recalibration \
  --exclude-depth-boundary-blocks \
  --boundary-compensation \
  --boundary-compensation-channel-ratio 0.5 \
  --boundary-compensation-max-batches 32 \
  --boundary-compensation-selection mismatch \
  --outlier-weight 0.0 \
  --output-dir exp/runs/llama3p2_3b_d025_boundary_mismatch_ch050_s1024 \
  > run_llama3p2_3b_d025_boundary_mismatch_ch050_s1024.log 2>&1 &

# Outlier-only channel compensation
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python main.py \
  --strategy qwen_hidden_cosine_outlier \
  --model /home/stella/work/hf_models/Llama-3.2-3B \
  --dataset Salesforce/wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --dtype bf16 \
  --score hidden_cosine_streamline \
  --pruning-mode depth_width_physical \
  --pruning-ratio 0.25 \
  --unit-score none \
  --unit-pruning-ratio 0.0 \
  --width-candidate-count 0 \
  --max-samples 128 \
  --seq-len 1024 \
  --post-depth-recalibration \
  --exclude-depth-boundary-blocks \
  --boundary-compensation \
  --boundary-compensation-channel-ratio 0.5 \
  --boundary-compensation-max-batches 32 \
  --boundary-compensation-selection outlier \
  --outlier-weight 1.0 \
  --output-dir exp/runs/llama3p2_3b_d025_boundary_outlier_ch050_s1024 \
  > run_llama3p2_3b_d025_boundary_outlier_ch050_s1024.log 2>&1 &

# Proposed mismatch+outlier channel compensation
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python main.py \
  --strategy qwen_hidden_cosine_outlier \
  --model /home/stella/work/hf_models/Llama-3.2-3B \
  --dataset Salesforce/wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --dtype bf16 \
  --score hidden_cosine_streamline \
  --pruning-mode depth_width_physical \
  --pruning-ratio 0.25 \
  --unit-score none \
  --unit-pruning-ratio 0.0 \
  --width-candidate-count 0 \
  --max-samples 128 \
  --seq-len 1024 \
  --post-depth-recalibration \
  --exclude-depth-boundary-blocks \
  --boundary-compensation \
  --boundary-compensation-channel-ratio 0.5 \
  --boundary-compensation-max-batches 32 \
  --boundary-compensation-selection mismatch_outlier \
  --outlier-weight 0.25 \
  --output-dir exp/runs/llama3p2_3b_d025_boundary_mismatch_outlier_ch050_s1024 \
  > run_llama3p2_3b_d025_boundary_mismatch_outlier_ch050_s1024.log 2>&1 &
```

Note: the current remote code may only accept `resource_aware` and `random`. Apply `scripts/patch_mismatch_outlier_compensation.py` before launching the ablations. If the remote CLI still requires `--memory-weight`, pass `--memory-weight 0.0`; it should not be part of the main selection score.

## Low-Rank Residual Compensation

The affine ablation is useful as a diagnostic, but if full affine beats gated affine while still leaving high PPL, the next performance-oriented variant is a low-rank residual boundary adapter. Apply:

```bash
cd ~/work/AMCPrune_rescomp

python patch_low_rank_residual_compensation.py
python -m py_compile main.py amcprune/compensation.py
```

First run the capacity sanity check with the current best channel coverage:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python main.py \
  --strategy qwen_hidden_cosine_outlier \
  --model /home/stella/work/hf_models/Llama-3.2-3B \
  --dataset Salesforce/wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --dtype bf16 \
  --score hidden_cosine_streamline \
  --pruning-mode depth_width_physical \
  --pruning-ratio 0.25 \
  --unit-score none \
  --unit-pruning-ratio 0.0 \
  --width-candidate-count 0 \
  --max-samples 128 \
  --seq-len 1024 \
  --post-depth-recalibration \
  --exclude-depth-boundary-blocks \
  --boundary-compensation \
  --boundary-compensation-type low_rank_residual \
  --boundary-compensation-rank 64 \
  --boundary-compensation-train-steps 200 \
  --boundary-compensation-lr 1.0e-3 \
  --boundary-compensation-gamma 1.0 \
  --boundary-compensation-channel-ratio 0.5 \
  --boundary-compensation-max-batches 32 \
  --boundary-compensation-selection random \
  --outlier-weight 1.0 \
  --output-dir exp/runs/llama3p2_3b_d025_boundary_random_ch050_lora_r64_s1024_v1 \
  > run_llama3p2_3b_d025_boundary_random_ch050_lora_r64_s1024_v1.log 2>&1 &
```

If random improves over the affine baseline, test the proposed channel criterion with the same adapter:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python main.py \
  --strategy qwen_hidden_cosine_outlier \
  --model /home/stella/work/hf_models/Llama-3.2-3B \
  --dataset Salesforce/wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --dtype bf16 \
  --score hidden_cosine_streamline \
  --pruning-mode depth_width_physical \
  --pruning-ratio 0.25 \
  --unit-score none \
  --unit-pruning-ratio 0.0 \
  --width-candidate-count 0 \
  --max-samples 128 \
  --seq-len 1024 \
  --post-depth-recalibration \
  --exclude-depth-boundary-blocks \
  --boundary-compensation \
  --boundary-compensation-type low_rank_residual \
  --boundary-compensation-rank 64 \
  --boundary-compensation-train-steps 200 \
  --boundary-compensation-lr 1.0e-3 \
  --boundary-compensation-gamma 1.0 \
  --boundary-compensation-channel-ratio 0.5 \
  --boundary-compensation-max-batches 32 \
  --boundary-compensation-selection mismatch_outlier \
  --outlier-weight 0.25 \
  --output-dir exp/runs/llama3p2_3b_d025_boundary_mismatch_outlier_ch050_lora_r64_s1024_v1 \
  > run_llama3p2_3b_d025_boundary_mismatch_outlier_ch050_lora_r64_s1024_v1.log 2>&1 &
```

Compare:

```bash
python - <<'PY'
import json
from pathlib import Path

runs = {
    "affine_random_ch50": Path("exp/runs/llama3p2_3b_d025_boundary_random_ch050_s1024_v1/result.json"),
    "lowrank_random_ch50_r64": Path("exp/runs/llama3p2_3b_d025_boundary_random_ch050_lora_r64_s1024_v1/result.json"),
    "lowrank_mismatch_outlier_ch50_r64": Path("exp/runs/llama3p2_3b_d025_boundary_mismatch_outlier_ch050_lora_r64_s1024_v1/result.json"),
}

for name, path in runs.items():
    print("==", name)
    if not path.exists():
        print("missing")
        continue
    j = json.load(open(path))
    bc = j.get("boundary_compensation", {})
    print("type", bc.get("compensation_type"))
    print("selection", bc.get("selection_objective"))
    print("rank", bc.get("rank"))
    print("adapter_loss", bc.get("adapter_loss"))
    print("pruned_ppl", j.get("pruned", {}).get("perplexity"))
    print("delta", j.get("perplexity_delta"))
PY
```
