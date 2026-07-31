# AMCPrune Ablation Runs

Goal: validate the channel-selection part of the method while keeping the depth-pruned backbone and pruning ratio fixed.

Main ablation table:

| Variant | Selection score | Purpose |
|---|---|---|
| random | random channels | checks whether affine capacity alone helps |
| mismatch-only | `m_c` | tests boundary distribution shift |
| outlier-only | `o_c` | tests activation-risk selection |
| mismatch+outlier | `Norm(m_c) + lambda_o Norm(o_c)` | proposed main variant |

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
