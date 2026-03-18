import torch
import timm
import time
import torch.nn as nn

def measure_throughput(model, device, batch_size=64, n_warmup=20, n_runs=100):
    model.eval()
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    print(f"워밍업 중... ({n_warmup}회)")
    for _ in range(n_warmup):
        with torch.no_grad():
            model(x)
    torch.cuda.synchronize()
    print(f"측정 중... ({n_runs}회)")
    torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    throughput = (batch_size * n_runs) / elapsed
    latency_ms = (elapsed / n_runs) * 1000
    peak_vram  = torch.cuda.max_memory_allocated(device) / (1024**2)
    return throughput, latency_ms, peak_vram

device = torch.device("cuda:0")
ckpt_path = "./exp/checkpoints/deit_small_patch16_224_vit_pdt_FINAL_COMPRESSED.pth"

# 버퍼 제거
state = torch.load(ckpt_path, map_location=device)
clean_state = {k: v for k, v in state.items()
               if not any(x in k for x in ['mask', 'grad_ema', 'hessian_score'])}

# ── 원본 모델 측정 ──
print("\n===== 원본 모델 측정 =====")
model_orig = timm.create_model('deit_small_patch16_224', pretrained=False).to(device)
model_orig.head = nn.Linear(384, 100).to(device)
orig_tp, orig_lat, orig_vram = measure_throughput(model_orig, device)
orig_params = sum(p.numel() for p in model_orig.parameters()) / 1e6
print(f"파라미터:   {orig_params:.2f}M")
print(f"Throughput: {orig_tp:.1f} img/s")
print(f"Latency:    {orig_lat:.2f} ms")
print(f"Peak VRAM:  {orig_vram:.2f} MB")
del model_orig
torch.cuda.empty_cache()

# ── 압축 모델: state_dict 크기로 레이어 직접 교체 ──
print("\n===== 압축 모델 측정 =====")
model_pruned = timm.create_model('deit_small_patch16_224', pretrained=False).to(device)
model_pruned.head = nn.Linear(384, 100).to(device)

# state_dict 크기에 맞게 레이어 직접 교체
for i in range(12):
    blk = model_pruned.blocks[i]
    prefix = f'blocks.{i}'

    # attn.qkv 교체
    qkv_key = f'{prefix}.attn.qkv.weight'
    if qkv_key in clean_state:
        new_out = clean_state[qkv_key].shape[0]  # 압축된 출력 크기
        old_in  = blk.attn.qkv.in_features        # 384 유지
        blk.attn.qkv = nn.Linear(old_in, new_out).to(device)

    # attn.proj 교체
    proj_key = f'{prefix}.attn.proj.weight'
    if proj_key in clean_state:
        new_in  = clean_state[proj_key].shape[1]  # 압축된 입력 크기
        old_out = blk.attn.proj.out_features       # 384 유지
        blk.attn.proj = nn.Linear(new_in, old_out).to(device)

    # mlp.fc1 교체
    fc1_key = f'{prefix}.mlp.fc1.weight'
    if fc1_key in clean_state:
        new_out = clean_state[fc1_key].shape[0]
        old_in  = blk.mlp.fc1.in_features
        blk.mlp.fc1 = nn.Linear(old_in, new_out).to(device)

    # mlp.fc2 교체
    fc2_key = f'{prefix}.mlp.fc2.weight'
    if fc2_key in clean_state:
        new_in  = clean_state[fc2_key].shape[1]
        old_out = blk.mlp.fc2.out_features
        blk.mlp.fc2 = nn.Linear(new_in, old_out).to(device)

# timm Attention 내부의 num_heads도 맞게 수정
for i in range(12):
    blk = model_pruned.blocks[i]
    new_qkv_out = blk.attn.qkv.out_features
    new_heads = new_qkv_out // (3 * 64)  # head_dim=64
    if new_heads > 0:
        blk.attn.num_heads = new_heads
        blk.attn.head_dim  = 64
        blk.attn.scale     = 64 ** -0.5
    else:
        blk.attn.num_heads = 1
        blk.attn.head_dim  = new_qkv_out // 3 if new_qkv_out > 0 else 1

# state_dict 로드
missing, unexpected = model_pruned.load_state_dict(clean_state, strict=False)
print(f"로드 완료 | missing: {len(missing)} | unexpected: {len(unexpected)}")
if missing:
    print(f"  missing 예시: {missing[:3]}")

pruned_tp, pruned_lat, pruned_vram = measure_throughput(model_pruned, device)
pruned_params = sum(p.numel() for p in model_pruned.parameters()) / 1e6
print(f"파라미터:   {pruned_params:.2f}M")
print(f"Throughput: {pruned_tp:.1f} img/s")
print(f"Latency:    {pruned_lat:.2f} ms")
print(f"Peak VRAM:  {pruned_vram:.2f} MB")

# ── 비교 결과 ──
print("\n" + "="*52)
print("            비교 결과 요약")
print("="*52)
print(f"{'':22s} {'원본':>10s} {'압축':>10s} {'비율':>8s}")
print("-"*52)
print(f"{'파라미터 (M)':22s} {orig_params:>10.2f} {pruned_params:>10.2f} {pruned_params/orig_params*100:>7.1f}%")
print(f"{'Throughput (img/s)':22s} {orig_tp:>10.1f} {pruned_tp:>10.1f} {pruned_tp/orig_tp:>7.2f}x")
print(f"{'Latency (ms)':22s} {orig_lat:>10.2f} {pruned_lat:>10.2f} {orig_lat/pruned_lat:>7.2f}x")
print(f"{'Peak VRAM (MB)':22s} {orig_vram:>10.2f} {pruned_vram:>10.2f} {pruned_vram/orig_vram*100:>7.1f}%")
print("="*52)
print(f"\n🚀 Throughput Speedup: {pruned_tp/orig_tp:.2f}x")
print(f"⚡ Latency Speedup:    {orig_lat/pruned_lat:.2f}x")
print(f"💾 VRAM 절감:          {(1-pruned_vram/orig_vram)*100:.1f}%")
print(f"📦 파라미터 감소:       {(1-pruned_params/orig_params)*100:.1f}%")
