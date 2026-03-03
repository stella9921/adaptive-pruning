import torch
import torch.nn as nn
from torchvision import models

def measure_dense_stats(model_name, batch_size=1, input_size=(3, 224, 224)):
    # 1. 모델 로드 (Dense)
    if 'efficientnet' in model_name:
        model = models.efficientnet_b0(pretrained=True).cuda()
    elif 'mobilenet' in model_name:
        model = models.mobilenet_v2(pretrained=True).cuda()
    elif 'vgg16' in model_name:
        model = models.vgg16(pretrained=True).cuda()
    elif 'resnet18' in model_name:
        model = models.resnet18(pretrained=True).cuda()
    elif 'resnet152' in model_name:
        model = models.resnet152(pretrained=True).cuda()
    
    model.eval()

    # 2. 파라미터 수 및 모델 크기 계산
    params_count = sum(p.numel() for p in model.parameters())
    params_m = params_count / 1e6
    model_size_mb = params_count * 4 / (1024**2)  # float32 기준

    # 3. Peak VRAM 측정
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 더미 입력 (배치 사이즈 반영)
    dummy_input = torch.randn(batch_size, *input_size).cuda()
    
    with torch.no_grad():
        _ = model(dummy_input)
    
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)

    return {
        'Model': model_name.upper(),
        'Params (M)': f"{params_m:.2f}",
        'Size (MB)': f"{model_size_mb:.2f}",
        'Peak VRAM (MB)': f"{peak_vram:.2f}"
    }

# 실행 부분
target_models = ['resnet18', 'resnet152', 'mobilenet_v2', 'efficientnet_b0', 'vgg16']
batch_size = 1 # 코드에서 확인된 배치 사이즈

print(f"{'Model':<15} | {'Params (M)':<12} | {'Size (MB)':<12} | {'Peak VRAM (MB)':<15}")
print("-" * 65)

for name in target_models:
    try:
        stats = measure_dense_stats(name, batch_size=batch_size)
        print(f"{stats['Model']:<15} | {stats['Params (M)']:<12} | {stats['Size (MB)']:<12} | {stats['Peak VRAM (MB)']:<15}")
    except Exception as e:
        print(f"Error measuring {name}: {e}")