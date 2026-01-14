import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataloaders(config):
    """
    CIFAR-100 데이터를 로드하여 Train, Val, Test 데이터로더를 반환합니다.
    """
    # 1. 설정값 추출
    # config['model'] 에 batch_size, num_workers 등이 정의되어 있다고 가정합니다.
    batch_size = config.get('model', {}).get('batch_size', 128)
    num_workers = config.get('model', {}).get('num_workers', 4)
    data_path = config.get('model', {}).get('data_path', './data')

    # 2. 전처리 설정 (CIFAR-100 전용 통계치 사용)
    # Mean/Std: (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # 3. 데이터셋 로드 (CIFAR100 클래스 사용)
    train_set = torchvision.datasets.CIFAR100(
        root=data_path, train=True, download=True, transform=transform_train)
    
    # 별도의 Validation Set이 없는 경우 Test Set을 Validation 용도로 활용합니다.
    val_set = torchvision.datasets.CIFAR100(
        root=data_path, train=False, download=True, transform=transform_test)
    
    test_set = torchvision.datasets.CIFAR100(
        root=data_path, train=False, download=True, transform=transform_test)

    # 4. 데이터로더 생성
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"✅ CIFAR-100 Data loaded: Train({len(train_set)}), Val/Test({len(test_set)})")
    
    return train_loader, val_loader, test_loader