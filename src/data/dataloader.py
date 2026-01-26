import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder # ImageNet용

def get_dataloaders(config):
    # 1. 설정값 추출
    model_cfg = config.get('model', {})
    dataset_cfg = config.get('dataset', {}) 
    
    # [수정 완료] dataset_cfg가 문자열인지 딕셔너리인지 확실히 체크
    if isinstance(dataset_cfg, str):
        dataset_name = dataset_cfg
        data_path = './data'
    elif isinstance(dataset_cfg, dict):
        dataset_name = dataset_cfg.get('name', 'cifar100')
        data_path = dataset_cfg.get('path', './data')
    else:
        dataset_name = 'cifar100'
        data_path = './data'

    batch_size = model_cfg.get('batch_size', 128)
    num_workers = model_cfg.get('num_workers', 4)

    # 2. 데이터셋별 분기
    if dataset_name.lower() == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        train_set = ImageFolder(root=f"{data_path}/train", transform=transform_train)
        val_set = ImageFolder(root=f"{data_path}/val", transform=transform_test)
        
        print(f"✅ ImageNet Data loaded: Train({len(train_set)}), Val({len(val_set)})")
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                                  num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers, pin_memory=True)
        
        return train_loader, val_loader

    else:
        # CIFAR-100 로직 (기존과 동일)
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

        train_set = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
        val_set = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        print(f"✅ CIFAR-100 Data loaded: Train({len(train_set)}), Val/Test({len(val_set)})")
        
        return train_loader, val_loader, test_loader