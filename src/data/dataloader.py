import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder

def get_dataloaders(config):
    model_cfg = config.get('model', {})
    dataset_cfg = config.get('dataset', {})

    if isinstance(dataset_cfg, str):
        dataset_name = dataset_cfg
        data_path = './data'
    elif isinstance(dataset_cfg, dict):
        dataset_name = dataset_cfg.get('name', 'cifar100')
        # 버그 1: 'path' 대신 'data_dir' 읽도록 수정
        data_path = dataset_cfg.get('data_dir', dataset_cfg.get('path', './data'))
    else:
        dataset_name = 'cifar100'
        data_path = './data'

    # 버그 2: batch_size를 config 최상위에서도 읽도록 수정
    batch_size = config.get('batch_size', model_cfg.get('batch_size', 128))
    num_workers = dataset_cfg.get('num_workers', model_cfg.get('num_workers', 4)) \
                  if isinstance(dataset_cfg, dict) else 4

    # ImageNet / ImageNet-100
    if dataset_name.lower() in ['imagenet', 'imagenet100']:
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

        # train 폴더들을 심볼릭 링크로 통합 (label 겹침 방지)
        merged_train = os.path.join(data_path, 'train_merged')
        if not os.path.exists(merged_train):
            os.makedirs(merged_train, exist_ok=True)
            for d in ['train', 'train.X1', 'train.X2', 'train.X3', 'train.X4']:
                src_dir = os.path.join(data_path, d)
                if os.path.exists(src_dir):
                    for cls in os.listdir(src_dir):
                        src_cls = os.path.join(src_dir, cls)
                        dst_cls = os.path.join(merged_train, cls)
                        if not os.path.exists(dst_cls):
                            os.symlink(src_cls, dst_cls)
            print(f"[*] train_merged 폴더 생성 완료")

        if not os.path.exists(merged_train) or len(os.listdir(merged_train)) == 0:
            raise FileNotFoundError(f"train 폴더를 찾을 수 없습니다: {data_path}")

        val_dir = None
        for d in ['val', 'val.X']:
            full = os.path.join(data_path, d)
            if os.path.exists(full):
                val_dir = full
                break

        if val_dir is None:
            raise FileNotFoundError(f"val 폴더를 찾을 수 없습니다: {data_path}")

        train_set = ImageFolder(root=merged_train, transform=transform_train)
        print(f"[*] Train classes: {len(train_set.classes)}")

        val_set = ImageFolder(root=val_dir, transform=transform_test)

        print(f"✅ {dataset_name} loaded: Train({len(train_set)}), Val({len(val_set)})")
        print(f"   Val dir: {val_dir}")

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)

        return train_loader, val_loader

    else:
        # CIFAR-100
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

        train_set = torchvision.datasets.CIFAR100(
            root=data_path, train=True, download=True, transform=transform_train)
        full_test_set = torchvision.datasets.CIFAR100(
            root=data_path, train=False, download=True, transform=transform_test)

        generator = torch.Generator().manual_seed(42)
        val_set, test_set = torch.utils.data.random_split(
            full_test_set, [5000, 5000], generator=generator)

        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_set, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_set, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)

        print(f"✅ CIFAR-100: Train(50,000), Val(5,000), Test(5,000)")

        return train_loader, val_loader, test_loader