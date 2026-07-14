import os
import random
from glob import glob
from io import BytesIO

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _loader_options(seed, num_workers, offset=0):
    generator = torch.Generator().manual_seed(int(seed) + offset)
    return {
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'worker_init_fn': _seed_worker,
        'generator': generator,
    }


def _find_arrow_dataset_path(data_path):
    candidates = [
        data_path,
        os.path.join(data_path, 'default', '0.0.0'),
    ]
    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        if os.path.exists(os.path.join(candidate, 'dataset_dict.json')):
            return candidate
        if os.path.exists(os.path.join(candidate, 'state.json')):
            return candidate
        if glob(os.path.join(candidate, '*.arrow')):
            return candidate

        for child in sorted(os.listdir(candidate)):
            child_path = os.path.join(candidate, child)
            if os.path.isdir(child_path) and glob(os.path.join(child_path, '*.arrow')):
                return child_path

    return None


def _load_hf_split_from_arrow_files(arrow_dir, split_names):
    from datasets import Dataset, concatenate_datasets

    files = []
    for split_name in split_names:
        files.extend(glob(os.path.join(arrow_dir, f"*{split_name}*.arrow")))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(
            f"No Arrow files found for splits {split_names} under {arrow_dir}"
        )

    shards = [Dataset.from_file(path) for path in files]
    return shards[0] if len(shards) == 1 else concatenate_datasets(shards)


def _load_hf_imagenet_dataset(arrow_path):
    from datasets import load_from_disk

    if (
        os.path.exists(os.path.join(arrow_path, 'dataset_dict.json'))
        or os.path.exists(os.path.join(arrow_path, 'state.json'))
    ):
        hf_dataset = load_from_disk(arrow_path)
        return hf_dataset['train'], hf_dataset['validation']

    train_split = _load_hf_split_from_arrow_files(arrow_path, ('train',))
    val_split = _load_hf_split_from_arrow_files(arrow_path, ('validation',))
    return train_split, val_split


def _to_pil_image(image):
    from PIL import Image

    if isinstance(image, Image.Image):
        return image
    if isinstance(image, dict):
        if image.get('bytes') is not None:
            return Image.open(BytesIO(image['bytes']))
        if image.get('path') is not None:
            return Image.open(image['path'])
    return Image.fromarray(image)


def _imagenet_transforms():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
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
    return transform_train, transform_test


def _build_hf_imagenet_loaders(
    dataset_name, data_path, batch_size, num_workers, seed,
    transform_train, transform_test,
):
    arrow_path = _find_arrow_dataset_path(data_path)
    if not arrow_path:
        return None

    from torch.utils.data import Dataset

    print(f"[*] HuggingFace Arrow dataset detected: {arrow_path}")
    hf_train, hf_val = _load_hf_imagenet_dataset(arrow_path)

    class HFImageNetDataset(Dataset):
        def __init__(self, hf_split, transform=None):
            self.data = hf_split
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            image = _to_pil_image(item['image']).convert('RGB')
            label = item['label']
            if self.transform:
                image = self.transform(image)
            return image, label

    train_set = HFImageNetDataset(hf_train, transform=transform_train)
    val_set = HFImageNetDataset(hf_val, transform=transform_test)

    print(f"[*] {dataset_name} loaded: Train({len(train_set)}), Val({len(val_set)})")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        **_loader_options(seed, num_workers, 1),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        **_loader_options(seed, num_workers, 2),
    )
    return train_loader, val_loader


def _build_imagefolder_imagenet_loaders(
    dataset_name, data_path, batch_size, num_workers, seed,
    transform_train, transform_test,
):
    merged_train = os.path.join(data_path, 'train_merged')
    if not os.path.exists(merged_train):
        os.makedirs(merged_train, exist_ok=True)
        for dirname in ('train', 'train.X1', 'train.X2', 'train.X3', 'train.X4'):
            src_dir = os.path.join(data_path, dirname)
            if not os.path.exists(src_dir):
                continue
            for class_name in os.listdir(src_dir):
                src_class = os.path.join(src_dir, class_name)
                dst_class = os.path.join(merged_train, class_name)
                if not os.path.exists(dst_class):
                    os.symlink(src_class, dst_class)
        print("[*] train_merged directory prepared")

    if not os.path.exists(merged_train) or len(os.listdir(merged_train)) == 0:
        raise FileNotFoundError(f"Train directory not found under: {data_path}")

    val_dir = None
    for dirname in ('val', 'val.X'):
        candidate = os.path.join(data_path, dirname)
        if os.path.exists(candidate):
            val_dir = candidate
            break

    if val_dir is None:
        raise FileNotFoundError(f"Validation directory not found under: {data_path}")

    train_set = ImageFolder(root=merged_train, transform=transform_train)
    val_set = ImageFolder(root=val_dir, transform=transform_test)

    print(f"[*] {dataset_name} loaded: Train({len(train_set)}), Val({len(val_set)})")
    print(f"[*] Train classes: {len(train_set.classes)}")
    print(f"[*] Val dir: {val_dir}")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        **_loader_options(seed, num_workers, 1),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        **_loader_options(seed, num_workers, 2),
    )
    return train_loader, val_loader


def get_dataloaders(config):
    model_cfg = config.get('model', {})
    dataset_cfg = config.get('dataset', {})
    seed = config.get('reproducibility', {}).get('seed', config.get('seed', 42))

    if isinstance(dataset_cfg, str):
        dataset_name = dataset_cfg
        data_path = './data'
    elif isinstance(dataset_cfg, dict):
        dataset_name = dataset_cfg.get('name', 'cifar100')
        data_path = dataset_cfg.get('data_dir', dataset_cfg.get('path', './data'))
    else:
        dataset_name = 'cifar100'
        data_path = './data'

    batch_size = config.get('batch_size', model_cfg.get('batch_size', 128))
    num_workers = (
        dataset_cfg.get('num_workers', model_cfg.get('num_workers', 4))
        if isinstance(dataset_cfg, dict) else 4
    )

    if dataset_name.lower() in ['imagenet', 'imagenet100', 'imagenet1k']:
        transform_train, transform_test = _imagenet_transforms()
        hf_loaders = _build_hf_imagenet_loaders(
            dataset_name,
            data_path,
            batch_size,
            num_workers,
            seed,
            transform_train,
            transform_test,
        )
        if hf_loaders is not None:
            return hf_loaders
        return _build_imagefolder_imagenet_loaders(
            dataset_name,
            data_path,
            batch_size,
            num_workers,
            seed,
            transform_train,
            transform_test,
        )

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
        root=data_path,
        train=True,
        download=True,
        transform=transform_train,
    )
    full_test_set = torchvision.datasets.CIFAR100(
        root=data_path,
        train=False,
        download=True,
        transform=transform_test,
    )

    generator = torch.Generator().manual_seed(int(seed))
    val_set, test_set = torch.utils.data.random_split(
        full_test_set, [5000, 5000], generator=generator
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        **_loader_options(seed, num_workers, 1),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        **_loader_options(seed, num_workers, 2),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        **_loader_options(seed, num_workers, 3),
    )

    print("[*] CIFAR-100 loaded: Train(50,000), Val(5,000), Test(5,000)")

    return train_loader, val_loader, test_loader
