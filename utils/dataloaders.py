from torchvision import datasets, transforms

def cifar10_32x32_loader():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def custom_32x32_loader():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    train_dataset = datasets.ImageFolder("custom_data/train", transform=transform)
    test_dataset = datasets.ImageFolder("custom_data/test", transform=transform)
    return train_dataset, test_dataset


def custom_64x64_loader():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    train_dataset = datasets.ImageFolder("custom_data/train", transform=transform)
    test_dataset = datasets.ImageFolder("custom_data/test", transform=transform)
    return train_dataset, test_dataset
