
import torch
import torchvision.transforms as transforms
from medmnist import PathMNIST
from config import IMAGE_SIZE, BATCH_SIZE, DEVICE

def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    print("Loading datasets...")
    train_dataset = PathMNIST(split='train', download=True, transform=transform)
    test_dataset = PathMNIST(split='test', download=True, transform=transform)

    # CPU Safety Check
    if DEVICE.type == 'cpu':
        print("WARNING: On CPU. Using a smaller subset to speed this up.")
        indices = torch.arange(2000)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    
    return train_loader, test_loader