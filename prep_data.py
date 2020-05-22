from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data():
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST("", download=True, transform=transform, train=True)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    return train_loader
