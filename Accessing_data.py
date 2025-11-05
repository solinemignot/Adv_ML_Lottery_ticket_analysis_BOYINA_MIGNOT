from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_mnist(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_set = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def load_cifar(batch_size):
    transform_CIFAR = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  ])

    train_set = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_CIFAR)
    test_set = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_CIFAR)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


