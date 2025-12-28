from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_mnist(batch_size):
    """
    Charge le dataset MNIST avec la normalisation standard.
    """
    # Normalisation standard pour MNIST (Mean: 0.1307, Std: 0.3081)
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # IMPORTANT : download=True pour le cloud
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # On utilise num_workers=2 pour accélérer le chargement des données vers le GPU
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


def load_cifar(batch_size):
    """
    Charge le dataset CIFAR-10 avec Data Augmentation (pour le train) et Normalisation.
    """
    # Stats standards pour CIFAR-10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    # Data Augmentation pour l'entraînement (comme dans le papier)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Juste la normalisation pour le test
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader