from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_set = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

print("Train set size:", len(train_set))
print("Test set size:", len(test_set))

train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=False)

for images, labels in train_loader:
    X_train = images
    y_train = labels
    break 

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

del train_set
del test_set

transform_CIFAR = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])

train_set = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_CIFAR)
test_set = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_CIFAR)

print("Number of training samples:", len(train_set))
print("Number of test samples:", len(test_set))
print("Classes:", train_set.classes)







