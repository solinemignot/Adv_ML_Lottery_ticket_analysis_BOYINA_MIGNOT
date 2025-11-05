import torch
import torch.nn as nn
from Accessing_data import load_mnist
import torch.nn.functional as F


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def apply_mask(model, mask):
    for name, param in model.named_parameters():
        param.data = param.data * mask[name]


def neural_network_MNIST():
    input_size = 784
    hidden_size = 128
    output_size = 10
    model = SimpleNN(input_size, hidden_size, output_size)

    mask = {}
    for name, param in model.named_parameters():
        mask[name] = torch.ones_like(param)
    apply_mask(model, mask)

    # Load data
    batch_size = 64
    train_loader, test_loader = load_mnist(batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training 
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            flattened_images = images.view(images.shape[0], -1)
            outputs = model(flattened_images)
            loss = criterion(outputs, labels)
            loss.backward()
            for name, param in model.named_parameters():
                if name in mask:
                    param.grad.data *= mask[name]
            optimizer.step()
            apply_mask(model, mask)
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            flattened_images = images.view(images.shape[0], -1)
            outputs = model(flattened_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

    return model


model_mnist = neural_network_MNIST()

def get_weights_per_layer(model):
    weights = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights[name] = param.data.clone()
    return weights

# After training, call this function:
weights_per_layer = get_weights_per_layer(model_mnist)

# Print the weights for each layer
for layer_name, weights in weights_per_layer.items():
    print(f"Layer: {layer_name}")
    print(f"Weights shape: {weights.shape}")
    print(f"Weights:\n{weights}\n")