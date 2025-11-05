import torch
import torch.nn as nn
import torch.nn.functional as F
from Accessing_data import load_mnist

"""
In the original article, the steps to identify a winning ticket are : 
1. Randomly initialize a neural network f(x; theta_0) (where theta_0 follow D_θ).
2. Train the network for j-iterations, arriving at parameters theta_j.
3. Prune p% of the parameters in theta_j, creating a mask m.
4. Reset the remaining parameters to their values in theta_0, creating the winning ticket f(x; m⊙theta_0).

"""

# Define the class of a Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
########################### Helper functions #################################################################

def training_the_model(model, train_loader, optimizer, criterion, num_epochs = 10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            flattened_images = images.view(images.shape[0], -1)
            outputs = model(flattened_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

def evaluate_the_model(model, test_loader):
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
    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

def get_weights(model):
    return {name: param.clone() for name, param in model.named_parameters()}

def apply_mask(model, mask):
    with torch.no_grad():
        for name, param in model.named_parameters():
            param *= mask[name]


########################### Lottery Ticket Algorithm #################################################################

# Step 1: train the randomly initialized neural network 
def dense_neural_network_MNIST():
    input_size = 784   
    hidden_size = 128
    output_size = 10   
    model = SimpleNN(input_size, hidden_size, output_size)

    batch_size = 64
    train_loader, test_loader = load_mnist(batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    training_the_model(model, train_loader, optimizer, criterion)
    dense_acc = evaluate_the_model(model, test_loader)

    return model, get_weights(model), get_weights(SimpleNN()), train_loader, test_loader


# Step 2: Prune the smallest weights
def prune_by_magnitude(model, prune_percent=20):
    all_weights = torch.cat([param.data.view(-1).abs() for param in model.parameters()])
    threshold = torch.quantile(all_weights, prune_percent / 100.0)

    mask = {}
    for name, param in model.named_parameters():
        mask[name] = (param.abs() > threshold).float()
    return mask















