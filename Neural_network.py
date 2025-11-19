import pandas as pd
import numpy as np
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
        #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

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
    #print(f"Test Accuracy: {acc:.2f}%")
    return acc

def get_weights(model):
    return {name: param.clone() for name, param in model.named_parameters()}

def apply_mask(model, mask):
    with torch.no_grad():
        for name, param in model.named_parameters():
            param *= mask[name]

def create_winning_ticket(model, mask, theta):
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.data = theta[name] * mask[name]
    print(f"Layer {name}: {param.numel()} weights, {(param == 0).sum().item()} zeros")
    return model

def randomly_reinitialize(model, mask):
    for name, param in model.named_parameters():
        if name in mask:
            param.data *= mask[name] 
            if param.dim() > 1:
                with torch.no_grad():
                    unpruned_mask = (mask[name] == 1).float()
                    param.data = param.data * unpruned_mask + torch.randn_like(param.data) * unpruned_mask
    return model

def calculate_actual_prune_percent(model):
    total_weights = sum(p.numel() for p in model.parameters() if p.dim() > 1)
    zero_weights = sum((p == 0).sum().item() for p in model.parameters() if p.dim() > 1)
    return 100 * zero_weights / total_weights

def count_zeros(model):
    return sum((p == 0).sum().item() for p in model.parameters() if p.dim() > 1)


########################### Lottery Ticket Algorithm #################################################################

# Step 1 and 2: train the randomly initialized neural network 
def dense_neural_network_MNIST(df_accuracies):
    print("\nStep 1 and 2: training the randomly initialized neural network ")
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
    print(f"Initial accuracy : {round(dense_acc, 2)}%.")
    df_accuracies.append({"Round": "Initial model", "Test Accuracy (with training)": dense_acc})
    return df_accuracies, model, get_weights(model), get_weights(SimpleNN(input_size, hidden_size, output_size)), train_loader, test_loader


# Step 3: Prune the smallest weights
def prune_by_magnitude(model, prune_percent=20):
    all_weights = torch.cat([param.data.abs().view(-1) for param in model.parameters() if param.dim() > 1])
    k = int(len(all_weights) * prune_percent / 100)
    threshold = torch.topk(all_weights, k, largest=False).values.max()
    print(f"Pruning threshold: {round(float(threshold.item()),2)}")
    mask = {}
    for name, param in model.named_parameters():
        if param.dim() > 1:  # Only prune weights, not biases
            mask[name] = (param.data.abs() > threshold).float()
            #print(f"Layer {name}: {mask[name].numel()} weights, {(mask[name] == 0).sum().item()} zeros")
        else:
            mask[name] = torch.ones_like(param)
    return mask


# Step 4: creating the winning ticket f(x; m⊙theta_0)
def iterative_pruning(total_prune_percent=90, rounds=8, epochs_per_round=10, lr=0.001, LTH = True):
    df_accuracies = []
    df_accuracies, model, theta_j, theta0, train_loader, test_loader = dense_neural_network_MNIST(df_accuracies)
    print("\nStep 4: Creating the Winning ticket")
    print(f"Number of zeros before pruning: {count_zeros(model)}")

    criterion = nn.CrossEntropyLoss()
    prune_percent = total_prune_percent**(1/rounds)
    print(f"At each round, we are pruning : {round(prune_percent,2)}% of the weights.")

    for pruning_round in range(rounds):
        print(f"\n--- Round {pruning_round + 1}/{rounds} ---")
        current_prune_percent = prune_percent**(pruning_round + 1)
        print(f"Current pruning percentage (Method 1): {current_prune_percent:.2f}%")
        # Pruning
        mask = prune_by_magnitude(model, current_prune_percent)
        if LTH : #Then we keep the initial weights
            # Reset to initial weights
            model = create_winning_ticket(model, mask, theta0)
        else: #If we are not doing LTH, then the initial weights are random !! 
            model = randomly_reinitialize(model, mask)

        # Making sure enough the right amount are getting pruned 
        actual_prune_percent = calculate_actual_prune_percent(model)
        print(f"Actual pruning percentage: {actual_prune_percent:.2f}%")

        # Evaluate (no retraining)
        acc = evaluate_the_model(model, test_loader)
        print(f"Accuracy after pruning (no retraining): {acc:.2f}%")

        # Train the pruned model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        training_the_model(model, train_loader, optimizer, criterion, epochs_per_round)

        #Test Accuracies
        acc_post_training = evaluate_the_model(model, test_loader)
        print(f"Accuracy after retraining: {acc_post_training:.2f}%")
        df_accuracies.append({"Round": f"Round {pruning_round + 1}", "Pruning percentage": actual_prune_percent, "Test Accuracy (no retraining)": acc, "Test Accuracy (with training)": acc_post_training})
        
    return df_accuracies, model












