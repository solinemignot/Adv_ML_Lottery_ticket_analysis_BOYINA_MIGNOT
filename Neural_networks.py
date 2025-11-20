import torch
import torch.nn as nn
import torch.nn.functional as F
from Accessing_data import load_mnist, load_cifar
from Helper_functions import *

"""
In the original article, the steps to identify a winning ticket are : 
1. Randomly initialize a neural network f(x; theta_0) (where theta_0 follow D_θ).
2. Train the network for j-iterations, arriving at parameters theta_j.
3. Prune p% of the parameters in theta_j, creating a mask m.
4. Reset the remaining parameters to their values in theta_0, creating the winning ticket f(x; m⊙theta_0).

"""

########################### Neural Network - for MNIST #################################################################

# Simple Neural Network - for MNIST
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 1 and 2: train the randomly initialized neural network 
def dense_neural_network_MNIST(df_accuracies):
    print("\nStep 1 and 2: training the randomly initialized neural network for MNIST.")
    input_size = 784   
    hidden_size = 128
    output_size = 10 
    model = SimpleNN(input_size, hidden_size, output_size)
    theta_0 =  get_weights(model)

    batch_size = 64
    train_loader, test_loader = load_mnist(batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    training_the_model(model, train_loader, optimizer, criterion)
    dense_acc = evaluate_the_model(model, test_loader)
    print(f"Initial accuracy : {round(dense_acc, 2)}%.")
    df_accuracies.append({"Round": "Initial model", "Test Accuracy (with training)": dense_acc})
    return df_accuracies, model, get_weights(model), theta_0, train_loader, test_loader

# Step 4: creating the winning ticket f(x; m⊙theta_0)
def iterative_pruning_MNIST(total_prune_percent=90, rounds=8, epochs_per_round=10, lr=0.001, LTH = True, strategy_1 = True):
    df_accuracies = []
    df_accuracies, model, thetaj, theta0, train_loader, test_loader = dense_neural_network_MNIST(df_accuracies)
    print("\nStep 4: Creating the Winning ticket")
    print(f"Number of zeros before pruning: {count_zeros(model)}")

    criterion = nn.CrossEntropyLoss()
    prune_percent = 1 - (1 - total_prune_percent/100)**(1/rounds)
    remaining_weights_percent = 1
    current_prune_percent = 0
    print(f"At each round, we are pruning : {round(prune_percent,2)}% of the weights.")

    for pruning_round in range(rounds):
        print(f"\n--- Round {pruning_round + 1}/{rounds} ---")
        current_prune_percent += remaining_weights_percent * prune_percent
        remaining_weights_percent = 1 - current_prune_percent
        print(f"Current pruning percentage (Method 1): {current_prune_percent*100:.2f}%")
        
        mask = prune_by_magnitude(model, current_prune_percent*100)
        if LTH : 
            if strategy_1 or (pruning_round + 1 == rounds): #Then we keep the initial weights
                model = create_winning_ticket(model, mask, theta0)
            else : #Then we keep the same weights as previously trained, but with the mask, except at the last round
                model = create_winning_ticket(model, mask, thetaj)
        else: #If we are not doing LTH, then the initial weights are random 
            model = randomly_reinitialize(model, mask)

        actual_prune_percent = calculate_actual_prune_percent(model)
        print(f"Current pruning percentage: {actual_prune_percent:.2f}%")

        acc = evaluate_the_model(model, test_loader)
        #print(f"Accuracy after pruning (no retraining): {acc:.2f}%")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        training_the_model(model, train_loader, optimizer, criterion, epochs_per_round)
        thetaj = get_weights(model)

        #Test Accuracies
        acc_post_training = evaluate_the_model(model, test_loader)
        print(f"Accuracy after retraining: {acc_post_training:.2f}%")
        df_accuracies.append({"Round": f"Round {pruning_round + 1}", "Pruning percentage": actual_prune_percent, "Test Accuracy (no retraining)": acc, "Test Accuracy (with training)": acc_post_training})
        
    return df_accuracies, model


########################### Convolutional Neural Network - for CIFAR - 10 #################################################################
#xxx - this whole section has just begun to exist so is not completely transformed from MNIST to CIFAR

class SimpleCNN(nn.Module):
    def __init__(self, hidden_size=512):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def dense_neural_network_CIFAR(df_accuracies):
    print("\nStep 1 and 2: training the randomly initialized neural network for CIFAR.")
    hidden_size = 1024
    model = SimpleCNN(hidden_size=hidden_size)
    theta_0 =  get_weights(model)

    batch_size = 64
    train_loader, test_loader = load_cifar(batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    training_the_model(model, train_loader, optimizer, criterion)
    dense_acc = evaluate_the_model(model, test_loader)
    print(f"Initial accuracy : {round(dense_acc, 2)}%.")
    df_accuracies.append({"Round": "Initial model", "Test Accuracy (with training)": dense_acc})
    return df_accuracies, model, get_weights(model), theta_0, train_loader, test_loader



def iterative_pruning_CIFAR(total_prune_percent=90, rounds=8, epochs_per_round=10, lr=0.001, LTH = True):
    #xxx - not done at all
    df_accuracies = []
    df_accuracies, model, _, theta0, train_loader, test_loader = dense_neural_network_CIFAR(df_accuracies)
    print("\nStep 4: Creating the Winning ticket")
    print(f"Number of zeros before pruning: {count_zeros(model)}")

    criterion = nn.CrossEntropyLoss()
    prune_percent = 1 - (1 - total_prune_percent/100)**(1/rounds)
    remaining_weights_percent = 1
    current_prune_percent = 0
    print(f"At each round, we are pruning : {round(prune_percent,2)}% of the weights.")

    for pruning_round in range(rounds):
        print(f"\n--- Round {pruning_round + 1}/{rounds} ---")
        current_prune_percent += remaining_weights_percent * prune_percent
        remaining_weights_percent = 1 - current_prune_percent
        print(f"Current pruning percentage (Method 1): {current_prune_percent*100:.2f}%")
        # Pruning
        mask = prune_by_magnitude(model, current_prune_percent*100)
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














