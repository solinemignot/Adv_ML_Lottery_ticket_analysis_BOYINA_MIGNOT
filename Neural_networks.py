import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from Accessing_data import load_mnist, load_cifar
from Helper_functions import *

"""
In the original article, the steps to identify a winning ticket are : 
1. Randomly initialize a neural network f(x; theta_0) (where theta_0 follow D_θ).
2. Train the network for j-iterations, arriving at parameters theta_j.
3. Prune p% of the parameters in theta_j, creating a mask m.
4. Reset the remaining parameters to their values in theta_0, creating the winning ticket f(x; m⊙theta_0).

"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Helper functions loaded. User device: {device}")

BATCH_SIZE = 256  
TRAIN_LOADER_MNIST, TEST_LOADER_MNIST = load_mnist(BATCH_SIZE)

BATCH_SIZE = 128
TRAIN_LOADER_CIFAR, TEST_LOADER_CIFAR = load_cifar(BATCH_SIZE)
########################### ARCHITECTURES #################################

class LeNet300_100(nn.Module):
    def __init__(self, input_size=784, output_size=10):
        super(LeNet300_100, self).__init__()
        self.fc1 = nn.Linear(input_size, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Conv2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),             
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),             
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        
        self.classifier = nn.Sequential(
            nn.Linear(256*4*4, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, num_classes))
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Je garde tes anciennes classes au cas où
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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

########################### LOGIQUE MNIST #################################

def dense_neural_network_MNIST(df_accuracies, beginning, epochs=10, lr=0.01, optimizer_type="sgd"):
    print("\nStep 1 and 2: training the randomly initialized neural network for MNIST.")
    
    model = LeNet300_100(input_size=784, output_size=10)
    model.to(device)
    
    theta_0 = get_weights(model)

    criterion = nn.CrossEntropyLoss()
    
    # Selection de l'optimiseur
    if optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Entraînement initial (sans masque)
    final_loss = training_the_model(model, TRAIN_LOADER_MNIST, optimizer, criterion, num_epochs=epochs)
    
    dense_acc = evaluate_the_model(model, TEST_LOADER_MNIST)
    print(f"Initial accuracy : {dense_acc:.2f}%.")
    
    df_accuracies.append({
        "Round": "Initial model", 
        "Pruning percentage": 0, 
        "Test Accuracy (with training)": dense_acc,
        "Time (min)": (time.time() - beginning) / 60,
        "Final Training Loss": final_loss

    })
    
    return df_accuracies, model, get_weights(model), theta_0, TRAIN_LOADER_MNIST, TEST_LOADER_MNIST


def iterative_pruning_MNIST(total_prune_percent=90, rounds=8, epochs_per_round=10, lr=0.01, LTH=True, strategy_1=True, one_shot=False, optimizer_type="sgd"):
    df_accuracies = []
    beginning = time.time()

    # Appel correct avec tous les arguments
    df_accuracies, model, thetaj, theta0, TRAIN_LOADER_MNIST, TEST_LOADER_MNIST = dense_neural_network_MNIST(
        df_accuracies, beginning, epochs=epochs_per_round, lr=lr, optimizer_type=optimizer_type
    )
    
    print("\nStep 4: Creating the Winning ticket")
    criterion = nn.CrossEntropyLoss()

    if not one_shot and (total_prune_percent!=0):
        prune_percent = 1 - (1 - total_prune_percent/100)**(1/rounds)
        remaining_weights_percent = 1
        current_prune_percent = 0
        print(f"At each round, we are pruning : {prune_percent*100:.2f}% of the remaining weights.")

        for pruning_round in range(rounds):
            print(f"\n--- Round {pruning_round + 1}/{rounds} ---")
            
            current_prune_percent += remaining_weights_percent * prune_percent
            remaining_weights_percent = 1 - current_prune_percent
            
            mask = prune_by_magnitude(model, current_prune_percent*100)
            
            # Reset des poids selon la stratégie
            if LTH: 
                if strategy_1 or (pruning_round + 1 == rounds):
                    model = create_winning_ticket(model, mask, theta0)
                else:
                    model = create_winning_ticket(model, mask, thetaj)
            else:
                model = randomly_reinitialize(model, mask)

            actual_prune_percent = calculate_actual_prune_percent(model)
            print(f"Actual Global Sparsity: {actual_prune_percent:.2f}%")

            acc = evaluate_the_model(model, TEST_LOADER_MNIST)
            
            if optimizer_type.lower() == "sgd":
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            final_loss = training_the_model(model, TRAIN_LOADER_MNIST, optimizer, criterion, epochs_per_round, mask=mask)
            
            thetaj = get_weights(model)
            acc_post_training = evaluate_the_model(model, TEST_LOADER_MNIST) 
            print(f"Accuracy after retraining: {acc_post_training:.2f}%")
            
            df_accuracies.append({
                "Round": f"Round {pruning_round + 1}", 
                "Pruning percentage": actual_prune_percent, 
                "Test Accuracy (no retraining)": acc, 
                "Test Accuracy (with training)": acc_post_training,
                "Time (min)": (time.time() - beginning) / 60,
                "Final Training Loss": final_loss
            })
            
    elif one_shot: # One Shot Logic
        mask = prune_by_magnitude(model, total_prune_percent)
        if LTH: 
            model = create_winning_ticket(model, mask, theta0)
        else: 
            model = randomly_reinitialize(model, mask)

        actual_prune_percent = calculate_actual_prune_percent(model)
        print(f"Current pruning percentage: {actual_prune_percent:.2f}%")

        acc = evaluate_the_model(model, TEST_LOADER_MNIST)

        if optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        final_loss = training_the_model(model, TRAIN_LOADER_MNIST, optimizer, criterion, epochs_per_round, mask=mask)
        
        thetaj = get_weights(model)
        acc_post_training = evaluate_the_model(model, TEST_LOADER_MNIST)
        print(f"Accuracy after retraining: {acc_post_training:.2f}%")
        
        df_accuracies.append({
            "Round": "One_shot", 
            "Pruning percentage": actual_prune_percent, 
            "Test Accuracy (no retraining)": acc, 
            "Test Accuracy (with training)": acc_post_training,
            "Time (min)": (time.time() - beginning) / 60,
            "Final Training Loss": final_loss
        })
            
    return df_accuracies, model


########################### LOGIQUE CIFAR #################################

def dense_neural_network_CIFAR(df_accuracies, beginning, epochs=10, lr=0.01, optimizer_type="sgd"):
    print("\nStep 1 and 2: training the randomly initialized neural network for CIFAR.")
    
    model = Conv2(num_classes=10)
    model.to(device)
    
    theta_0 = get_weights(model)

    criterion = nn.CrossEntropyLoss()
    
    if optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        print("   -> Optimizer used: SGD (momentum=0.9)")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        print("   -> Optimizer used: Adam")

    final_loss = training_the_model(model, TRAIN_LOADER_CIFAR, optimizer, criterion, num_epochs=epochs) 
    
    dense_acc = evaluate_the_model(model, TEST_LOADER_CIFAR)
    print(f"Initial accuracy : {dense_acc:.2f}%.")
    
    df_accuracies.append({
        "Round": "Initial model", 
        "Pruning percentage": 0, 
        "Test Accuracy (with training)": dense_acc,
        "Time (min)": (time.time() - beginning) / 60,
        "Final Training Loss": final_loss
    })
    
    return df_accuracies, model, get_weights(model), theta_0, TRAIN_LOADER_CIFAR, TEST_LOADER_CIFAR


def iterative_pruning_CIFAR(total_prune_percent=90, rounds=8, epochs_per_round=10, lr=0.01, LTH=True, strategy_1=True, one_shot=False, optimizer_type="sgd"):
    df_accuracies = []
    beginning = time.time()

    df_accuracies, model, thetaj, theta0, TRAIN_LOADER_CIFAR, TEST_LOADER_CIFAR = dense_neural_network_CIFAR(
        df_accuracies, beginning, epochs=epochs_per_round, lr=lr, optimizer_type=optimizer_type
    )
    
    print("\nStep 4: Creating the Winning ticket")
    criterion = nn.CrossEntropyLoss()

    if not one_shot and (total_prune_percent!=0):
        prune_percent = 1 - (1 - total_prune_percent/100)**(1/rounds)
        remaining_weights_percent = 1
        current_prune_percent = 0
        print(f"At each round, we are pruning : {prune_percent*100:.2f}% of the weights.")

        for pruning_round in range(rounds):
            print(f"\n--- Round {pruning_round + 1}/{rounds} ---")
            
            current_prune_percent += remaining_weights_percent * prune_percent
            remaining_weights_percent = 1 - current_prune_percent
            
            mask = prune_by_magnitude(model, current_prune_percent*100)
            
            if LTH: 
                if strategy_1 or (pruning_round + 1 == rounds):
                    model = create_winning_ticket(model, mask, theta0)
                else:
                    model = create_winning_ticket(model, mask, thetaj)
            else:
                model = randomly_reinitialize(model, mask)

            actual_prune_percent = calculate_actual_prune_percent(model)
            print(f"Current pruning percentage: {actual_prune_percent:.2f}%")
            
            acc = evaluate_the_model(model, TEST_LOADER_CIFAR) 

            if optimizer_type.lower() == "sgd":
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            final_loss = training_the_model(model, TRAIN_LOADER_CIFAR, optimizer, criterion, epochs_per_round, mask=mask)
            
            thetaj = get_weights(model)
            acc_post_training = evaluate_the_model(model, TEST_LOADER_CIFAR)
            print(f"Accuracy after retraining: {acc_post_training:.2f}%")
            
            df_accuracies.append({
                "Round": f"Round {pruning_round + 1}", 
                "Pruning percentage": actual_prune_percent, 
                "Test Accuracy (no retraining)": acc, 
                "Test Accuracy (with training)": acc_post_training,
                "Time (min)": (time.time() - beginning) / 60,
                "Final Training Loss": final_loss
            })
            
    elif one_shot: # One Shot
        mask = prune_by_magnitude(model, total_prune_percent)
        if LTH: 
            model = create_winning_ticket(model, mask, theta0)
        else: 
            model = randomly_reinitialize(model, mask)

        actual_prune_percent = calculate_actual_prune_percent(model)
        print(f"Current pruning percentage: {actual_prune_percent:.2f}%")

        acc = evaluate_the_model(model, TEST_LOADER_CIFAR) 

        if optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        final_loss = training_the_model(model, TRAIN_LOADER_CIFAR, optimizer, criterion, epochs_per_round, mask=mask)
        
        thetaj = get_weights(model)
        acc_post_training = evaluate_the_model(model, TEST_LOADER_CIFAR)
        print(f"Accuracy after retraining: {acc_post_training:.2f}%")
        
        df_accuracies.append({
            "Round": "One_shot", 
            "Pruning percentage": actual_prune_percent, 
            "Test Accuracy (no retraining)": acc, 
            "Test Accuracy (with training)": acc_post_training,
            "Time (min)": (time.time() - beginning) / 60,
            "Final Training Loss": final_loss
        })

    return df_accuracies, model