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

# --- 1. DÉTECTION GPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Neural Networks loaded. User device: {device}")

########################### ARCHITECTURES #################################

# 1. LeNet-300-100 (Architecture officielle pour MNIST)
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

# 2. Conv-2 (Architecture officielle pour CIFAR-10 / VGG-style)
class Conv2(nn.Module):
    def __init__(self, output_size=10):
        super(Conv2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Après 2 convs et 1 pool sur 32x32 -> 16x16
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 3. SimpleNN (Ton ancien modèle, gardé pour compatibilité)
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

# 4. SimpleCNN (Ton ancien modèle CIFAR, gardé pour compatibilité)
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

########################### LOGIQUE D'ENTRAINEMENT (MNIST) #################################

# Step 1 and 2: train the randomly initialized neural network 
def dense_neural_network_MNIST(df_accuracies, epochs=10, lr=1.2e-3):
    print("\nStep 1 and 2: training the randomly initialized neural network for MNIST.")
    
    # --- MODIF ARCHITECTURE & GPU ---
    # On utilise LeNet300_100 au lieu de SimpleNN
    model = LeNet300_100(input_size=784, output_size=10)
    model.to(device) # <--- Envoi sur GPU
    
    # Sauvegarde Theta 0
    theta_0 = get_weights(model)

    batch_size = 60 # Ajusté pour LeNet
    train_loader, test_loader = load_mnist(batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    training_the_model(model, train_loader, optimizer, criterion, num_epochs=epochs)
    
    dense_acc = evaluate_the_model(model, test_loader)
    print(f"Initial accuracy : {dense_acc:.2f}%.")
    
    df_accuracies.append({
        "Round": "Initial model", 
        "Pruning percentage": 0, 
        "Test Accuracy (with training)": dense_acc
    })
    
    return df_accuracies, model, get_weights(model), theta_0, train_loader, test_loader

# Step 4: creating the winning ticket f(x; m⊙theta_0)
def iterative_pruning_MNIST(total_prune_percent=90, rounds=8, epochs_per_round=10, lr=1.2e-3, LTH=True, strategy_1=True, one_shot=False):
    df_accuracies = []
    
    # Appel de la fonction dense (adaptée pour accepter epochs et lr)
    df_accuracies, model, thetaj, theta0, train_loader, test_loader = dense_neural_network_MNIST(
        df_accuracies, epochs=epochs_per_round, lr=lr
    )
    
    print("\nStep 4: Creating the Winning ticket")
    # print(f"Number of zeros before pruning: {count_zeros(model)}")

    criterion = nn.CrossEntropyLoss()

    if not one_shot:
        # Formule IMP : p = 1 - (1 - P_total)^(1/n)
        prune_percent = 1 - (1 - total_prune_percent/100)**(1/rounds)
        remaining_weights_percent = 1
        current_prune_percent = 0
        print(f"At each round, we are pruning : {prune_percent*100:.2f}% of the remaining weights.")

        for pruning_round in range(rounds):
            print(f"\n--- Round {pruning_round + 1}/{rounds} ---")
            
            # Calcul cumulatif comme dans ton code original
            current_prune_percent += remaining_weights_percent * prune_percent
            remaining_weights_percent = 1 - current_prune_percent
            
            print(f"Current pruning percentage (Method 1): {current_prune_percent*100:.2f}%")
            
            # Pruning (On passe le pourcentage total cumulé)
            mask = prune_by_magnitude(model, current_prune_percent*100)
            
            # Reset Logique (Strategy 1 vs 2)
            if LTH: 
                if strategy_1 or (pruning_round + 1 == rounds): # Reset à Theta 0
                    model = create_winning_ticket(model, mask, theta0)
                else: # Fine tuning (Keep weights)
                    model = create_winning_ticket(model, mask, thetaj)
            else: # Random Reinit
                model = randomly_reinitialize(model, mask)

            actual_prune_percent = calculate_actual_prune_percent(model)
            print(f"Current pruning percentage: {actual_prune_percent:.2f}%")

            acc = evaluate_the_model(model, test_loader)
            # print(f"Accuracy after pruning (no retraining): {acc:.2f}%")

            # Retrain
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            training_the_model(model, train_loader, optimizer, criterion, epochs_per_round)
            thetaj = get_weights(model)

            # Test Accuracies
            acc_post_training = evaluate_the_model(model, test_loader)
            print(f"Accuracy after retraining: {acc_post_training:.2f}%")
            
            df_accuracies.append({
                "Round": f"Round {pruning_round + 1}", 
                "Pruning percentage": actual_prune_percent, 
                "Test Accuracy (no retraining)": acc, 
                "Test Accuracy (with training)": acc_post_training
            })
            
    else: # One Shot Logic
        mask = prune_by_magnitude(model, total_prune_percent)
        if LTH: 
            model = create_winning_ticket(model, mask, theta0)
        else: 
            # Note: Si One shot et LTH=False, on fait du random
            model = randomly_reinitialize(model, mask)

        actual_prune_percent = calculate_actual_prune_percent(model)
        print(f"Current pruning percentage: {actual_prune_percent:.2f}%")

        acc = evaluate_the_model(model, test_loader)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        training_the_model(model, train_loader, optimizer, criterion, epochs_per_round)
        thetaj = get_weights(model)

        acc_post_training = evaluate_the_model(model, test_loader)
        print(f"Accuracy after retraining: {acc_post_training:.2f}%")
        
        df_accuracies.append({
            "Round": "One_shot", 
            "Pruning percentage": actual_prune_percent, 
            "Test Accuracy (no retraining)": acc, 
            "Test Accuracy (with training)": acc_post_training
        })
            
    return df_accuracies, model


########################### LOGIQUE D'ENTRAINEMENT (CIFAR-10) #################################

def dense_neural_network_CIFAR(df_accuracies, epochs=10, lr=2e-4):
    print("\nStep 1 and 2: training the randomly initialized neural network for CIFAR.")
    
    # --- MODIF ARCHITECTURE & GPU ---
    # On utilise Conv2 au lieu de SimpleCNN pour CIFAR
    model = Conv2(output_size=10)
    model.to(device) # <--- Envoi sur GPU
    
    theta_0 = get_weights(model)

    batch_size = 60
    train_loader, test_loader = load_cifar(batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    training_the_model(model, train_loader, optimizer, criterion, num_epochs=epochs)
    
    dense_acc = evaluate_the_model(model, test_loader)
    print(f"Initial accuracy : {dense_acc:.2f}%.")
    
    df_accuracies.append({
        "Round": "Initial model", 
        "Test Accuracy (with training)": dense_acc
    })
    
    return df_accuracies, model, get_weights(model), theta_0, train_loader, test_loader


def iterative_pruning_CIFAR(total_prune_percent=90, rounds=8, epochs_per_round=10, lr=2e-4, LTH=True, strategy_1=True, one_shot=False):
    df_accuracies = []
    
    df_accuracies, model, thetaj, theta0, train_loader, test_loader = dense_neural_network_CIFAR(
        df_accuracies, epochs=epochs_per_round, lr=lr
    )
    
    print("\nStep 4: Creating the Winning ticket")
    # print(f"Number of zeros before pruning: {count_zeros(model)}")
    
    criterion = nn.CrossEntropyLoss()

    if not one_shot:
        prune_percent = 1 - (1 - total_prune_percent/100)**(1/rounds)
        remaining_weights_percent = 1
        current_prune_percent = 0
        print(f"At each round, we are pruning : {prune_percent*100:.2f}% of the weights.")

        for pruning_round in range(rounds):
            print(f"\n--- Round {pruning_round + 1}/{rounds} ---")
            
            current_prune_percent += remaining_weights_percent * prune_percent
            remaining_weights_percent = 1 - current_prune_percent
            print(f"Current pruning percentage (Method 1): {current_prune_percent*100:.2f}%")
            
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
            
            acc = evaluate_the_model(model, test_loader)
            #print(f"Accuracy after pruning (no retraining): {acc:.2f}%")

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            training_the_model(model, train_loader, optimizer, criterion, epochs_per_round)
            thetaj = get_weights(model)

            acc_post_training = evaluate_the_model(model, test_loader)
            print(f"Accuracy after retraining: {acc_post_training:.2f}%")
            
            df_accuracies.append({
                "Round": f"Round {pruning_round + 1}", 
                "Pruning percentage": actual_prune_percent, 
                "Test Accuracy (no retraining)": acc, 
                "Test Accuracy (with training)": acc_post_training
            })
            
    else: # One Shot
        mask = prune_by_magnitude(model, total_prune_percent)
        if LTH: 
            model = create_winning_ticket(model, mask, theta0)
        else: 
            model = randomly_reinitialize(model, mask)

        actual_prune_percent = calculate_actual_prune_percent(model)
        print(f"Current pruning percentage: {actual_prune_percent:.2f}%")

        acc = evaluate_the_model(model, test_loader)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        training_the_model(model, train_loader, optimizer, criterion, epochs_per_round)
        thetaj = get_weights(model)

        acc_post_training = evaluate_the_model(model, test_loader)
        print(f"Accuracy after retraining: {acc_post_training:.2f}%")
        
        df_accuracies.append({
            "Round": "One_shot", 
            "Pruning percentage": actual_prune_percent, 
            "Test Accuracy (no retraining)": acc, 
            "Test Accuracy (with training)": acc_post_training
        })

    return df_accuracies, model