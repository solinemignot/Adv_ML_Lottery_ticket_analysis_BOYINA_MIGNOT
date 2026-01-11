import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from Helper_functions import training_the_model, evaluate_the_model
from Accessing_data import load_cifar
from Neural_networks import *
from Helper_functions import *


####################################################################
#  EXTENSIONS : "Late Rewinding" (Reset at Epoch k) for CIFAR-10 ---


def calculate_loss(model, loader, criterion):
    """
    This helper function computes the average loss of the model over a specific dataset (training or testing) 
    without updating any gradients, effectively monitoring the model's convergence state.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            total_samples += labels.size(0)
    return running_loss / total_samples


def dense_neural_network_CIFAR_Rewinding(df_accuracies, epochs=10, rewind_epoch=2, lr=0.1, optimizer_type="sgd"):
    """
    This function trains the initial dense network from scratch but pauses at a specific "rewind epoch" (epoch $k$) 
    to save a snapshot of the weights ($\theta_k$), which serves as the stable initialization point for future pruning rounds.
    """
    print(f"\nStep 1: Training Dense Network with Rewinding capture at Epoch {rewind_epoch}...")
    
    start_total = time.time()
    model = Conv2(num_classes=10).to(device)
    batch_size = 60
    train_loader, test_loader = load_cifar(batch_size)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"   -> Warming up for {rewind_epoch} epochs...")
    training_the_model(model, train_loader, optimizer, criterion, num_epochs=rewind_epoch)
    
    theta_k = get_weights(model) 
    print(f"   -> Theta_{rewind_epoch} saved! (Checkpoint created)")

    remaining_epochs = epochs - rewind_epoch
    if remaining_epochs > 0:
        training_the_model(model, train_loader, optimizer, criterion, num_epochs=remaining_epochs)
    
    dense_acc = evaluate_the_model(model, test_loader)
    dense_loss = calculate_loss(model, train_loader, criterion) 
    
    end_total = time.time()
    duration = (end_total - start_total) / 60 

    print(f"Initial dense accuracy : {dense_acc:.2f}%. Loss: {dense_loss:.4f}")
    
    df_accuracies.append({
        "Round": "Initial model", 
        "Pruning percentage": 0, 
        "Test Accuracy (with training)": dense_acc,
        "Time (min)": duration,
        "Final Training Loss": dense_loss 
    })
    
    return df_accuracies, model, get_weights(model), theta_k, train_loader, test_loader



def iterative_pruning_CIFAR_Rewinding(total_prune_percent=90, rounds=8, epochs_per_round=10, rewind_epoch=2, lr=0.1, optimizer_type="sgd"):
    """
    This function orchestrates the full Late Rewinding experiment by iteratively pruning the network and resetting the remaining weights 
    to the saved state $\theta_k$ (instead of epoch 0) to ensure stability at high sparsity levels.
    """
    df_accuracies = []
    
    df_accuracies, model, thetaj, theta_k, train_loader, test_loader = dense_neural_network_CIFAR_Rewinding(
        df_accuracies, epochs=epochs_per_round, rewind_epoch=rewind_epoch, lr=lr, optimizer_type=optimizer_type
    )
    
    print(f"\nStep 4: Creating the Winning ticket (Reset to Epoch {rewind_epoch})")
    criterion = nn.CrossEntropyLoss()

    prune_percent = 1 - (1 - total_prune_percent/100)**(1/rounds)
    remaining_weights_percent = 1
    current_prune_percent = 0
    
    for pruning_round in range(rounds):
        start_round = time.time()
        print(f"\n--- Round {pruning_round + 1}/{rounds} ---")
        
        current_prune_percent += remaining_weights_percent * prune_percent
        remaining_weights_percent = 1 - current_prune_percent
        
        mask = prune_by_magnitude(model, current_prune_percent*100)
        
        model = create_winning_ticket(model, mask, theta_k)
        
        actual_sparsity = calculate_actual_prune_percent(model)
        
        acc = evaluate_the_model(model, test_loader)
        
        if optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        training_the_model(model, train_loader, optimizer, criterion, epochs_per_round, mask=mask)
        
        acc_post_training = evaluate_the_model(model, test_loader)
        final_loss = calculate_loss(model, train_loader, criterion) 
        
        end_round = time.time()
        duration = (end_round - start_round) / 60
        
        print(f"Accuracy: {acc_post_training:.2f}%. Loss: {final_loss:.4f}")
        
        df_accuracies.append({
            "Round": f"Round {pruning_round + 1}", 
            "Pruning percentage": actual_sparsity, 
            "Test Accuracy (no retraining)": acc, 
            "Test Accuracy (with training)": acc_post_training,
            "Time (min)": duration,
            "Final Training Loss": final_loss 
        })
            
    return df_accuracies, model



#########  EXTENSIONS STRONG LTH (Edge-Popup Algorithm) ###############

class GetSubnet(torch.autograd.Function):
    """
    This class implements the "Straight-Through Estimator," which allows the network to generate a discrete binary mask (0 or 1) 
    during the forward pass while still permitting gradients to flow back to update the continuous scores during the backward pass.
    """
    @staticmethod
    def forward(ctx, scores, k):
        out = scores.clone()
        
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())
        
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1
        
        return out

    @staticmethod
    def backward(ctx, g):
        return g, None

class SupermaskConv2d(nn.Conv2d):
    """
    A custom convolutional layer that freezes the actual weight parameters and instead optimizes a separate "score" matrix 
    to learn which connections should remain active (top-$k$ scores) and which should be pruned.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        self.weight.requires_grad = False 
        self.sparsity_ratio = 1.0 

    def set_prune_rate(self, prune_rate):
        """Définit le % de poids à garder (k)"""
        self.sparsity_ratio = 1.0 - (prune_rate / 100.0)

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), self.sparsity_ratio)
        w = self.weight * subnet
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

class SupermaskLinear(nn.Linear):
    """
    Similar to SupermaskConv2d, this is a custom fully connected layer that keeps the weights frozen and learns a topology mask 
    based on learnable scores to determine connectivity.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        
        self.weight.requires_grad = False 
        self.sparsity_ratio = 1.0

    def set_prune_rate(self, prune_rate):
        self.sparsity_ratio = 1.0 - (prune_rate / 100.0)

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), self.sparsity_ratio)
        w = self.weight * subnet
        return F.linear(x, w, self.bias)

class Conv2_Strong(nn.Module):
    """
    This defines the specific neural network architecture for the Strong LTH experiment by replacing standard layers with Supermask
    layers, enabling the model to learn a structure rather than weight values.
    """
    def __init__(self, output_size=10):
        super(Conv2_Strong, self).__init__()
        self.conv1 = SupermaskConv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = SupermaskConv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = SupermaskLinear(64 * 16 * 16, 256)
        self.fc2 = SupermaskLinear(256, 256)
        self.fc3 = SupermaskLinear(256, output_size)

    def set_global_pruning_rate(self, prune_rate):
        """
        This utility method iterates through every layer of the network to enforce a global sparsity constraint, 
        setting the percentage of weights that must be masked out in the Supermask layers.
        """
        for module in self.modules():
            if isinstance(module, (SupermaskConv2d, SupermaskLinear)):
                module.set_prune_rate(prune_rate)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#  main function to run Strong LTH experiment
def run_strong_lth_experiment(target_prune_percent=50, epochs=20, lr=0.1):
    """
    This function executes the Strong LTH experiment by initializing a network with frozen random weights and training 
    only the mask scores (Edge-Popup) to discover a high-performing subnetwork at a specific sparsity target.
    """
    print(f"\n Lancement Strong LTH (Edge-Popup) | Cible Sparsité: {target_prune_percent}%")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Conv2_Strong(output_size=10).to(device)
    
    model.set_global_pruning_rate(target_prune_percent)
    
    train_loader, test_loader = load_cifar(batch_size=60)
    
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], 
        lr=lr, 
        momentum=0.9, 
        weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    print("-> Training mask scores...")
    training_the_model(model, train_loader, optimizer, criterion, num_epochs=epochs)
    
    final_acc = evaluate_the_model(model, test_loader)
    print(f"Résultat Strong LTH : {final_acc:.2f}% d'accuracy avec {target_prune_percent}% de poids supprimés (et poids figés!).")
    
    return final_acc