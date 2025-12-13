import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from Accessing_data import load_mnist

"""
xxx - Goal
"""
    
########################### Helper functions #################################################################

def training_the_model(model, train_loader, optimizer, criterion, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

def evaluate_the_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    return acc

def get_weights(model):
    return {name: param.clone() for name, param in model.named_parameters()}

def apply_mask(model, mask):
    with torch.no_grad():
        for name, param in model.named_parameters():
            param *= mask[name]

def calculate_actual_prune_percent(model):
    total_weights = sum(p.numel() for p in model.parameters() if p.dim() > 1)
    zero_weights = sum((p == 0).sum().item() for p in model.parameters() if p.dim() > 1)
    return 100 * zero_weights / total_weights

def count_zeros(model):
    return sum((p == 0).sum().item() for p in model.parameters() if p.dim() > 1)


########################### Lottery Ticket Algorithm functions #################################################################
# Step 3: Prune the smallest weights

"""
Cette fonction prune tous les poids du réseau pour calculer un seuil unique et ensuite élaguer.
Cela signifie que si une couche a naturellement des poids plus petits qu'une autre, elle pourrait être entièrement effacée.
Dans l'article (Layer-wise Pruning) :Pour les réseaux Fully-Connected (MNIST) et les petits ConvNets, 
les auteurs utilisent le Layer-wise pruning. 
Ils calculent un seuil différent pour chaque couche afin d'enlever P% des poids de cette couche spécifique. 

-> Il faut calculer le seuil couche par couche.

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
"""

# In the article, the authors mention that for the output layer, they prune at half the rate.
# Therefore, we need to identify the last layer dynamically and apply a different pruning rate to it
#This function prunes weights layer-wise and applies half the pruning rate to the last layer.
def prune_by_magnitude(model, prune_percent=20):
    mask = {}
    
    # 1. DYNAMIC IDENTIFICATION OF THE LAST LAYER
    # Iterate through all parameters to find the name of the last parameter that is a weight (dim > 1)
    last_weight_layer_name = None
    for name, param in model.named_parameters():
        if param.dim() > 1: # Ignore biases
            last_weight_layer_name = name
            
    # At this point, last_weight_layer_name contains (e.g.) "fc3.weight" or "classifier.6.weight"

    # 2. COMPUTE MASKS
    for name, param in model.named_parameters():
        if param.dim() > 1:
            # By default, use the standard pruning rate
            current_prune_percent = prune_percent
            
            # If it's the last layer found previously -> Half the pruning rate
            if name == last_weight_layer_name:
                current_prune_percent = prune_percent / 2
                print(f"Output layer detected ({name}): Pruning at {current_prune_percent}% (vs {prune_percent}%)")
            
            # Calculate threshold (Layer-wise)
            k = int(param.numel() * current_prune_percent / 100)
            
            if k > 0:
                # Flatten weights, find the k-th smallest weight value
                threshold = torch.topk(param.data.abs().view(-1), k, largest=False).values.max()
                mask[name] = (param.data.abs() > threshold).float()
            else:
                # If k=0 (layer too small or % too low), do not prune anything
                mask[name] = torch.ones_like(param)
                
        else:
            # Never prune biases (dim=1)
            mask[name] = torch.ones_like(param)
            
    return mask

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













