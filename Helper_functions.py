import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from Accessing_data import load_mnist, load_cifar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Helper functions loaded. User device: {device}")

########################### Helper functions #################################

def training_the_model(model, train_loader, optimizer, criterion, num_epochs=10, mask=None):
    """
    In this function, the model is trained during num_epochs epochs.
    The output of the function the final training loss after those epochs. 
    """
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Mask gradients
            if mask is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in mask:
                            param.grad.mul_(mask[name].to(device))

            optimizer.step()

            # Hard mask weights (CRITICAL)
            if mask is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in mask:
                            param.mul_(mask[name].to(device))

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)

    return epoch_loss
        

def evaluate_the_model(model, test_loader):
    """
    This function outputs the test accuract of the model.
    """
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            # --- GPU MAGIC ---
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    return acc

def get_weights(model):
    return {name: param.clone().detach().cpu() for name, param in model.named_parameters()}

def apply_mask(model, mask):
    """
    Given the mask, it masks those weights for the model.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                param.data *= mask[name].to(device)

def calculate_actual_prune_percent(model):
    """
    To calculate the pruning percentage. Counts the amount of weights that are null.
    That way, we can check the masking is going well.
    """
    total_weights = sum(p.numel() for p in model.parameters() if p.dim() > 1)
    zero_weights = sum((p == 0).sum().item() for p in model.parameters() if p.dim() > 1)
    if total_weights == 0: return 0
    return 100 * zero_weights / total_weights

def count_zeros(model):
    return sum((p == 0).sum().item() for p in model.parameters() if p.dim() > 1)


########################### Lottery Ticket Algorithm functions #################################

def prune_by_magnitude(model, prune_percent=20):
    mask = {}
    
    last_weight_layer_name = None
    for name, param in model.named_parameters():
        if param.dim() > 1:
            last_weight_layer_name = name
            
    #Pruning
    for name, param in model.named_parameters():
        if param.dim() > 1:
            current_prune_percent = prune_percent
            if name == last_weight_layer_name:
                current_prune_percent = prune_percent / 2
            k = int(param.numel() * current_prune_percent / 100)
            
            if k > 0:
                threshold = torch.topk(param.data.abs().view(-1), k, largest=False).values.max()
                mask[name] = (param.data.abs() > threshold).float().to(device)
            else:
                mask[name] = torch.ones_like(param).to(device)
                
        else:
            mask[name] = torch.ones_like(param).to(device)
            
    return mask

def create_winning_ticket(model, mask, theta):
    """
    Reinitializes the model to the theta weights.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                # theta[name] est sur CPU, param est sur GPU. On envoie theta sur GPU.
                original_weights = theta[name].to(device)
                param.data = original_weights * mask[name].to(device)
            elif name in theta:
                 # Pour les params sans masque (ex: biais), on reset aussi à l'initial
                 param.data = theta[name].to(device)
    
    return model

def randomly_reinitialize(model, mask):
    """
    Randomly reinitializes the weights that are not masked in the model.
    """
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    model.apply(init_weights)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                 param.data *= mask[name].to(device)
    return model