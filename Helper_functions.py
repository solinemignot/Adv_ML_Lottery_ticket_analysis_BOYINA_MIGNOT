import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from Accessing_data import load_mnist, load_cifar

# --- 1. DÉTECTION GLOBALE DU GPU ---
# C'est la ligne la plus importante : elle décide où les calculs se font.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Helper functions loaded. User device: {device}")

########################### Helper functions #################################

def training_the_model(model, train_loader, optimizer, criterion, num_epochs=10):
    # On s'assure que le modèle est sur le bon device
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            # --- GPU MAGIC : Envoi des données vers la carte graphique ---
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # --- MASQUAGE DES GRADIENTS ---
            # Si le modèle a un masque (après le 1er round), on force le gradient des 
            # poids prunés à 0 pour qu'ils ne "ressuscitent" pas pendant la descente de gradient.
            if hasattr(model, 'mask') and model.mask is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in model.mask:
                            param.grad *= model.mask[name]
            
            optimizer.step()
            running_loss += loss.item()

def evaluate_the_model(model, test_loader):
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
    # OPTIMISATION MEMOIRE : On stocke les sauvegardes (Theta0, Thetaj) sur le CPU.
    # Sinon, on va saturer la mémoire du GPU très vite.
    return {name: param.clone().cpu() for name, param in model.named_parameters()}

def apply_mask(model, mask):
    # Utile pour forcer l'application du masque hors entraînement
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                param.data *= mask[name]

def calculate_actual_prune_percent(model):
    total_weights = sum(p.numel() for p in model.parameters() if p.dim() > 1)
    zero_weights = sum((p == 0).sum().item() for p in model.parameters() if p.dim() > 1)
    if total_weights == 0: return 0
    return 100 * zero_weights / total_weights

def count_zeros(model):
    return sum((p == 0).sum().item() for p in model.parameters() if p.dim() > 1)


########################### Lottery Ticket Algorithm functions #################################

def prune_by_magnitude(model, prune_percent=20):
    """
    Layer-wise pruning qui respecte la règle du papier :
    - On calcule un seuil par couche.
    - La dernière couche (output) est élaguée moitié moins fort.
    - Tout est géré sur GPU.
    """
    mask = {}
    
    # 1. Identification dernière couche
    last_weight_layer_name = None
    for name, param in model.named_parameters():
        if param.dim() > 1:
            last_weight_layer_name = name
            
    # 2. Pruning
    print(f"   [Pruning Debug] Target Global Percent: {prune_percent:.2f}%")
    
    for name, param in model.named_parameters():
        if param.dim() > 1:
            current_prune_percent = prune_percent
            
            # Gestion couche de sortie
            if name == last_weight_layer_name:
                current_prune_percent = prune_percent / 2
                print(f"   -> Output Layer detected ({name}): Pruning half ({current_prune_percent:.2f}%)")
            
            k = int(param.numel() * current_prune_percent / 100)
            
            if k > 0:
                threshold = torch.topk(param.data.abs().view(-1), k, largest=False).values.max()
                mask[name] = (param.data.abs() > threshold).float().to(device)
                
                # --- NOUVEAU : PRINT COMME TU VEUX ---
                zeros = (mask[name] == 0).sum().item()
                total = mask[name].numel()
                print(f"      Layer {name:<10} | Threshold: {threshold:.5f} | Zeros: {zeros}/{total} ({100*zeros/total:.1f}%)")
                
            else:
                mask[name] = torch.ones_like(param).to(device)
                print(f"      Layer {name:<10} | Keep All (k=0)")
                
        else:
            mask[name] = torch.ones_like(param).to(device)
            
    return mask

def create_winning_ticket(model, mask, theta):
    """
    Réinitialise les poids à leur valeur d'origine (Theta 0) tout en appliquant le masque.
    Gère le transfert CPU (theta stocké) -> GPU (modèle actif).
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                # theta[name] est sur CPU, param est sur GPU. On envoie theta sur GPU.
                original_weights = theta[name].to(device)
                param.data = original_weights * mask[name]
    
    # Petit check de debug
    # zero_count = (param == 0).sum().item()
    # print(f"Ticket Reset -> Layer {name}: {zero_count} zeros")
    return model

def randomly_reinitialize(model, mask):
    """
    Réinitialise aléatoirement le réseau (Random Ticket) mais garde la structure du masque.
    Utilise l'initialisation standard de PyTorch (Kaiming/Xavier) puis remet les zéros.
    """
    # 1. On réinitialise tout le module proprement
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            module.reset_parameters()
            
    # 2. On réapplique les zéros du masque
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                 param.data *= mask[name]
    return model

