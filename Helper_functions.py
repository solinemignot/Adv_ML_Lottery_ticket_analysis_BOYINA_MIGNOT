import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from Accessing_data import load_mnist, load_cifar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Helper functions loaded. User device: {device}")

# --- 1. DÉTECTION GLOBALE DU GPU ---
########################### Helper functions #################################

def training_the_model(model, train_loader, optimizer, criterion, num_epochs=10, mask=None):
    """
    CORRECTION : Ajout de l'argument 'mask=None' pour fixer l'erreur TypeError.
    Application stricte du masque sur les gradients.
    """
    # On s'assure que le modèle est sur le bon device
    model.to(device)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            # --- GPU MAGIC : Envoi des données vers la carte graphique ---
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()
            
            # --- MASQUAGE DES GRADIENTS (CORRIGÉ) ---
            # On utilise l'argument 'mask' passé à la fonction.
            # On force le gradient à 0 pour que les poids prunés ne changent pas.
            if mask is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in mask:
                            # On s'assure que le masque est sur le GPU
                            param.grad *= mask[name].to(device)
            
            optimizer.step()
            running_loss += loss.item()
        epoch_loss /= len(train_loader)
        return epoch_loss
        
    

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
    return {name: param.clone().detach().cpu() for name, param in model.named_parameters()}

def apply_mask(model, mask):
    # Utile pour forcer l'application du masque hors entraînement
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                param.data *= mask[name].to(device)

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
    Layer-wise pruning qui respecte la règle du papier.
    """
    mask = {}
    
    # 1. Identification dernière couche
    last_weight_layer_name = None
    for name, param in model.named_parameters():
        if param.dim() > 1:
            last_weight_layer_name = name
            
    # 2. Pruning
    # print(f"   [Pruning Debug] Target Global Percent: {prune_percent:.2f}%")
    
    for name, param in model.named_parameters():
        if param.dim() > 1:
            current_prune_percent = prune_percent
            
            # Gestion couche de sortie (souvent élaguée moins agressivement ou pas du tout)
            if name == last_weight_layer_name:
                current_prune_percent = prune_percent / 2
                # print(f"   -> Output Layer detected ({name}): Pruning half ({current_prune_percent:.2f}%)")
            
            # Calcul du nombre de poids à couper
            k = int(param.numel() * current_prune_percent / 100)
            
            if k > 0:
                # On utilise abs() pour la magnitude
                threshold = torch.topk(param.data.abs().view(-1), k, largest=False).values.max()
                # 1 si > seuil, 0 sinon
                mask[name] = (param.data.abs() > threshold).float().to(device)
                
            else:
                mask[name] = torch.ones_like(param).to(device)
                
        else:
            # On garde les biais et autres paramètres 1D
            mask[name] = torch.ones_like(param).to(device)
            
    return mask

def create_winning_ticket(model, mask, theta):
    """
    Réinitialise les poids à leur valeur d'origine (Theta 0) tout en appliquant le masque.
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
    Réinitialise aléatoirement le réseau (Random Ticket) mais garde la structure du masque.
    """
    # 1. On réinitialise tout le module proprement
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
            
    # 2. On réapplique les zéros du masque
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                 param.data *= mask[name].to(device)
    return model