import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from Helper_functions import training_the_model, evaluate_the_model
from Accessing_data import load_cifar
from Neural_networks import *
from Helper_functions import *

# --- Méthode "Late Rewinding" (Reset à Epoch $k$) pour CIFAR-10 ---

# --- A. Fonction Dense qui capture le "Checkpoint" (Theta_k) ---
def dense_neural_network_CIFAR_Rewinding(df_accuracies, epochs=10, rewind_epoch=2, lr=0.1, optimizer_type="sgd"):
    print(f"\nStep 1: Training Dense Network with Rewinding capture at Epoch {rewind_epoch}...")
    
    model = Conv2(output_size=10).to(device)
    
    # On charge les données
    batch_size = 60
    train_loader, test_loader = load_cifar(batch_size)
    criterion = nn.CrossEntropyLoss()
    
    # Optimiseur
    if optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 1. WARMUP : On entraîne jusqu'à l'époque de rewind
    print(f"   -> Warming up for {rewind_epoch} epochs...")
    training_the_model(model, train_loader, optimizer, criterion, num_epochs=rewind_epoch)
    
    # 2. CAPTURE DU CHECKPOINT : Theta_k (l'état stabilisé)
    theta_k = get_weights(model) 
    print(f"   -> Theta_{rewind_epoch} saved! (Checkpoint created)")

    # 3. FIN DE L'ENTRAÎNEMENT DENSE
    remaining_epochs = epochs - rewind_epoch
    if remaining_epochs > 0:
        training_the_model(model, train_loader, optimizer, criterion, num_epochs=remaining_epochs)
    
    dense_acc = evaluate_the_model(model, test_loader)
    print(f"Initial dense accuracy : {dense_acc:.2f}%.")
    
    df_accuracies.append({
        "Round": "Initial model", 
        "Pruning percentage": 0, 
        "Test Accuracy (with training)": dense_acc
    })
    
    # IMPORTANT : On retourne theta_k au lieu de theta_0 !
    return df_accuracies, model, get_weights(model), theta_k, train_loader, test_loader


# --- B. Fonction de Pruning Itératif utilisant le Rewinding ---
def iterative_pruning_CIFAR_Rewinding(total_prune_percent=90, rounds=8, epochs_per_round=10, rewind_epoch=2, lr=0.1, optimizer_type="sgd"):
    df_accuracies = []
    
    # Appel de la fonction Dense SPÉCIALE qui renvoie theta_k
    df_accuracies, model, thetaj, theta_k, train_loader, test_loader = dense_neural_network_CIFAR_Rewinding(
        df_accuracies, epochs=epochs_per_round, rewind_epoch=rewind_epoch, lr=lr, optimizer_type=optimizer_type
    )
    
    print(f"\nStep 4: Creating the Winning ticket (Reset to Epoch {rewind_epoch})")
    criterion = nn.CrossEntropyLoss()

    prune_percent = 1 - (1 - total_prune_percent/100)**(1/rounds)
    remaining_weights_percent = 1
    current_prune_percent = 0
    
    for pruning_round in range(rounds):
        print(f"\n--- Round {pruning_round + 1}/{rounds} ---")
        
        current_prune_percent += remaining_weights_percent * prune_percent
        remaining_weights_percent = 1 - current_prune_percent
        
        # 1. Pruning
        mask = prune_by_magnitude(model, current_prune_percent*100)
        
        # 2. RESET : On utilise theta_k (le checkpoint) au lieu de l'initialisation
        model = create_winning_ticket(model, mask, theta_k)
        
        actual_sparsity = calculate_actual_prune_percent(model)
        print(f"Sparsity: {actual_sparsity:.2f}%")
        
        acc = evaluate_the_model(model, test_loader)
        
        # 3. Retraining
        if optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # On utilise le masque pour figer les zéros
        training_the_model(model, train_loader, optimizer, criterion, epochs_per_round, mask=mask)
        
        acc_post_training = evaluate_the_model(model, test_loader)
        print(f"Accuracy after retraining: {acc_post_training:.2f}%")
        
        df_accuracies.append({
            "Round": f"Round {pruning_round + 1}", 
            "Pruning percentage": actual_sparsity, 
            "Test Accuracy (no retraining)": acc, 
            "Test Accuracy (with training)": acc_post_training
        })
            
    return df_accuracies, model



# ==============================================================================
#  EXTENSIONS STRONG LTH (Edge-Popup Algorithm)
#  Ces classes permettent de trouver un sous-réseau performant SANS toucher aux poids.
# ==============================================================================

class GetSubnet(torch.autograd.Function):
    """
    Fonction "Magique" (Straight Through Estimator) pour le masque binaire.
    - Forward : Calcule un masque binaire (0 ou 1) en gardant les meilleurs scores.
    - Backward : Laisse passer le gradient tel quel vers les scores (comme si c'était continu).
    """
    @staticmethod
    def forward(ctx, scores, k):
        # k est la fraction de poids à GARDER (ex: 0.1 pour 10%)
        # On clone pour ne pas modifier l'original en place
        out = scores.clone()
        
        # On trouve le seuil (k-ème percentile)
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())
        
        # Création du masque binaire : 0 pour les faibles scores, 1 pour les forts
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1
        
        return out

    @staticmethod
    def backward(ctx, g):
        # On renvoie le gradient 'g' pour les scores, et None pour 'k' (qui est constant)
        return g, None

class SupermaskConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 1. On crée les "Scores" (ce sont eux qu'on va entraîner !)
        # Ils ont la même forme que les poids.
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        
        # 2. IMPORTANT : On GÈLE les poids (ils ne bougeront plus jamais)
        self.weight.requires_grad = False 
        
        # Par défaut, on garde tout (sera écrasé par l'appel global)
        self.sparsity_ratio = 1.0 

    def set_prune_rate(self, prune_rate):
        """Définit le % de poids à garder (k)"""
        self.sparsity_ratio = 1.0 - (prune_rate / 100.0)

    def forward(self, x):
        # 3. On calcule le masque à la volée grâce aux scores
        subnet = GetSubnet.apply(self.scores.abs(), self.sparsity_ratio)
        
        # 4. On utilise les poids effectifs (Poids figés * Masque binaire)
        w = self.weight * subnet
        
        # 5. Convolution standard
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

class SupermaskLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Idem que pour Conv2d
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        
        self.weight.requires_grad = False # Poids gelés
        self.sparsity_ratio = 1.0

    def set_prune_rate(self, prune_rate):
        self.sparsity_ratio = 1.0 - (prune_rate / 100.0)

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), self.sparsity_ratio)
        w = self.weight * subnet
        return F.linear(x, w, self.bias)

# ==============================================================================
#  NOUVELLE ARCHITECTURE : Conv2 pour Strong LTH
#  C'est la copie exacte de votre Conv2, mais avec les couches Supermask.
# ==============================================================================

class Conv2_Strong(nn.Module):
    def __init__(self, output_size=10):
        super(Conv2_Strong, self).__init__()
        # Remplacement des nn.Conv2d par SupermaskConv2d
        self.conv1 = SupermaskConv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = SupermaskConv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Remplacement des nn.Linear par SupermaskLinear
        self.fc1 = SupermaskLinear(64 * 16 * 16, 256)
        self.fc2 = SupermaskLinear(256, 256)
        self.fc3 = SupermaskLinear(256, output_size)

    def set_global_pruning_rate(self, prune_rate):
        """Applique le taux de pruning à toutes les couches du réseau"""
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

# ==============================================================================
#  FONCTION D'EXPÉRIENCE : Strong LTH sur CIFAR-10
# ==============================================================================

def run_strong_lth_experiment(target_prune_percent=50, epochs=20, lr=0.1):
    """
    Lance l'entraînement 'Edge-Popup' pour trouver un sous-réseau performant
    sans modifier les poids initiaux.
    """
    print(f"\n🚀 Lancement Strong LTH (Edge-Popup) | Cible Sparsité: {target_prune_percent}%")
    
    # 1. Initialisation du modèle 'Strong'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Conv2_Strong(output_size=10).to(device)
    
    # 2. On règle la sparsité voulue (ex: 50% de poids à supprimer)
    model.set_global_pruning_rate(target_prune_percent)
    
    # 3. Chargement des données
    train_loader, test_loader = load_cifar(batch_size=60)
    
    # 4. Optimiseur : On n'entraîne QUE les scores (self.scores), pas les poids !
    # Les poids ont déjà requires_grad=False, mais on filtre par sécurité.
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], 
        lr=lr, 
        momentum=0.9, 
        weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    # 5. Entraînement des SCORES (le Masque)
    # On réutilise votre fonction 'training_the_model' existante !
    # Pas besoin de passer 'mask' car le masquage est interne aux couches Supermask.
    print("   -> Training mask scores...")
    training_the_model(model, train_loader, optimizer, criterion, num_epochs=epochs)
    
    # 6. Évaluation
    final_acc = evaluate_the_model(model, test_loader)
    print(f"   ✅ Résultat Strong LTH : {final_acc:.2f}% d'accuracy avec {target_prune_percent}% de poids supprimés (et poids figés!).")
    
    return final_acc