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

def calculate_actual_prune_percent(model):
    total_weights = sum(p.numel() for p in model.parameters() if p.dim() > 1)
    zero_weights = sum((p == 0).sum().item() for p in model.parameters() if p.dim() > 1)
    return 100 * zero_weights / total_weights

def count_zeros(model):
    return sum((p == 0).sum().item() for p in model.parameters() if p.dim() > 1)


########################### Lottery Ticket Algorithm functions #################################################################
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













