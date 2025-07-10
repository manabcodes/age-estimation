#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm.notebook import tqdm
import logging
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
import clip
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_logger = logging.getLogger('poe_farl_age_estimation')

class AgeClassificationDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        age = self.annotations.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
            
        return image, age

class POEFaRLModel(nn.Module):
    """FaRL backbone with POE (Probabilistic Ordinal Embeddings) for age estimation"""
    
    def __init__(self, farl_model_path=None, embedding_dim=512, 
                 dropout_rate=0.1, freeze_backbone=True, max_t=50):
        super().__init__()
        
        # Load CLIP ViT-B/16 architecture
        self.clip_model, self.preprocess = clip.load("ViT-B/16", device="cpu")
        
        # Load FaRL weights if provided
        if farl_model_path and os.path.exists(farl_model_path):
            _logger.info(f"Loading FaRL weights from {farl_model_path}")
            farl_state = torch.load(farl_model_path, map_location="cpu")
            self.clip_model.load_state_dict(farl_state["state_dict"], strict=False)
        else:
            _logger.warning("No FaRL weights provided, using CLIP weights only")
        
        # Extract the visual encoder (backbone)
        self.backbone = self.clip_model.visual
        
        # Get the actual feature dimension from the backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_features = self.backbone(dummy_input)
            backbone_dim = dummy_features.shape[-1]
            
        _logger.info(f"Detected backbone feature dimension: {backbone_dim}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            _logger.info("FaRL backbone frozen")
        
        # POE Components
        # Mean branch (μ(x))
        self.mu_head = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Variance branch (Σ(x)) - diagonal covariance
        self.sigma_head = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim, eps=0.001, affine=False),
            # Use exp to ensure positive variance
        )
        
        # Final regression head
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, 1)
        )
        
        self.max_t = max_t
        self.freeze_backbone = freeze_backbone
        self.embedding_dim = embedding_dim
        
    def forward(self, x, use_sto=True, training=True):
        # Extract features using FaRL backbone
        if self.freeze_backbone:
            with torch.no_grad():
                backbone_features = self.backbone(x)
        else:
            backbone_features = self.backbone(x)
        
        # Get mean and log variance
        mu = self.mu_head(backbone_features)  # μ(x)
        log_var = self.sigma_head(backbone_features)  # log(σ²(x))
        
        if use_sto and training:
            # Reparameterization trick: z = μ + σ * ε, where ε ~ N(0,I)
            # Sample multiple times for Monte Carlo approximation
            batch_size = mu.size(0)
            
            # Expand for multiple samples
            mu_expanded = mu.unsqueeze(0).expand(self.max_t, -1, -1)  # [max_t, batch, dim]
            log_var_expanded = log_var.unsqueeze(0).expand(self.max_t, -1, -1)
            
            # Sample noise
            eps = torch.randn_like(mu_expanded)
            
            # Reparameterization: z = μ + σ * ε
            z_samples = mu_expanded + torch.exp(0.5 * log_var_expanded) * eps
            
            # Apply regression head to each sample
            z_samples_flat = z_samples.view(-1, self.embedding_dim)  # [max_t * batch, dim]
            age_pred_flat = self.regression_head(z_samples_flat)  # [max_t * batch, 1]
            age_pred = age_pred_flat.view(self.max_t, batch_size)  # [max_t, batch]
            
            return age_pred, mu, log_var
        else:
            # Deterministic forward pass (inference or no stochastic sampling)
            age_pred = self.regression_head(mu)
            return age_pred.squeeze(), mu, log_var
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
        _logger.info("FaRL backbone unfrozen")

class POELoss(nn.Module):
    """POE Loss combining regression loss, ordinal constraint, and VIB regularization"""
    
    def __init__(self, alpha_coeff=1e-4, beta_coeff=1e-4, margin=5.0, 
                 distance='JDistance', main_loss_type='reg'):
        super().__init__()
        self.alpha_coeff = alpha_coeff  # VIB regularization coefficient
        self.beta_coeff = beta_coeff    # Ordinal constraint coefficient
        self.margin = margin
        self.main_loss_type = main_loss_type
        self.distance = distance
        
    def jdistance(self, mu1, sigma1, mu2, sigma2):
        """Jensen-Shannon Distance (symmetrized KL divergence)"""
        # Forward KL: KL(P||Q)
        kl_forward = -0.5 * torch.sum(
            torch.log(sigma1) - torch.log(sigma2) - 
            sigma1/sigma2 - (mu1 - mu2)**2 / sigma2 + 1, dim=1
        )
        
        # Reverse KL: KL(Q||P) 
        kl_reverse = -0.5 * torch.sum(
            torch.log(sigma2) - torch.log(sigma1) - 
            sigma2/sigma1 - (mu1 - mu2)**2 / sigma1 + 1, dim=1
        )
        
        return kl_forward + kl_reverse
    
    def wasserstein_distance(self, mu1, sigma1, mu2, sigma2):
        """2-Wasserstein distance between two Gaussian distributions"""
        mu_dist = torch.sum((mu1 - mu2)**2, dim=1)
        sigma_dist = torch.sum((torch.sqrt(sigma1) - torch.sqrt(sigma2))**2, dim=1)
        return torch.sqrt(mu_dist + sigma_dist)
    
    def compute_ordinal_loss(self, mu, log_var, targets):
        """Compute ordinal distribution constraint loss"""
        batch_size = mu.size(0)
        
        if batch_size < 3:
            return torch.tensor(0.0, device=mu.device)
        
        # Convert log variance to variance
        sigma = torch.exp(log_var)
        
        # Create target distance matrix
        target_distances = torch.abs(targets.unsqueeze(1) - targets.unsqueeze(0))
        
        total_loss = 0.0
        valid_triplets = 0
        
        # Sample triplets for ordinal constraint
        for i in range(batch_size):
            # Find indices where |y_i - y_j| < |y_i - y_k|
            target_diff_i = torch.abs(targets - targets[i])
            
            for j in range(batch_size):
                if i == j:
                    continue
                    
                for k in range(batch_size):
                    if i == k or j == k:
                        continue
                    
                    # Check ordinal constraint: |y_i - y_j| < |y_i - y_k|
                    if target_diff_i[j] < target_diff_i[k]:
                        # Compute distances in embedding space
                        if self.distance == 'JDistance':
                            dist_ij = self.jdistance(mu[i:i+1], sigma[i:i+1], 
                                                   mu[j:j+1], sigma[j:j+1])
                            dist_ik = self.jdistance(mu[i:i+1], sigma[i:i+1], 
                                                   mu[k:k+1], sigma[k:k+1])
                        else:  # Wasserstein
                            dist_ij = self.wasserstein_distance(mu[i:i+1], sigma[i:i+1], 
                                                              mu[j:j+1], sigma[j:j+1])
                            dist_ik = self.wasserstein_distance(mu[i:i+1], sigma[i:i+1], 
                                                              mu[k:k+1], sigma[k:k+1])
                        
                        # Hinge loss: max(0, d(z_i, z_j) + margin - d(z_i, z_k))
                        loss_triplet = torch.max(torch.tensor(0.0, device=mu.device), 
                                               dist_ij + self.margin - dist_ik)
                        total_loss += loss_triplet
                        valid_triplets += 1
        
        if valid_triplets > 0:
            return total_loss / valid_triplets
        else:
            return torch.tensor(0.0, device=mu.device)
    
    def compute_vib_loss(self, mu, log_var):
        """Compute Variational Information Bottleneck regularization"""
        # KL divergence between N(μ, σ²) and N(0, I)
        kl_loss = 0.5 * torch.sum(mu**2 + torch.exp(log_var) - log_var - 1, dim=1)
        return torch.mean(kl_loss)
    
    def forward(self, age_pred, mu, log_var, targets, use_sto=True):
        # Main regression loss
        if use_sto and len(age_pred.shape) == 2:  # [max_t, batch]
            # Monte Carlo approximation
            regression_loss = F.l1_loss(age_pred, targets.unsqueeze(0).expand_as(age_pred))
        else:
            # Deterministic case
            regression_loss = F.l1_loss(age_pred, targets.float())
        
        # Ordinal constraint loss
        ordinal_loss = self.compute_ordinal_loss(mu, log_var, targets)
        
        # VIB regularization loss
        vib_loss = self.compute_vib_loss(mu, log_var)
        
        # Total loss
        total_loss = (regression_loss + 
                     self.beta_coeff * ordinal_loss + 
                     self.alpha_coeff * vib_loss)
        
        return total_loss, regression_loss, ordinal_loss, vib_loss

def apply_poe_unfreezing_strategy(model, epoch, config):
    """Apply progressive unfreezing strategy for POE-FaRL"""
    unfreezing_milestones = {
        0: 'head_only',
        10: 'backbone_partial', 
        15: 'backbone_full',
    }
    
    if epoch in unfreezing_milestones:
        strategy = unfreezing_milestones[epoch]
        print(f"Updating unfreezing strategy to: {strategy}")
        
        # FREEZE EVERYTHING FIRST
        for param in model.parameters():
            param.requires_grad = False
            
        if strategy == 'head_only':
            # Only unfreeze POE heads and regression head
            for param in model.mu_head.parameters():
                param.requires_grad = True
            for param in model.sigma_head.parameters():
                param.requires_grad = True
            for param in model.regression_head.parameters():
                param.requires_grad = True
                
        elif strategy == 'backbone_partial':
            # POE heads + last few transformer blocks
            for param in model.mu_head.parameters():
                param.requires_grad = True
            for param in model.sigma_head.parameters():
                param.requires_grad = True
            for param in model.regression_head.parameters():
                param.requires_grad = True
            # Add backbone partial unfreezing here if needed
                
        elif strategy == 'backbone_full':
            # Everything
            for param in model.parameters():
                param.requires_grad = True

        # Adjust learning rate based on unfreezing stage
        lr_divisors = {
            'head_only': 1,
            'backbone_partial': 2,
            'backbone_full': 4
        }
        
        lr = config['lr'] / lr_divisors[strategy]
        
        # Create new optimizer with adjusted learning rate
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=config['weight_decay']
        )
        
        # Create new scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs'] - epoch,
            eta_min=1e-6
        )
        
        # Print trainable parameters info
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f'Trainable parameters: {trainable:,} ({trainable/total:.2%} of total {total:,})')
        print(f'Learning rate: {lr:.6f}')
        
        return optimizer, scheduler
    
    return None, None

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, config):
    model.train()
    running_loss = 0.0
    running_reg_loss = 0.0
    running_ord_loss = 0.0
    running_vib_loss = 0.0
    running_mae = 0.0
    total = 0
    
    print(f"Epoch {epoch+1}/{config['num_epochs']}")
    print("----------")
    
    pbar = tqdm(total=len(dataloader), desc="Train", file=sys.stdout)
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with stochastic sampling
        age_pred, mu, log_var = model(inputs, use_sto=True, training=True)
        
        # Calculate POE loss
        total_loss, reg_loss, ord_loss, vib_loss = criterion(
            age_pred, mu, log_var, targets, use_sto=True
        )
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Calculate metrics
        if len(age_pred.shape) == 2:  # Stochastic case
            pred_mean = torch.mean(age_pred, dim=0)  # Average over samples
        else:
            pred_mean = age_pred
            
        mae = F.l1_loss(pred_mean, targets.float())
        
        running_loss += total_loss.item() * inputs.size(0)
        running_reg_loss += reg_loss.item() * inputs.size(0)
        running_ord_loss += ord_loss.item() * inputs.size(0)
        running_vib_loss += vib_loss.item() * inputs.size(0)
        running_mae += mae.item() * inputs.size(0)
        total += inputs.size(0)
        
        # Update progress bar
        current_avg_loss = running_loss / total
        current_avg_mae = running_mae / total
        
        pbar.set_description(
            f"Train: loss={current_avg_loss:.4f}, mae={current_avg_mae:.2f}"
        )
        pbar.update(1)
    
    pbar.close()
    
    epoch_loss = running_loss / total
    epoch_reg_loss = running_reg_loss / total
    epoch_ord_loss = running_ord_loss / total
    epoch_vib_loss = running_vib_loss / total
    epoch_mae = running_mae / total
    
    print(f"Train Loss: {epoch_loss:.4f} (Reg: {epoch_reg_loss:.4f}, "
          f"Ord: {epoch_ord_loss:.4f}, VIB: {epoch_vib_loss:.4f}) MAE: {epoch_mae:.4f}")
    
    return epoch_loss, epoch_mae

def validate(model, dataloader, criterion, device, config):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    total = 0
    
    pbar = tqdm(total=len(dataloader), desc="Val", file=sys.stdout)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass without stochastic sampling for validation
            age_pred, mu, log_var = model(inputs, use_sto=False, training=False)
            
            # Calculate loss
            total_loss, _, _, _ = criterion(age_pred, mu, log_var, targets, use_sto=False)
            
            # Calculate metrics
            mae = F.l1_loss(age_pred, targets.float())
            
            running_loss += total_loss.item() * inputs.size(0)
            running_mae += mae.item() * inputs.size(0)
            total += inputs.size(0)
            
            current_avg_loss = running_loss / total
            current_avg_mae = running_mae / total
            
            pbar.set_description(
                f"Val: loss={current_avg_loss:.4f}, mae={current_avg_mae:.2f}"
            )
            pbar.update(1)
    
    pbar.close()
    
    val_loss = running_loss / total
    val_mae = running_mae / total
    
    print(f"Val Loss: {val_loss:.4f} MAE: {val_mae:.4f}")
    
    return val_loss, val_mae

def main():
    # Configuration
    config = {
        # Data paths - UPDATE THESE TO YOUR PATHS
        'train_csv': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/train_annotations.csv',
        'val_csv': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/val_annotations.csv',
        'train_dir': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/train',
        'val_dir': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/val',
        
        # FaRL model path
        'farl_model_path': './FaRL-Base-Patch16-LAIONFace20M-ep64.pth',
        
        # Output paths
        'output': './output_poe_farl',
        'checkpoint_dir': './checkpoints_poe_farl',
        
        # Training parameters
        'batch_size': 16,
        'num_epochs': 25,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        
        # POE model parameters
        'embedding_dim': 512,
        'dropout_rate': 0.1,
        'freeze_backbone': True,
        'max_t': 50,  # Number of stochastic samples
        
        # POE loss parameters
        'alpha_coeff': 1e-4,   # VIB regularization coefficient
        'beta_coeff': 1e-4,    # Ordinal constraint coefficient
        'margin': 5.0,         # Margin for ordinal constraint
        'distance': 'JDistance',  # 'JDistance' or 'Wasserstein'
    }
    
    # Create output directories
    os.makedirs(config['output'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Set device and seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _logger.info(f"Using device: {device}")

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Set seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Data transforms - use CLIP preprocessing for consistency
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                           std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                           std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    
    # Create datasets and dataloaders
    train_dataset = AgeClassificationDataset(
        csv_file=config['train_csv'],
        img_dir=config['train_dir'],
        transform=train_transform
    )
    
    val_dataset = AgeClassificationDataset(
        csv_file=config['val_csv'],
        img_dir=config['val_dir'],
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create POE-FaRL model
    model = POEFaRLModel(
        farl_model_path=config['farl_model_path'],
        embedding_dim=config['embedding_dim'],
        dropout_rate=config['dropout_rate'],
        freeze_backbone=config['freeze_backbone'],
        max_t=config['max_t']
    )
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _logger.info(f"Total parameters: {total_params:,}")
    _logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    
    # Create POE loss function
    criterion = POELoss(
        alpha_coeff=config['alpha_coeff'],
        beta_coeff=config['beta_coeff'],
        margin=config['margin'],
        distance=config['distance']
    )
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['num_epochs'],
        eta_min=1e-6
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        # Apply progressive unfreezing if needed
        new_optimizer, new_scheduler = apply_poe_unfreezing_strategy(model, epoch, config)
        if new_optimizer is not None:
            optimizer = new_optimizer
        if new_scheduler is not None:
            scheduler = new_scheduler
        
        # Train
        train_loss, train_mae = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        
        # Validate
        val_loss, val_mae = validate(model, val_loader, criterion, device, config)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(config['output'], "poe_farl_best.pth")
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'config': config,
            }, best_model_path)
            print(f"Saved best model with loss {val_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(config['checkpoint_dir'], f"poe_farl_epoch{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_mae': val_mae,
            'config': config,
        }, checkpoint_path)
    
    # Save final model
    final_model_path = os.path.join(config['output'], "poe_farl_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")

if __name__ == "__main__":
    main()

