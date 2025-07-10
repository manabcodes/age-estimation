#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
import logging
import time
import random
import numpy as np
import sys

import sys
import os
    
# First make sure the VOLO directory is in the Python path
volo_dir = os.path.join(os.getcwd(), 'volo')
if os.path.exists(volo_dir) and volo_dir not in sys.path:
    sys.path.append(volo_dir)
    
# Now try to import from the volo directory
try:
    from volo.models import volo_d1
    from volo import models
    from volo.utils import load_pretrained_weights
except ImportError:
    # If that fails, try direct import (assuming we're in the volo directory)
    try:
        import models
        from models import volo_d1
        from utils import load_pretrained_weights
    except ImportError:
        raise ImportError(
            "Could not import VOLO modules. Please make sure you're either:\n"
            "1. Running from the VOLO directory, or\n"
            "2. Have the VOLO directory in your current working directory\n"
            "Current directory: " + os.getcwd()
        )


# In[3]:


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_logger = logging.getLogger('age_classification')

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


# In[5]:


class MeanVarianceLoss(nn.Module):
    def __init__(self, num_classes=121, lambda_mean=0.2, lambda_var=0.05):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_mean = lambda_mean
        self.lambda_var = lambda_var
        self.register_buffer("ages", torch.arange(num_classes).float())  # stays on model's device

    def forward(self, logits, target):
        # logits: (N, num_classes)
        # target: (N,) with integer age labels

        # ensure ages is on the same device as logits
        ages = self.ages.to(logits.device)

        prob = F.softmax(logits, dim=1)  # (N, C)

        # compute predicted mean: ∑ j * p_ij
        mean = torch.sum(prob * ages.unsqueeze(0), dim=1)  # (N,)

        # compute predicted variance: ∑ p_ij * (j - mean)^2
        var = torch.sum(prob * (ages.unsqueeze(0) - mean.unsqueeze(1))**2, dim=1)  # (N,)

        # mean loss (squared difference to true label)
        loss_mean = 0.5 * (mean - target.float())**2

        # total loss: weighted sum
        total_loss = self.lambda_mean * loss_mean.mean() + self.lambda_var * var.mean()
        return total_loss


# In[6]:


def create_model(num_classes=122, checkpoint_path=None, model_name='volo_d1'):
    """Create and load pre-trained VOLO model"""
    # Create the model
    model = getattr(models, model_name)(img_size=224)
    
    # Load pre-trained weights
    if checkpoint_path:
        _logger.info(f"Loading pre-trained weights from {checkpoint_path}")
        load_pretrained_weights(
            model=model,
            checkpoint_path=checkpoint_path,
            use_ema=False,
            strict=False,
            num_classes=num_classes
        )
    
    # Modify the classifier head for age classification
    model.head = nn.Linear(model.head.in_features, num_classes)
    if hasattr(model, 'aux_head'):
        model.aux_head = nn.Linear(model.aux_head.in_features, num_classes)
    
    return model


# In[7]:


def apply_unfreezing_strategy(model, epoch, config):
    """Apply the appropriate unfreezing strategy based on the current epoch"""
    # Define unfreezing schedule - which epochs to change strategy
    unfreezing_milestones = {
        0: 'head_only',          # Start with only head unfrozen
        20: 'post_network',       # After 3 epochs, unfreeze post_network
        40: 'partial_transformer', # After 6 epochs, unfreeze partial transformer
        55: 'full_transformer',    # After 9 epochs, unfreeze full transformer
        70: 'include_outlooker'   # After 12 epochs, unfreeze everything
    }
    
    # Get the base model
    base_model = model
    
    # Check if we need to update unfreezing strategy
    if epoch in unfreezing_milestones:
        strategy = unfreezing_milestones[epoch]
        print(f"Updating unfreezing strategy to: {strategy}")
        
        # Freeze all parameters first
        for param in base_model.parameters():
            param.requires_grad = False
            
        # Apply the appropriate unfreezing strategy
        if strategy == 'head_only':
            # Just the final layer
            for param in base_model.head.parameters():
                param.requires_grad = True
            if hasattr(base_model, 'aux_head'):
                for param in base_model.aux_head.parameters():
                    param.requires_grad = True
                
        elif strategy == 'post_network':
            # Unfreeze head and post_network (class attention blocks)
            for param in base_model.head.parameters():
                param.requires_grad = True
            if hasattr(base_model, 'aux_head'):
                for param in base_model.aux_head.parameters():
                    param.requires_grad = True
            if base_model.post_network is not None:
                for block in base_model.post_network:
                    for param in block.parameters():
                        param.requires_grad = True
                        
        elif strategy == 'partial_transformer':
            # Unfreeze head, post_network, and last transformer blocks
            for param in base_model.head.parameters():
                param.requires_grad = True
            if hasattr(base_model, 'aux_head'):
                for param in base_model.aux_head.parameters():
                    param.requires_grad = True
            if base_model.post_network is not None:
                for block in base_model.post_network:
                    for param in block.parameters():
                        param.requires_grad = True
            # Unfreeze the last transformer blocks (indices depend on model architecture)
            # Usually the last blocks are transformer blocks
            transformer_idx = [2, 3]  # This may vary based on model architecture
            for i in transformer_idx:
                if i < len(base_model.network):
                    for block in base_model.network[i]:
                        for param in block.parameters():
                            param.requires_grad = True
                    
        elif strategy == 'full_transformer':
            # Unfreeze all transformer blocks (but not the early outlooker blocks)
            for param in base_model.head.parameters():
                param.requires_grad = True
            if hasattr(base_model, 'aux_head'):
                for param in base_model.aux_head.parameters():
                    param.requires_grad = True
            if base_model.post_network is not None:
                for block in base_model.post_network:
                    for param in block.parameters():
                        param.requires_grad = True
            # Unfreeze all transformer blocks
            transformer_idx = [2, 3, 4]  # This may vary based on model architecture
            for i in transformer_idx:
                if i < len(base_model.network):
                    for block in base_model.network[i]:
                        for param in block.parameters():
                            param.requires_grad = True

        elif strategy == 'include_outlooker':
            # Unfreeze everything
            for param in base_model.parameters():
                param.requires_grad = True
        
        # Create a new optimizer with the updated parameters
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['lr'] / (2 ** (epoch // 3)),  # Reduce learning rate as we unfreeze more
            weight_decay=config['weight_decay']
        )
        
        # Create a new scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs'] - epoch,
            eta_min=1e-6
        )
        
        # Print trainable parameters info
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f'Trainable parameters: {trainable:,} ({trainable/total:.2%} of total {total:,})')
        
        return optimizer, scheduler
    
    return None, None


# In[8]:


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, checkpoint_dir, save_freq=50):
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    total = 0
    start_time = time.time()
    
    print(f"Epoch {epoch+1}/{config['num_epochs']}")
    print("----------")
    
    # Create a progress bar using tqdm
    pbar = tqdm(total=len(dataloader), desc="Train", file=sys.stdout)
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Handle different return formats of VOLO
        if isinstance(outputs, tuple):
            # Use only the main classification output (first element)
            outputs_for_loss = outputs[0]
        else:
            outputs_for_loss = outputs
            
        loss = criterion(outputs_for_loss, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate loss and MAE
        running_loss += loss.item() * inputs.size(0)
        
        # For MAE calculation (age prediction as the index with highest probability)
        if isinstance(outputs, tuple):
            _, predicted = outputs[0].max(1)
        else:
            _, predicted = outputs.max(1)
        
        mae = F.l1_loss(predicted.float(), targets.float())
        running_mae += mae.item() * inputs.size(0)
        
        total += inputs.size(0)
        
        # Calculate current metrics
        current_avg_loss = running_loss / total
        current_avg_mae = running_mae / total
        
        # Update progress bar
        progress_percent = (batch_idx + 1) / len(dataloader) * 100
        pbar.set_description(
            f"Train: {progress_percent:3.0f}%|{'█' * int(progress_percent/10)}| {batch_idx+1}/{len(dataloader)} "
            f"[{time.time()-start_time:.2f}/{(time.time()-start_time)/(batch_idx+1)*len(dataloader):.2f}] "
            f"loss={current_avg_loss:.4f}, mae={current_avg_mae:.2f}, batch={batch_idx}"
        )
        pbar.update(1)
        
        # Save checkpoint at specified frequency
        if (batch_idx + 1) % save_freq == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch+1}_batch{batch_idx+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'batch': batch_idx + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_avg_loss,
                'mae': current_avg_mae
            }, checkpoint_path)
            print(f"\nSaved checkpoint at epoch {epoch+1}, batch {batch_idx+1} to {checkpoint_path}")
    
    # Close the progress bar
    pbar.close()
    
    # Calculate final metrics for the epoch
    epoch_loss = running_loss / total
    epoch_mae = running_mae / total
    
    print(f"train Loss: {epoch_loss:.4f} MAE: {epoch_mae:.4f}")
    
    return epoch_loss, epoch_mae


# In[9]:


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    total = 0
    start_time = time.time()
    
    # Create a progress bar using tqdm
    pbar = tqdm(total=len(dataloader), desc="Val", file=sys.stdout)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Handle different return formats of VOLO
            if isinstance(outputs, tuple):
                outputs_for_loss = outputs[0]
            else:
                outputs_for_loss = outputs
                
            loss = criterion(outputs_for_loss, targets)
            
            # Calculate loss and MAE
            running_loss += loss.item() * inputs.size(0)
            
            # NEW CODE: Use expected value for age prediction instead of argmax
            if isinstance(outputs, tuple):
                outputs_for_calc = outputs[0]
            else:
                outputs_for_calc = outputs
                
            # Convert logits to probabilities
            probs = F.softmax(outputs_for_calc, dim=1)
            
            # Create age range tensor (0 to num_classes-1)
            ages = torch.arange(122).to(device).float()  # For 0-121 age range
            
            # Calculate expected age value (weighted average)
            predicted = torch.sum(probs * ages.unsqueeze(0), dim=1)
            
            # Calculate MAE using expected age
            mae = F.l1_loss(predicted, targets.float())
            running_mae += mae.item() * inputs.size(0)
            
            total += inputs.size(0)
            
            # Calculate current metrics
            current_avg_loss = running_loss / total
            current_avg_mae = running_mae / total
            
            # Update progress bar
            progress_percent = (batch_idx + 1) / len(dataloader) * 100
            elapsed_time = time.time() - start_time
            estimated_total = elapsed_time / (batch_idx + 1) * len(dataloader)
            remaining_time = estimated_total - elapsed_time
            
            # Update the progress bar with current metrics
            pbar.set_description(
                f"Val: {progress_percent:3.0f}%|{'█' * int(progress_percent/10)}| {batch_idx+1}/{len(dataloader)} "
                f"[{elapsed_time:.2f}/{estimated_total:.2f}] "
                f"loss={current_avg_loss:.4f}, mae={current_avg_mae:.2f}, batch={batch_idx}"
            )
            pbar.update(1)
    
    # Close the progress bar
    pbar.close()
    
    # Calculate final metrics for validation
    val_loss = running_loss / total
    val_mae = running_mae / total
    
    print(f"val Loss: {val_loss:.4f} MAE: {val_mae:.4f}")
    
    return val_loss, val_mae


# In[10]:


# Configuration
config = {
    'model': 'volo_d1',
    'checkpoint': '/home/meem/backup/d1_224_84.2.pth.tar',
    'train_csv': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/train_annotations.csv',
    'val_csv': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/val_annotations.csv',
    'train_dir': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/train',
    'val_dir': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/val',
    'output': './output',
    'checkpoint_dir': './checkpoints1',
    'batch_size': 16,
    'num_epochs': 80,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'save_freq': 50  # Save checkpoint every 50 batches
}

# Create output directories
os.makedirs(config['output'], exist_ok=True)
os.makedirs(config['checkpoint_dir'], exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    
# 2. Set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 3. Delete any existing checkpoints to avoid loading old state
import shutil
if os.path.exists('checkpoints'):
    shutil.rmtree('checkpoints')
os.makedirs('checkpoints', exist_ok=True)

_logger.info(f"Using device: {device}")

# Data transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

# Create model
model = create_model(
    num_classes=122,  # 0-121 age range
    checkpoint_path=config['checkpoint'],
    model_name=config['model']
)
model = model.to(device)

# Loss function
# criterion = nn.CrossEntropyLoss()
criterion = MeanVarianceLoss(num_classes=122, lambda_mean=0.2, lambda_var=0.05)

# Apply initial unfreezing strategy (head only)
optimizer, scheduler = apply_unfreezing_strategy(model, 0, config)

# Initial lr scheduler if not created by unfreezing strategy
if scheduler is None:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])

# Training loop
best_val_loss = float('inf')

for epoch in range(config['num_epochs']):
    # Apply progressive unfreezing if needed
    new_optimizer, new_scheduler = apply_unfreezing_strategy(model, epoch, config)
    if new_optimizer is not None:
        optimizer = new_optimizer
    if new_scheduler is not None:
        scheduler = new_scheduler
    
    # Train
    epoch_checkpoint_dir = os.path.join(config['checkpoint_dir'], f"epoch{epoch+1}")
    os.makedirs(epoch_checkpoint_dir, exist_ok=True)
    
    train_loss, train_mae = train_one_epoch(
        model, train_loader, criterion, optimizer, device, 
        epoch, epoch_checkpoint_dir, config['save_freq']
    )
    
    # Validate
    val_loss, val_mae = validate(model, val_loader, criterion, device)
    
    # Update learning rate
    scheduler.step()
    
    # Save checkpoint if it's the best model so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_path = os.path.join(config['output'], f"{config['model']}_best.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_mae': val_mae,
            'unfreezing_strategy': next((strategy for ep, strategy in {
                0: 'head_only', 3: 'post_network', 6: 'partial_transformer', 
                9: 'full_transformer', 12: 'include_outlooker'
            }.items() if ep <= epoch), 'include_outlooker')
        }, best_model_path)
        print(f"Saved best model with loss {val_loss:.4f}")
    
    # Save checkpoint at the end of each epoch
    epoch_checkpoint_path = os.path.join(config['output'], f"{config['model']}_epoch{epoch+1}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_mae': val_mae,
        'unfreezing_strategy': next((strategy for ep, strategy in {
            0: 'head_only', 3: 'post_network', 6: 'partial_transformer', 
            9: 'full_transformer', 12: 'include_outlooker'
        }.items() if ep <= epoch), 'include_outlooker')
    }, epoch_checkpoint_path)
    print(f"Saved checkpoint at end of epoch {epoch+1}")

# Save final model
final_model_path = os.path.join(config['output'], f"{config['model']}_meanvar_gradual_final.pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'unfreezing_strategy': 'include_outlooker',  # Final model has everything unfrozen
    'config': config,
}, final_model_path)
print(f"Saved final model to {final_model_path}")

