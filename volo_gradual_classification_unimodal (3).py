#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

import matplotlib.pyplot as plt

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


# In[2]:


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


# In[3]:


class UnimodalConcentratedLoss(nn.Module):
    def __init__(self, lambda_value=1000):  # Reduced lambda from 1000 to 100
        """
        Implementation of the Unimodal-Concentrated Loss from the paper
        
        Args:
            lambda_value: Hyperparameter to weight the unimodal loss
        """
        super(UnimodalConcentratedLoss, self).__init__()
        self.lambda_value = lambda_value
        
    def forward(self, outputs, targets):
        """
        Compute the unimodal-concentrated loss
        
        Args:
            outputs: Model outputs (logits)
            targets: Ground truth labels
            
        Returns:
            Tuple of (total_loss, concentrated_loss, weighted_unimodal_loss)
        """
        batch_size = outputs.size(0)
        num_classes = outputs.size(1)
        device = outputs.device
        
        # Convert targets to one-hot encoding for calculating loss
        targets = targets.view(-1)
        targets_one_hot = torch.zeros(batch_size, num_classes, device=device)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Apply softmax to get probability distribution
        probs = F.softmax(outputs, dim=1)
        
        # Calculate predicted values (expectation)
        labels = torch.arange(0, num_classes, device=device).float()
        pred_values = torch.sum(probs * labels.view(1, -1), dim=1)
        
        # Calculate variance of the distribution
        centered_labels = labels.view(1, -1) - pred_values.view(-1, 1)
        variance = torch.sum(probs * (centered_labels ** 2), dim=1)
        
        # Avoid division by zero
        variance = torch.clamp(variance, min=1e-6)
        
        # Concentrated loss: ln(v)/2 + (yhat - y)^2/(2v)
        estimation_error = (pred_values - targets.float()) ** 2
        concentrated_loss = 0.5 * torch.log(variance) + estimation_error / (2 * variance)
        concentrated_loss = torch.mean(concentrated_loss)
        
        # Add a CE component to help with initial training
        ce_loss = F.cross_entropy(outputs, targets)
        
        # Unimodal loss
        unimodal_loss = torch.zeros(1, device=device)
        for i in range(batch_size):
            true_label = targets[i].long()
            for j in range(num_classes - 1):
                # Calculate sign[j-yi] as described in the paper
                sign_value = -1 if j < true_label else 1
                
                # Calculate difference between adjacent probabilities
                prob_diff = probs[i, j] - probs[i, j+1]
                
                # Multiply by sign value according to the paper's formula
                # max(0, -(pi,j-pi,j+1)*sign[j-yi])
                penalty = -prob_diff * sign_value
                unimodal_loss += torch.max(torch.zeros(1, device=device), penalty)
        
        unimodal_loss = unimodal_loss / batch_size
        
        # Weight the unimodal loss
        weighted_unimodal_loss = self.lambda_value * unimodal_loss
        
        # Total loss: add CE loss with a weight to help with early training
        total_loss = concentrated_loss + weighted_unimodal_loss # + 0.1 * ce_loss
        
        # Return all components for monitoring
        return total_loss, concentrated_loss, weighted_unimodal_loss

# Advanced version with warmup period
class UnimodalConcentratedLossWithWarmup(nn.Module):
    def __init__(self, lambda_value=1000, warmup_epochs=2, ce_weight=1.0, lambda_warmup=10):
        """
        Implementation of the Unimodal-Concentrated Loss with a warmup period
        
        Args:
            lambda_value: Final hyperparameter value to weight the unimodal loss
            warmup_epochs: Number of epochs to gradually increase lambda
            ce_weight: Weight of the cross-entropy loss during warmup
            lambda_warmup: Initial lambda value during warmup
        """
        super(UnimodalConcentratedLossWithWarmup, self).__init__()
        self.lambda_value = lambda_value
        self.warmup_epochs = warmup_epochs
        self.ce_weight = ce_weight
        self.lambda_warmup = lambda_warmup
        self.current_epoch = 0
        
    def set_epoch(self, epoch):
        """Update the current epoch for lambda scheduling"""
        self.current_epoch = epoch
        
    def get_current_lambda(self):
        """Calculate lambda based on current epoch"""
        if self.current_epoch < self.warmup_epochs:
            # Linear interpolation from lambda_warmup to lambda_value
            return self.lambda_warmup + (self.lambda_value - self.lambda_warmup) * (self.current_epoch / self.warmup_epochs)
        else:
            return self.lambda_value
        
    def forward(self, outputs, targets):
        """
        Compute the unimodal-concentrated loss with warmup
        
        Args:
            outputs: Model outputs (logits)
            targets: Ground truth labels
            
        Returns:
            Tuple of (total_loss, concentrated_loss, weighted_unimodal_loss)
        """
        batch_size = outputs.size(0)
        num_classes = outputs.size(1)
        device = outputs.device
        
        # Get current lambda value based on epoch
        current_lambda = self.get_current_lambda()
        
        # Calculate cross-entropy loss for early training stability
        ce_loss = F.cross_entropy(outputs, targets)
        
        # Early in training, rely more on cross-entropy
        if self.current_epoch < self.warmup_epochs:
            ce_weight = self.ce_weight * (1 - self.current_epoch / self.warmup_epochs)
        else:
            ce_weight = 0.0
            
        # Apply softmax to get probability distribution
        probs = F.softmax(outputs, dim=1)
        
        # Calculate predicted values (expectation)
        labels = torch.arange(0, num_classes, device=device).float()
        pred_values = torch.sum(probs * labels.view(1, -1), dim=1)
        
        # Calculate variance of the distribution
        centered_labels = labels.view(1, -1) - pred_values.view(-1, 1)
        variance = torch.sum(probs * (centered_labels ** 2), dim=1)
        
        # Avoid division by zero
        variance = torch.clamp(variance, min=1e-6)
        
        # Concentrated loss: ln(v)/2 + (yhat - y)^2/(2v)
        estimation_error = (pred_values - targets.float()) ** 2
        concentrated_loss = 0.5 * torch.log(variance) + estimation_error / (2 * variance)
        concentrated_loss = torch.mean(concentrated_loss)
        
        # Unimodal loss
        unimodal_loss = torch.zeros(1, device=device)
        for i in range(batch_size):
            true_label = targets[i].long()
            for j in range(num_classes - 1):
                # Calculate sign[j-yi] as described in the paper
                sign_value = -1 if j < true_label else 1
                
                # Calculate difference between adjacent probabilities
                prob_diff = probs[i, j] - probs[i, j+1]
                
                # Multiply by sign value according to the paper's formula
                # max(0, -(pi,j-pi,j+1)*sign[j-yi])
                penalty = -prob_diff * sign_value
                unimodal_loss += torch.max(torch.zeros(1, device=device), penalty)
        
        unimodal_loss = unimodal_loss / batch_size
        
        # Weight the unimodal loss
        weighted_unimodal_loss = current_lambda * unimodal_loss
        
        # Total loss with cross-entropy component for stability
        total_loss = concentrated_loss + weighted_unimodal_loss + ce_weight * ce_loss
        
        # Return all components for monitoring
        return total_loss, concentrated_loss, weighted_unimodal_loss


# In[4]:


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


# In[5]:


def apply_unfreezing_strategy(model, epoch, config):
    """Apply the appropriate unfreezing strategy based on the current epoch"""
    # Define unfreezing schedule - which epochs to change strategy
    unfreezing_milestones = {
        0: 'head_only',          # Start with only head unfrozen
        8: 'post_network',       # After 3 epochs, unfreeze post_network
        16: 'partial_transformer', # After 6 epochs, unfreeze partial transformer
        24: 'full_transformer',    # After 9 epochs, unfreeze full transformer
        36: 'include_outlooker'   # After 12 epochs, unfreeze everything
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


# In[6]:


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, checkpoint_dir, save_freq=50, num_classes=122):
    """
    Train the model for one epoch with proper metric calculation
    
    Args:
        model: The neural network model
        dataloader: DataLoader for the training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on ('cuda' or 'cpu')
        epoch: Current epoch number
        checkpoint_dir: Directory to save checkpoints
        save_freq: Frequency to save checkpoints (in batches)
        num_classes: Number of age classes
        
    Returns:
        Dictionary containing all training metrics
    """
    model.train()
    
    # If criterion is None, use Unimodal-Concentrated Loss
    if criterion is None:
        criterion = UnimodalConcentratedLoss(lambda_value=1000)
    
    # Initialize metrics tracking
    running_loss = 0.0
    running_conc_loss = 0.0
    running_uni_loss = 0.0
    correct = 0
    total = 0
    
    # Add tracking for MAE calculation
    all_pred_values = []
    all_target_values = []
    
    # Determine if we're using UnimodalConcentratedLoss
    using_uc_loss = isinstance(criterion, UnimodalConcentratedLoss) or \
                    isinstance(criterion, UnimodalConcentratedLossWithWarmup) if \
                    'UnimodalConcentratedLossWithWarmup' in globals() else False
    
    # Use tqdm for progress bar
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} Training', ncols=100)
    
    for batch_idx, (inputs, ages) in enumerate(pbar):
        inputs, ages = inputs.to(device), ages.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Handle different return formats of VOLO
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Main classification output
        
        # Calculate loss - using either provided criterion or our UC loss
        if using_uc_loss:
            loss, conc_loss, uni_loss = criterion(outputs, ages)
            running_conc_loss += conc_loss.item()
            running_uni_loss += uni_loss.item()
        else:
            # If using a standard loss function, just call it normally
            loss = criterion(outputs, ages)
            conc_loss = torch.tensor(0.0)
            uni_loss = torch.tensor(0.0)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update running losses
        running_loss += loss.item()
        
        # Get predictions for accuracy calculation
        _, predicted = torch.max(outputs.data, 1)
        total += ages.size(0)
        correct += (predicted == ages).sum().item()
        
        # Calculate expected values for MAE
        probs = F.softmax(outputs, dim=1)
        labels = torch.arange(0, num_classes, device=device).float()
        pred_values = torch.sum(probs * labels.view(1, -1), dim=1)
        
        # Store for MAE calculation
        all_pred_values.extend(pred_values.detach().cpu().numpy())
        all_target_values.extend(ages.cpu().numpy())
        
        # Calculate current MAE for display
        if len(all_pred_values) > 0:
            current_mae = np.mean(np.abs(np.array(all_pred_values) - np.array(all_target_values)))
        else:
            current_mae = 0.0
        
        # Update progress bar
        avg_loss = running_loss / (batch_idx + 1)
        acc = 100.0 * correct / total
        
        pbar_postfix = {'loss': f'{avg_loss:.4f}', 'acc': f'{acc:.2f}%', 'MAE': f'{current_mae:.2f}'}
        if using_uc_loss:
            avg_conc = running_conc_loss / (batch_idx + 1)
            avg_uni = running_uni_loss / (batch_idx + 1)
            pbar_postfix.update({'conc': f'{avg_conc:.4f}', 'uni': f'{avg_uni:.4f}'})
            
        pbar.set_postfix(pbar_postfix)
        
        # Save checkpoint at specified frequency if provided
        if (batch_idx + 1) % save_freq == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f'checkpoint_epoch{epoch+1}_batch{batch_idx+1}.pth'
            )
            torch.save({
                'epoch': epoch,
                'batch': batch_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'mae': current_mae
            }, checkpoint_path)
            print(f'Checkpoint saved at {checkpoint_path}')
    
    # Calculate final metrics
    train_mae = np.mean(np.abs(np.array(all_pred_values) - np.array(all_target_values)))
    
    # Calculate accuracy within thresholds
    errors = np.abs(np.array(all_pred_values) - np.array(all_target_values))
    within_1 = np.mean(errors <= 1) * 100
    within_3 = np.mean(errors <= 3) * 100
    within_5 = np.mean(errors <= 5) * 100
    
    # Create comprehensive metrics dictionary
    metrics = {
        'loss': running_loss / len(dataloader),
        'acc': 100.0 * correct / total,
        'mae': train_mae,
        'within_1': within_1,
        'within_3': within_3,
        'within_5': within_5
    }
    
    # Add component losses for UC loss
    if using_uc_loss:
        metrics.update({
            'conc_loss': running_conc_loss / len(dataloader),
            'uni_loss': running_uni_loss / len(dataloader)
        })
    
    return metrics


# In[7]:


def validate(model, val_loader, criterion, device, num_classes=122):
    model.eval()
    
    running_loss = 0.0
    running_conc_loss = 0.0
    running_uni_loss = 0.0
    all_preds = []
    all_targets = []
    correct = 0
    total = 0
    
    # Determine if we're using UnimodalConcentratedLoss
    using_uc_loss = isinstance(criterion, UnimodalConcentratedLoss)
    
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc='Validating', file=sys.stdout, ncols=150):
            # Handle different batch data formats
            if isinstance(batch_data, (list, tuple)):
                if len(batch_data) == 3:
                    # Format is (inputs, ages, filenames)
                    inputs, ages, _ = batch_data
                elif len(batch_data) == 2:
                    # Format is (inputs, ages)
                    inputs, ages = batch_data
                else:
                    print(f"Unexpected batch structure: {[type(item) for item in batch_data]}")
                    print(f"Batch data length: {len(batch_data)}")
                    raise ValueError("Unexpected batch data format")
            else:
                print(f"Batch data type: {type(batch_data)}")
                raise ValueError("Expected batch data to be a tuple or list")
                
            # Move data to device
            inputs = inputs.to(device)
            ages = ages.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Handle different return formats of VOLO
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Main classification output
            
            # Calculate loss based on the criterion type
            if using_uc_loss:
                loss, conc_loss, uni_loss = criterion(outputs, ages)
                running_conc_loss += conc_loss.item()
                running_uni_loss += uni_loss.item()
            else:
                loss = criterion(outputs, ages)
            
            running_loss += loss.item()
            
            # Get predictions for classification accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += ages.size(0)
            correct += (predicted == ages).sum().item()
            
            # Calculate expected value for MAE
            probs = F.softmax(outputs, dim=1)
            labels = torch.arange(0, num_classes, device=device).float()
            pred_values = torch.sum(probs * labels.view(1, -1), dim=1)
            
            # Save predictions and targets for MAE calculation
            all_preds.extend(pred_values.cpu().numpy())
            all_targets.extend(ages.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    mae = np.mean(np.abs(all_preds - all_targets))
    
    # Calculate accuracy within thresholds
    errors = np.abs(all_preds - all_targets)
    within_1 = np.mean(errors <= 1) * 100
    within_3 = np.mean(errors <= 3) * 100
    within_5 = np.mean(errors <= 5) * 100
    
    # Print validation results
    print(f"Validation MAE: {mae:.4f}")
    print(f"Within 1 year: {within_1:.2f}%, Within 3 years: {within_3:.2f}%, Within 5 years: {within_5:.2f}%")
    
    # Create metrics dictionary
    metrics = {
        'loss': running_loss / len(val_loader),
        'acc': 100.0 * correct / total,
        'mae': mae,
        'within_1': within_1,
        'within_3': within_3,
        'within_5': within_5
    }
    
    if using_uc_loss:
        metrics.update({
            'conc_loss': running_conc_loss / len(val_loader),
            'uni_loss': running_uni_loss / len(val_loader)
        })
    
    return metrics


# In[8]:


def generate_unimodal_concentrated_distribution(age, num_classes=100, variance_scale=3.0):
    """
    Generate a distribution following the Unimodal-Concentrated approach
    
    Args:
        age: The target age
        num_classes: Total number of age classes
        variance_scale: Controls the width of the distribution (sample difficulty)
    
    Returns:
        Probability distribution as a numpy array
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a tensor for all possible ages
    labels = torch.arange(0, num_classes, device=device).float()
    
    # Create a normal distribution centered at the target age
    distances = (labels - age).pow(2)
    distribution = torch.exp(-distances / (2 * variance_scale))
    
    # Normalize to create a probability distribution
    distribution = distribution / distribution.sum()
    
    # Ensure the distribution is unimodal by enforcing monotonicity
    # Before the peak (age), probabilities should increase
    # After the peak, probabilities should decrease
    for i in range(1, num_classes):
        if i < age:
            # Ensure monotonically increasing before peak
            distribution[i] = torch.max(distribution[i], distribution[i-1])
        elif i > age:
            # Ensure monotonically decreasing after peak
            distribution[i] = torch.min(distribution[i], distribution[i-1])
    
    # Re-normalize
    distribution = distribution / distribution.sum()
    
    return distribution.cpu().numpy()

def visualize_unimodal_concentrated():
    """
    Visualize Unimodal-Concentrated distributions for different ages and variance scales
    """
    # Set up the figure with 3 rows and 3 columns
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Unimodal-Concentrated Distributions', fontsize=16)
    
    # Sample ages to visualize
    sample_ages = [10, 30, 60]
    
    # Different variance scales to visualize (simulating sample difficulty)
    variance_scales = [1.0, 3.0, 5.0]
    
    # Row labels
    variance_labels = ["Low Variance (Easy Sample)", 
                       "Medium Variance", 
                       "High Variance (Hard Sample)"]
    
    # Generate and plot distributions for each age and variance
    for i, (scale, label) in enumerate(zip(variance_scales, variance_labels)):
        for j, age in enumerate(sample_ages):
            # Generate the distribution
            distribution = generate_unimodal_concentrated_distribution(
                age=age,
                num_classes=100,
                variance_scale=scale
            )
            
            # Get the subplot
            ax = axes[i, j]
            
            # Plot the distribution
            ax.bar(range(100), distribution, alpha=0.7)
            ax.set_title(f'Age {age}, {label}')
            ax.axvline(x=age, color='r', linestyle='--', label='True Age')
            
            # Set axis labels
            ax.set_xlabel('Age')
            ax.set_ylabel('Probability')
            
            # Focus the view around the target age
            view_margin = max(20, int(10 * scale))
            ax.set_xlim(max(0, age - view_margin), min(99, age + view_margin))
            
            # Add a legend
            ax.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the subtitle
    plt.savefig('unimodal_concentrated_only.png', dpi=300)
    plt.show()
    
    # plt.close()

def visualize_unimodal_properties():
    """
    Visualize the unimodal property of the distribution
    """
    # Set up the figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Unimodal Property of Unimodal-Concentrated Distributions', fontsize=16)
    
    # Sample ages to visualize
    sample_ages = [10, 30, 60]
    
    # Generate distributions
    for j, age in enumerate(sample_ages):
        # Generate the distribution
        distribution = generate_unimodal_concentrated_distribution(
            age=age,
            num_classes=100,
            variance_scale=3.0
        )
        
        # Get the subplot
        ax = axes[j]
        
        # Plot the distribution as line for better visibility of monotonicity
        x = np.arange(100)
        ax.plot(x, distribution, linewidth=2)
        ax.fill_between(x, 0, distribution, alpha=0.3)
        
        # Add vertical line for true age
        ax.axvline(x=age, color='r', linestyle='--', label='True Age')
        
        # Highlight the monotonicity property
        # Before the peak
        if age > 10:
            ax.annotate('Monotonically\nIncreasing', 
                     xy=(age-10, distribution[age-10]),
                     xytext=(age-20, distribution[age-10] + 0.02),
                     arrowprops=dict(arrowstyle='->'))
        
        # After the peak
        if age < 90:
            ax.annotate('Monotonically\nDecreasing', 
                     xy=(age+10, distribution[age+10]),
                     xytext=(age+20, distribution[age+10] + 0.02),
                     arrowprops=dict(arrowstyle='->'))
        
        # Set axis labels and title
        ax.set_title(f'Age {age}')
        ax.set_xlabel('Age')
        ax.set_ylabel('Probability')
        
        # Focus the view around the target age
        view_margin = 30
        ax.set_xlim(max(0, age - view_margin), min(99, age + view_margin))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the subtitle
    plt.savefig('unimodal_property.png', dpi=300)
    plt.show()
    
    # plt.close()

def visualize_adaptive_concentration():
    """
    Visualize how the distribution adapts to sample difficulty
    """
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Adaptive Concentration Based on Sample Difficulty', fontsize=16)
    
    # Target age
    age = 40
    
    # Generate distributions with different variances
    variance_scales = [1.0, 2.0, 4.0, 8.0]
    labels = ['Very Easy', 'Easy', 'Difficult', 'Very Difficult']
    colors = ['darkgreen', 'green', 'orange', 'red']
    
    # Plot each distribution
    x = np.arange(100)
    for scale, label, color in zip(variance_scales, labels, colors):
        distribution = generate_unimodal_concentrated_distribution(
            age=age,
            num_classes=100,
            variance_scale=scale
        )
        
        ax.plot(x, distribution, label=f'{label} Sample (Variance={scale})', 
               linewidth=2, color=color)
    
    # Add vertical line for true age
    ax.axvline(x=age, color='black', linestyle='--', label='True Age')
    
    # Set axis labels
    ax.set_title(f'Adaptive Concentration for Age {age}')
    ax.set_xlabel('Age')
    ax.set_ylabel('Probability')
    
    # Focus the view
    ax.set_xlim(10, 70)
    
    # Add a legend
    ax.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the subtitle
    plt.savefig('adaptive_concentration.png', dpi=300)
    plt.show()
    # plt.close()

# Run all visualizations
visualize_unimodal_concentrated()
visualize_unimodal_properties()
visualize_adaptive_concentration()


# In[9]:


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
    'num_epochs': 50,
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
# criterion = MeanVarianceLoss(num_classes=122, lambda_mean=0.2, lambda_var=0.05)

# criterion = SORDLoss(num_classes=122, temperature=0.5)
criterion = UnimodalConcentratedLoss(lambda_value=100)

# Apply initial unfreezing strategy (head only)
optimizer, scheduler = apply_unfreezing_strategy(model, 0, config)

# Initial lr scheduler if not created by unfreezing strategy
if scheduler is None:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])

# Training loop
best_val_loss = float('inf')

print(model)



# Modified training loop with unimodal-concentrated loss
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
    
    # Check if we're using the UnimodalConcentratedLoss
    using_uc_loss = isinstance(criterion, UnimodalConcentratedLoss)
    
    # Train one epoch
    train_results = train_one_epoch(
        model, train_loader, criterion, optimizer, device, 
        epoch, epoch_checkpoint_dir, config['save_freq']
    )
    
    # Extract all metrics properly
    train_loss = train_results['loss']
    train_acc = train_results['acc']
    train_mae = train_results['mae']  # This is now properly calculated in train_one_epoch
    
    # If using UC loss, extract component losses
    if 'conc_loss' in train_results:
        train_conc_loss = train_results['conc_loss']
        train_uni_loss = train_results['uni_loss']
        print(f"Train Loss: {train_loss:.4f} (Conc: {train_conc_loss:.4f}, Uni: {train_uni_loss:.4f}) - MAE: {train_mae:.2f} - Acc: {train_acc:.2f}%")
    else:
        print(f"Train Loss: {train_loss:.4f} - MAE: {train_mae:.2f} - Acc: {train_acc:.2f}%")
    
    # Validate
    val_results = validate(model, val_loader, criterion, device)
    
    # Extract validation metrics
    if using_uc_loss:
        val_loss = val_results['loss']
        val_mae = val_results['mae']
        val_within_5 = val_results['within_5']
        print(f"Val Loss: {val_loss:.4f} - MAE: {val_mae:.4f} - Within 5 years: {val_within_5:.2f}%")
    else:
        # For backward compatibility with your existing code
        val_loss = val_results if isinstance(val_results, tuple) and len(val_results) == 2 else val_results['loss']
        val_mae = val_results[1] if isinstance(val_results, tuple) and len(val_results) == 2 else val_results.get('mae', 0.0)
    
    # Update learning rate
    scheduler.step()
    
    # Save checkpoint if it's the best model so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_path = os.path.join(config['output'], f"{config['model']}_best.pth")
        
        # Determine unfreezing strategy
        unfreezing_strategy = next((strategy for ep, strategy in {
            0: 'head_only', 3: 'post_network', 6: 'partial_transformer', 
            9: 'full_transformer', 12: 'include_outlooker'
        }.items() if ep <= epoch), 'include_outlooker')
        
        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_mae': val_mae,
            'unfreezing_strategy': unfreezing_strategy,
            'using_uc_loss': using_uc_loss
        }, best_model_path)
        print(f"Saved best model with loss {val_loss:.4f}")
    
    # Save checkpoint at the end of each epoch
    epoch_checkpoint_path = os.path.join(config['output'], f"{config['model']}_epoch{epoch+1}.pth")
    
    # Determine unfreezing strategy
    unfreezing_strategy = next((strategy for ep, strategy in {
        0: 'head_only', 3: 'post_network', 6: 'partial_transformer', 
        9: 'full_transformer', 12: 'include_outlooker'
    }.items() if ep <= epoch), 'include_outlooker')
    
    # Save checkpoint
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_mae': val_mae,
        'unfreezing_strategy': unfreezing_strategy,
        'using_uc_loss': using_uc_loss
    }, epoch_checkpoint_path)
    print(f"Saved checkpoint at end of epoch {epoch+1}")

# Save final model
final_model_path = os.path.join(config['output'], f"{config['model']}_unimodal_concentrated_final.pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'unfreezing_strategy': 'include_outlooker',  # Final model has everything unfrozen
    'config': config,
    'using_uc_loss': isinstance(criterion, UnimodalConcentratedLoss)
}, final_model_path)
print(f"Saved final model to {final_model_path}")

