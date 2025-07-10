#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

# DLDL-v2 Loss function with proper two-head design
class DLDLv2Loss(nn.Module):
    def __init__(self, num_classes=122, lambda_val=1.0, sigma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_val = lambda_val  # Weight balance between KL and L1 loss
        self.sigma = sigma  # Standard deviation for Gaussian label distribution
        
    def create_label_distribution(self, targets):
        """
        Create Gaussian label distributions for each target age
        targets: batch of integer age labels
        returns: label distribution vectors for each target
        """
        batch_size = targets.size(0)
        device = targets.device
        
        # Create the ages tensor on the same device as targets
        ages = torch.arange(self.num_classes, device=device).float()
        
        # Create distribution vectors for the batch
        distributions = torch.zeros(batch_size, self.num_classes, device=device)
        
        for i, target in enumerate(targets):
            # Create Gaussian distribution centered at the target age
            mu = target.float()
            sigma = self.sigma
            
            # Formula: exp(-0.5 * ((x - mu)/sigma)^2)
            gauss = torch.exp(-0.5 * ((ages - mu) / sigma) ** 2)
            
            # Normalize to sum to 1
            gauss = gauss / gauss.sum()
            
            distributions[i] = gauss
            
        return distributions
    
    def forward(self, distribution_logits, regression_output, targets):
        """
        Calculate combined loss: KL divergence for distribution + L1 for direct regression
        distribution_logits: output from the distribution head
        regression_output: output from the regression head
        targets: ground truth ages
        """
        # Create target label distributions
        target_distributions = self.create_label_distribution(targets)
        
        # For KL divergence loss
        log_probs = F.log_softmax(distribution_logits, dim=1)
        kl_loss = F.kl_div(log_probs, target_distributions, reduction='batchmean')
        
        # L1 loss for direct regression
        l1_loss = F.l1_loss(regression_output.squeeze(), targets.float())
        
        # Combined loss
        total_loss = kl_loss + self.lambda_val * l1_loss
        
        # Also calculate expected value from the distribution for evaluation
        probs = F.softmax(distribution_logits, dim=1)
        device = distribution_logits.device
        ages = torch.arange(self.num_classes, device=device).float()
        expected_ages = torch.sum(probs * ages.unsqueeze(0), dim=1)
        
        return total_loss, kl_loss, l1_loss, expected_ages, regression_output.squeeze()

# Dual-head VOLO model for DLDL-v2
class DualHeadVOLO(nn.Module):
    def __init__(self, num_classes=122, checkpoint_path=None, model_name='volo_d1'):
        super(DualHeadVOLO, self).__init__()
        
        # Create the base VOLO model
        self.base_model = getattr(models, model_name)(img_size=224)
        
        # Load pre-trained weights
        if checkpoint_path:
            _logger.info(f"Loading pre-trained weights from {checkpoint_path}")
            load_pretrained_weights(
                model=self.base_model,
                checkpoint_path=checkpoint_path,
                use_ema=False,
                strict=False
            )
        
        # Get the feature dimension
        in_features = self.base_model.head.in_features
        
        # Replace the classification head with two separate heads
        self.base_model.head = nn.Identity()  # Remove the original head
        
        # 1. Distribution head for label distribution learning
        self.distribution_head = nn.Linear(in_features, num_classes)
        
        # 2. Regression head for direct age prediction
        self.regression_head = nn.Linear(in_features, 1)
        
        # Disable auxiliary head if present
        if hasattr(self.base_model, 'aux_head'):
            self.base_model.aux_head = nn.Identity()
    
    def forward(self, x):
        # Get features from the base model
        features = self.base_model(x)
        
        # Handle different output formats of VOLO
        if isinstance(features, tuple):
            features = features[0]
        
        # Forward through both heads
        distribution_logits = self.distribution_head(features)
        regression_output = self.regression_head(features)
        
        return distribution_logits, regression_output

def create_model(num_classes=122, checkpoint_path=None, model_name='volo_d1'):
    """Create and load dual-head VOLO model for DLDL-v2"""
    model = DualHeadVOLO(num_classes=num_classes, 
                         checkpoint_path=checkpoint_path, 
                         model_name=model_name)
    return model

def apply_unfreezing_strategy(model, epoch, config):
    """Apply the appropriate unfreezing strategy based on the current epoch"""
    # Define unfreezing schedule - which epochs to change strategy
    unfreezing_milestones = {
        0: 'head_only',          # Start with only head unfrozen
        3: 'post_network',       # After 3 epochs, unfreeze post_network
        6: 'partial_transformer', # After 6 epochs, unfreeze partial transformer
        9: 'full_transformer',    # After 9 epochs, unfreeze full transformer
        12: 'include_outlooker'   # After 12 epochs, unfreeze everything
    }
    
    # Check if we need to update unfreezing strategy
    if epoch in unfreezing_milestones:
        strategy = unfreezing_milestones[epoch]
        print(f"Updating unfreezing strategy to: {strategy}")
        
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
            
        # Always unfreeze the two heads
        for param in model.distribution_head.parameters():
            param.requires_grad = True
        for param in model.regression_head.parameters():
            param.requires_grad = True
            
        # Apply the appropriate unfreezing strategy to the base model
        if strategy == 'head_only':
            pass  # Only heads are unfrozen
                
        elif strategy == 'post_network':
            # Unfreeze post_network (class attention blocks)
            if model.base_model.post_network is not None:
                for block in model.base_model.post_network:
                    for param in block.parameters():
                        param.requires_grad = True
                        
        elif strategy == 'partial_transformer':
            # Unfreeze post_network and last transformer blocks
            if model.base_model.post_network is not None:
                for block in model.base_model.post_network:
                    for param in block.parameters():
                        param.requires_grad = True
            # Unfreeze the last transformer blocks
            transformer_idx = [2, 3]
            for i in transformer_idx:
                if i < len(model.base_model.network):
                    for block in model.base_model.network[i]:
                        for param in block.parameters():
                            param.requires_grad = True
                    
        elif strategy == 'full_transformer':
            # Unfreeze all transformer blocks (but not the early outlooker blocks)
            if model.base_model.post_network is not None:
                for block in model.base_model.post_network:
                    for param in block.parameters():
                        param.requires_grad = True
            # Unfreeze all transformer blocks
            transformer_idx = [2, 3, 4]
            for i in transformer_idx:
                if i < len(model.base_model.network):
                    for block in model.base_model.network[i]:
                        for param in block.parameters():
                            param.requires_grad = True

        elif strategy == 'include_outlooker':
            # Unfreeze everything
            for param in model.parameters():
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

def train_one_epoch_dldlv2(model, dataloader, criterion, optimizer, device, epoch, checkpoint_dir, save_freq=50):
    model.train()
    running_loss = 0.0
    running_kl_loss = 0.0
    running_l1_loss = 0.0
    running_distr_mae = 0.0
    running_reg_mae = 0.0
    total = 0
    start_time = time.time()
    
    print(f"Epoch {epoch+1}/{config['num_epochs']}")
    print("----------")
    
    # Create a progress bar using tqdm
    pbar = tqdm(total=len(dataloader), desc="Train", file=sys.stdout) #, ncols=150)
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass through the dual-head model
        distribution_logits, regression_output = model(inputs)
        
        # Calculate losses using the criterion
        total_loss, kl_loss, l1_loss, expected_ages, direct_ages = criterion(
            distribution_logits, regression_output, targets)
        
        # Backward pass and optimize
        total_loss.backward()
        optimizer.step()
        
        # Calculate metrics
        batch_size = inputs.size(0)
        running_loss += total_loss.item() * batch_size
        running_kl_loss += kl_loss.item() * batch_size
        running_l1_loss += l1_loss.item() * batch_size
        
        # Calculate MAE for both prediction methods
        distr_mae = F.l1_loss(expected_ages, targets.float())
        reg_mae = F.l1_loss(direct_ages, targets.float())
        running_distr_mae += distr_mae.item() * batch_size
        running_reg_mae += reg_mae.item() * batch_size
        
        total += batch_size
        
        # Calculate current metrics
        current_avg_loss = running_loss / total
        current_avg_kl = running_kl_loss / total
        current_avg_l1 = running_l1_loss / total
        current_avg_distr_mae = running_distr_mae / total
        current_avg_reg_mae = running_reg_mae / total
        
        # Update progress bar
        pbar.set_description(
            f"Train: loss={current_avg_loss:.4f}, kl={current_avg_kl:.4f}, l1={current_avg_l1:.4f}, "
            f"distr_mae={current_avg_distr_mae:.2f}, reg_mae={current_avg_reg_mae:.2f}"
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
                'distr_mae': current_avg_distr_mae,
                'reg_mae': current_avg_reg_mae
            }, checkpoint_path)
            print(f"\nSaved checkpoint at epoch {epoch+1}, batch {batch_idx+1} to {checkpoint_path}")
    
    # Close the progress bar
    pbar.close()
    
    # Calculate final metrics for the epoch
    epoch_loss = running_loss / total
    epoch_kl_loss = running_kl_loss / total
    epoch_l1_loss = running_l1_loss / total
    epoch_distr_mae = running_distr_mae / total
    epoch_reg_mae = running_reg_mae / total
    
    print(f"Train Loss: {epoch_loss:.4f} (KL: {epoch_kl_loss:.4f}, L1: {epoch_l1_loss:.4f})")
    print(f"Train MAE: Distribution: {epoch_distr_mae:.4f}, Regression: {epoch_reg_mae:.4f}")
    
    return epoch_loss, epoch_distr_mae, epoch_reg_mae

def validate_dldlv2(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_kl_loss = 0.0
    running_l1_loss = 0.0
    running_distr_mae = 0.0
    running_reg_mae = 0.0
    total = 0
    start_time = time.time()
    
    # Create a progress bar using tqdm
    pbar = tqdm(total=len(dataloader), desc="Val", file=sys.stdout) #, ncols=150)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass through the dual-head model
            distribution_logits, regression_output = model(inputs)
            
            # Calculate losses using the criterion
            total_loss, kl_loss, l1_loss, expected_ages, direct_ages = criterion(
                distribution_logits, regression_output, targets)
            
            # Calculate metrics
            batch_size = inputs.size(0)
            running_loss += total_loss.item() * batch_size
            running_kl_loss += kl_loss.item() * batch_size
            running_l1_loss += l1_loss.item() * batch_size
            
            # Calculate MAE for both prediction methods
            distr_mae = F.l1_loss(expected_ages, targets.float())
            reg_mae = F.l1_loss(direct_ages, targets.float())
            running_distr_mae += distr_mae.item() * batch_size
            running_reg_mae += reg_mae.item() * batch_size
            
            total += batch_size
            
            # Update progress bar
            pbar.set_description(
                f"Val: loss={running_loss/total:.4f}, kl={running_kl_loss/total:.4f}, l1={running_l1_loss/total:.4f}, "
                f"distr_mae={running_distr_mae/total:.2f}, reg_mae={running_reg_mae/total:.2f}"
            )
            pbar.update(1)
    
    # Close the progress bar
    pbar.close()
    
    # Calculate final metrics for validation
    val_loss = running_loss / total
    val_kl_loss = running_kl_loss / total
    val_l1_loss = running_l1_loss / total
    val_distr_mae = running_distr_mae / total
    val_reg_mae = running_reg_mae / total
    
    print(f"Val Loss: {val_loss:.4f} (KL: {val_kl_loss:.4f}, L1: {val_l1_loss:.4f})")
    print(f"Val MAE: Distribution: {val_distr_mae:.4f}, Regression: {val_reg_mae:.4f}")
    
    # Return the best MAE (minimum of distribution-based and regression-based)
    best_mae = min(val_distr_mae, val_reg_mae)
    return val_loss, best_mae


# In[3]:


def visualize_dldlv2_distributions(lambda_val=1.0, sigma_values=[1.0, 2.0, 5.0]):
    # Set up the figure
    fig, axes = plt.subplots(len(sigma_values), 3, figsize=(15, 4*len(sigma_values)))
    fig.suptitle('DLDL-v2 Label Distributions for Different Ages and Sigma Values', fontsize=16)
    
    # Sample ages to visualize
    sample_ages = [10, 40, 80]
    
    # Create a tensor with the sample ages
    targets = torch.tensor(sample_ages).long()
    
    # Generate distributions for each sigma
    for i, sigma in enumerate(sigma_values):
        # Create a DLDL-v2 loss instance with this sigma
        dldl_loss = DLDLv2Loss(num_classes=101, lambda_val=lambda_val, sigma=sigma)
        
        # Get the label distributions for all sample ages
        label_distributions = dldl_loss.create_label_distribution(targets)
        
        # Plot each distribution
        for j, age in enumerate(sample_ages):
            if len(sigma_values) > 1:
                ax = axes[i, j]
            else:
                ax = axes[j]
            
            # Get the distribution for this age
            distribution = label_distributions[j].detach().numpy()
            
            # Plot the distribution
            ax.bar(range(101), distribution, alpha=0.7)
            ax.set_title(f'Age {age}, Sigma={sigma}')
            ax.axvline(x=age, color='r', linestyle='--', label='True Age')
            
            # Set axis labels
            ax.set_xlabel('Age')
            ax.set_ylabel('Probability')
            
            # Focus the view around the target age
            view_margin = max(20, int(10 * sigma))  # Adjust view based on sigma
            ax.set_xlim(max(0, age - view_margin), min(100, age + view_margin))
            
            # Add a legend
            ax.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the subtitle
    plt.savefig('dldlv2_distributions.png')
    plt.show()

# Configuration
config = {
    'model': 'volo_d1',
    'checkpoint': '/home/meem/backup/d1_224_84.2.pth.tar',
    # Data paths - UPDATE THESE TO YOUR PATHS
        # 'train_csv': '/home/meem/backup/Age Datasets/train_annotations.csv',
        # 'val_csv': '/home/meem/backup/Age Datasets/val_annotations.csv',
        # 'train_dir': '/home/meem/backup/Age Datasets/train',
        # 'val_dir': '/home/meem/backup/Age Datasets/val',


    'train_csv': '/home/meem/filtered/unified_age_dataset/train_annotations.csv',
    'val_csv': '/home/meem/filtered/unified_age_dataset/val_annotations.csv',
    'train_dir': '/home/meem/filtered/unified_age_dataset/train',
    'val_dir': '/home/meem/filtered/unified_age_dataset/val',
    'output': './output',
    'checkpoint_dir': './checkpoints_dldlv2',
    'batch_size': 16,
    'num_epochs': 20,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'save_freq': 50,  # Save checkpoint every 50 batches
    'lambda_val': 1.0,  # Weight between KL and L1 loss
    'sigma': 2.0,  # Sigma for the Gaussian label distributions
}

if __name__ == "__main__":
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
    if os.path.exists(config['checkpoint_dir']):
        shutil.rmtree(config['checkpoint_dir'])
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
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
    
    # Create model - now using the dual-head architecture
    model = create_model(
        num_classes=122,  # 0-121 age range
        checkpoint_path=config['checkpoint'],
        model_name=config['model']
    )
    model = model.to(device)
    
    
    # Print model structure
    print(model)
    
    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Trainable parameters: {trainable:,} ({trainable/total:.2%} of total {total:,})')
    
    # Loss function - use DLDL-v2 loss
    criterion = DLDLv2Loss(
        num_classes=122,
        lambda_val=config['lambda_val'],
        sigma=config['sigma']
    )
    
    # Apply initial unfreezing strategy (head only)
    optimizer, scheduler = apply_unfreezing_strategy(model, 0, config)
    
    # Initial lr scheduler if not created by unfreezing strategy
    if scheduler is None:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    # Visualize the distributions before training
    visualize_dldlv2_distributions(lambda_val=config['lambda_val'], sigma_values=[1.0, 2.0, 5.0])
    
    # Training loop
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    
    # For tracking metrics
    train_losses = []
    train_maes = []
    val_losses = []
    val_maes = []
    
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
        
        train_loss, train_distr_mae, train_reg_mae = train_one_epoch_dldlv2(
            model, train_loader, criterion, optimizer, device, 
            epoch, epoch_checkpoint_dir, config['save_freq']
        )
        
        # Track training metrics (use the better of the two MAEs)
        train_losses.append(train_loss)
        train_maes.append(min(train_distr_mae, train_reg_mae))
        
        # Validate
        val_loss, val_mae = validate_dldlv2(model, val_loader, criterion, device)
        
        # Track validation metrics
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint if it's the best model so far
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_loss = val_loss
            best_model_path = os.path.join(config['output'], f"{config['model']}_dldlv2_best.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'config': {
                    'lambda_val': config['lambda_val'],
                    'sigma': config['sigma']
                }
            }, best_model_path)
            print(f"Saved best model with MAE {val_mae:.4f}")
        
        # Save checkpoint at the end of each epoch
        epoch_checkpoint_path = os.path.join(config['output'], f"{config['model']}_dldlv2_epoch{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_mae': val_mae
        }, epoch_checkpoint_path)
        print(f"Saved checkpoint at end of epoch {epoch+1}")
        
        # Plot training progress
        plt.figure(figsize=(12, 5))
        
        # Plot MAE
        plt.subplot(1, 2, 1)
        plt.plot(train_maes, label='Train MAE')
        plt.plot(val_maes, label='Val MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.title('MAE Over Training')
        
        # Plot Loss
        plt.subplot(1, 2, 2)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Over Training')
        
        plt.tight_layout()
        plt.savefig(os.path.join(config['output'], 'training_progress.png'))
        plt.close()
    
    # Save final model
    final_model_path = os.path.join(config['output'], f"{config['model']}_dldlv2_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'lambda_val': config['lambda_val'],
            'sigma': config['sigma']
        }
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")
    print(f"Best validation MAE: {best_val_mae:.4f}")

