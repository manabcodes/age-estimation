#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from torch.profiler import profile, record_function, ProfilerActivity
import clip
import warnings
# warnings.filterwarnings('ignore')


# In[2]:


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_logger = logging.getLogger('farl_age_estimation')

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

class FaRLMLP(nn.Module):
    """FaRL backbone with MLP head for age estimation (regression mode as in original paper)"""
    
    def __init__(self, farl_model_path=None, hidden_dims=[512], 
                 dropout_rate=0.1, freeze_backbone=True):
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
        # CLIP ViT-B/16 outputs 512 features, not 768
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_features = self.backbone(dummy_input)
            feature_dim = dummy_features.shape[-1]
            
        _logger.info(f"Detected feature dimension: {feature_dim}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            _logger.info("FaRL backbone frozen")
        
        # Build simple MLP head for regression (as in original FaRL paper)
        layers = []
        prev_dim = feature_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final output layer - single neuron for age regression
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        self.freeze_backbone = freeze_backbone
        
    def forward(self, x):
        # Extract features using FaRL backbone
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        
        # Pass through MLP head - returns single age value
        age_pred = self.mlp(features).squeeze()
        
        return age_pred
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
        _logger.info("FaRL backbone unfrozen")

# Original FaRL MLP paper uses simple regression with MAE or MSE loss
# No SORD loss needed - just direct age regression

# def apply_farl_unfreezing_strategy(model, epoch, config):
#     """Apply progressive unfreezing strategy for FaRL MLP"""
#     unfreezing_milestones = {
#         0: 'head_only',           # Only MLP head
#         5: 'backbone_partial',    # Unfreeze last few transformer blocks
#         10: 'backbone_full',      # Unfreeze entire backbone
#     }
    
#     if epoch in unfreezing_milestones:
#         strategy = unfreezing_milestones[epoch]
#         print(f"Updating unfreezing strategy to: {strategy}")
        
#         if strategy == 'head_only':
#             # Keep backbone frozen, only train MLP
#             for param in model.backbone.parameters():
#                 param.requires_grad = False
#             for param in model.mlp.parameters():
#                 param.requires_grad = True
                
#         elif strategy == 'backbone_partial':
#             # Unfreeze last few transformer blocks
#             for param in model.backbone.parameters():
#                 param.requires_grad = False
            
#             # Unfreeze last 3 transformer blocks
#             if hasattr(model.backbone, 'transformer'):
#                 for i in range(len(model.backbone.transformer.resblocks) - 3, 
#                              len(model.backbone.transformer.resblocks)):
#                     for param in model.backbone.transformer.resblocks[i].parameters():
#                         param.requires_grad = True
            
#             for param in model.mlp.parameters():
#                 param.requires_grad = True
                
#         elif strategy == 'backbone_full':
#             # Unfreeze entire model
#             model.unfreeze_backbone()
#             for param in model.mlp.parameters():
#                 param.requires_grad = True
        
#         # Adjust learning rate based on unfreezing stage
#         lr_divisors = {
#             'head_only': 1,
#             'backbone_partial': 2,
#             'backbone_full': 4
#         }
        
#         lr = config['lr'] / lr_divisors[strategy]
        
#         # Create new optimizer with adjusted learning rate
#         optimizer = optim.AdamW(
#             filter(lambda p: p.requires_grad, model.parameters()),
#             lr=lr,
#             weight_decay=config['weight_decay']
#         )
        
#         # Create new scheduler
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(
#             optimizer,
#             T_max=config['num_epochs'] - epoch,
#             eta_min=1e-6
#         )
        
#         # Print trainable parameters info
#         trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         total = sum(p.numel() for p in model.parameters())
#         print(f'Trainable parameters: {trainable:,} ({trainable/total:.2%} of total {total:,})')
#         print(f'Learning rate: {lr:.6f}')
        
#         return optimizer, scheduler
    
#     return None, None


def apply_farl_unfreezing_strategy(model, epoch, config):
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
            # Only unfreeze MLP head
            for param in model.mlp.parameters():
                param.requires_grad = True
                
        elif strategy == 'backbone_partial':
            # MLP + last few transformer blocks
            for param in model.mlp.parameters():
                param.requires_grad = True
            # Add backbone partial unfreezing here
                
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
    running_mae = 0.0
    total = 0
    start_time = time.time()
    
    print(f"Epoch {epoch+1}/{config['num_epochs']}")
    print("----------")
    
    pbar = tqdm(total=len(dataloader), desc="Train", file=sys.stdout)
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass - direct age regression
        age_predictions = model(inputs)
        
        # Calculate loss (MSE or L1)
        loss = criterion(age_predictions, targets.float())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        running_loss += loss.item() * inputs.size(0)
        mae = F.l1_loss(age_predictions, targets.float())
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
    epoch_mae = running_mae / total
    
    print(f"train Loss: {epoch_loss:.4f} MAE: {epoch_mae:.4f}")
    
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
            
            # Forward pass - direct age regression
            age_predictions = model(inputs)
            
            # Calculate loss (MSE or L1)
            loss = criterion(age_predictions, targets.float())
            
            # Calculate metrics
            running_loss += loss.item() * inputs.size(0)
            mae = F.l1_loss(age_predictions, targets.float())
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
    
    print(f"val Loss: {val_loss:.4f} MAE: {val_mae:.4f}")
    
    return val_loss, val_mae

def main():
    # Configuration
    config = {
        # Data paths - UPDATE THESE TO YOUR PATHS
        'train_csv': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/train_annotations.csv',
        'val_csv': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/val_annotations.csv',
        'train_dir': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/train',
        'val_dir': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/val',
        
        # FaRL model path - UPDATE THIS TO YOUR FARL MODEL PATH
        'farl_model_path': './FaRL-Base-Patch16-LAIONFace20M-ep64.pth',  # Set to path of FaRL .pth file if available
        
        # Output paths
        'output': './output_farl',
        'checkpoint_dir': './checkpoints_farl_mlp',
        
        # Training parameters
        'batch_size': 16,
        'num_epochs': 20,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        
        # Model parameters (simple regression as in original FaRL paper)
        'hidden_dims': [512],      # Simple 2-layer MLP (auto-detected->512->1)
        'dropout_rate': 0.1,
        'freeze_backbone': True,   # Start with frozen backbone
        
        # Loss function: 'mse' or 'l1' (MAE)
        'loss_function': 'l1',     # Original paper uses MAE (L1 loss)
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
                           std=[0.26862954, 0.26130258, 0.27577711]),  # CLIP normalization
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                           std=[0.26862954, 0.26130258, 0.27577711]),  # CLIP normalization
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
    model = FaRLMLP(
        farl_model_path=config['farl_model_path'],
        hidden_dims=config['hidden_dims'],
        dropout_rate=config['dropout_rate'],
        freeze_backbone=config['freeze_backbone']
    )
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _logger.info(f"Total parameters: {total_params:,}")
    _logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    
    # Loss function - simple regression as in original FaRL paper
    if config['loss_function'] == 'l1':
        criterion = nn.L1Loss()  # MAE loss (as used in original paper)
    else:
        criterion = nn.MSELoss()  # Alternative MSE loss
    
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
    lr_history = []
    
    for epoch in range(config['num_epochs']):
        # Apply progressive unfreezing if needed
        new_optimizer, new_scheduler = apply_farl_unfreezing_strategy(model, epoch, config)
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
        lr_history.append(optimizer.param_groups[0]['lr'])
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(config['output'], "farl_mlp_best.pth")
            
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
        checkpoint_path = os.path.join(config['checkpoint_dir'], f"farl_mlp_epoch{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_mae': val_mae,
            'config': config,
        }, checkpoint_path)
    
    # Save final model
    final_model_path = os.path.join(config['output'], "farl_mlp_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    # Plot learning rate schedule
    plt.figure(figsize=(10, 5))
    plt.plot(lr_history)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(config['output'], 'farl_learning_rate_schedule.png'))
    plt.show()

if __name__ == "__main__":
    main()


# In[10]:


from torchview import draw_graph

# For FaRL MLP

config = {
        # Data paths - UPDATE THESE TO YOUR PATHS
        'train_csv': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/train_annotations.csv',
        'val_csv': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/val_annotations.csv',
        'train_dir': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/train',
        'val_dir': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/val',
        
        # FaRL model path - UPDATE THIS TO YOUR FARL MODEL PATH
        'farl_model_path': './FaRL-Base-Patch16-LAIONFace20M-ep64.pth',  # Set to path of FaRL .pth file if available
        
        # Output paths
        'output': './output_farl',
        'checkpoint_dir': './checkpoints_farl_mlp',
        
        # Training parameters
        'batch_size': 16,
        'num_epochs': 20,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        
        # Model parameters (simple regression as in original FaRL paper)
        'hidden_dims': [512],      # Simple 2-layer MLP (auto-detected->512->1)
        'dropout_rate': 0.1,
        'freeze_backbone': True,   # Start with frozen backbone
        
        # Loss function: 'mse' or 'l1' (MAE)
        'loss_function': 'l1',     # Original paper uses MAE (L1 loss)
    }

# Set device and seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_logger.info(f"Using device: {device}")

torch.cuda.empty_cache()
torch.cuda.synchronize()

model = FaRLMLP(
    farl_model_path=config['farl_model_path'],
    hidden_dims=config['hidden_dims'],
    dropout_rate=config['dropout_rate'],
    freeze_backbone=config['freeze_backbone']
)
model = model.to(device)

model_graph = draw_graph(model, input_size=(1, 3, 224, 224))
model_graph.visual_graph
# Configure for high resolution and quality
model_graph.visual_graph.attr(
    dpi='300',           # High DPI
    size='24,24',        # Larger canvas
    rankdir='TB',        # Top to bottom layout
    fontsize='12',       # Readable font size
    fontname='Arial'     # Clean font
)
model_graph.visual_graph.render('./output_farl/farl_model_architecture', format='png')


# In[ ]:




