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
import clip
import warnings
# warnings.filterwarnings('ignore')


# In[2]:


def apply_ordinalclip_unfreezing_strategy(model, epoch, config):
    """Apply progressive unfreezing strategy for OrdinalCLIP"""
    unfreezing_milestones = {
        0: 'prompts_only',        # Only learnable prompts (context + rank embeddings)
        8: 'clip_partial',        # Unfreeze last few CLIP layers + prompts
        15: 'clip_full',          # Unfreeze entire CLIP model + prompts
    }
    
    if epoch in unfreezing_milestones:
        strategy = unfreezing_milestones[epoch]
        print(f"Updating unfreezing strategy to: {strategy}")
        
        # FREEZE EVERYTHING FIRST
        for param in model.parameters():
            param.requires_grad = False
            
        if strategy == 'prompts_only':
            # Only unfreeze learnable prompts
            model.context_tokens.requires_grad = True
            model.rank_embeddings.requires_grad = True
            for param in model.rank_projection.parameters():
                param.requires_grad = True
                
        elif strategy == 'clip_partial':
            # Unfreeze prompts + last few CLIP transformer blocks
            model.context_tokens.requires_grad = True
            model.rank_embeddings.requires_grad = True
            for param in model.rank_projection.parameters():
                param.requires_grad = True
            
            # Unfreeze last 3 transformer blocks of CLIP visual encoder
            if hasattr(model.clip_model.visual, 'transformer'):
                for i in range(len(model.clip_model.visual.transformer.resblocks) - 3, 
                             len(model.clip_model.visual.transformer.resblocks)):
                    for param in model.clip_model.visual.transformer.resblocks[i].parameters():
                        param.requires_grad = True
                        
        elif strategy == 'clip_full':
            # Unfreeze everything except text encoder (keep it frozen as per OrdinalCLIP design)
            for name, param in model.named_parameters():
                if 'clip_model.transformer' not in name and 'clip_model.positional_embedding' not in name and 'clip_model.text_projection' not in name and 'clip_model.ln_final' not in name:
                    param.requires_grad = True
        
        # Adjust learning rate based on unfreezing stage
        lr_divisors = {
            'prompts_only': 1,
            'clip_partial': 2,
            'clip_full': 4
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


def create_param_groups_ordinalclip(model, config, strategy):
    """Create parameter groups with different learning rates for OrdinalCLIP components"""
    if strategy == 'clip_full':
        # Different LRs for CLIP backbone vs prompt components
        clip_params = []
        prompt_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'clip_model.visual' in name:
                    clip_params.append(param)
                elif any(component in name for component in ['context_tokens', 'rank_embeddings', 'rank_projection']):
                    prompt_params.append(param)
        
        param_groups = []
        if prompt_params:
            param_groups.append({'params': prompt_params, 'lr': config['lr'], 'name': 'prompts'})
        if clip_params:
            param_groups.append({'params': clip_params, 'lr': config['lr'] / 10, 'name': 'clip_visual'})
        
        return param_groups
    else:
        return [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': config['lr']}]


def print_trainable_parameters(model):
    """Print detailed information about trainable parameters by component"""
    print("\n" + "="*50)
    print("TRAINABLE PARAMETERS BREAKDOWN")
    print("="*50)
    
    component_params = {
        'context_tokens': 0,
        'rank_embeddings': 0,
        'rank_projection': 0,
        'clip_visual': 0,
        'clip_text': 0,
        'other': 0
    }
    
    total_trainable = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        if param.requires_grad:
            total_trainable += param.numel()
            
            if 'context_tokens' in name:
                component_params['context_tokens'] += param.numel()
            elif 'rank_embeddings' in name:
                component_params['rank_embeddings'] += param.numel()
            elif 'rank_projection' in name:
                component_params['rank_projection'] += param.numel()
            elif 'clip_model.visual' in name:
                component_params['clip_visual'] += param.numel()
            elif any(text_component in name for text_component in ['clip_model.transformer', 'clip_model.positional_embedding', 'clip_model.text_projection', 'clip_model.ln_final']):
                component_params['clip_text'] += param.numel()
            else:
                component_params['other'] += param.numel()
    
    for component, count in component_params.items():
        if count > 0:
            percentage = (count / total_trainable) * 100 if total_trainable > 0 else 0
            print(f"{component:15}: {count:>10,} ({percentage:5.1f}% of trainable)")
    
    print("-" * 50)
    print(f"{'Total trainable':15}: {total_trainable:>10,} ({(total_trainable/total_params)*100:5.1f}% of total)")
    print(f"{'Total params':15}: {total_params:>10,}")
    print("=" * 50)



# In[ ]:


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_logger = logging.getLogger('ordinalclip_age_estimation')

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

class OrdinalCLIP(nn.Module):
    """OrdinalCLIP implementation for age estimation with learnable rank prompts"""
    
    def __init__(self, num_classes=101, context_length=4, rank_embedding_dim=512):
        super().__init__()
        
        # Load CLIP model
        self.clip_model, self.preprocess = clip.load("ViT-B/16", device="cpu")
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.num_classes = num_classes
        self.context_length = context_length
        self.rank_embedding_dim = rank_embedding_dim
        
        # Get CLIP dimensions
        self.clip_dim = self.clip_model.ln_final.weight.shape[0]  # 512 for ViT-B/16
        
        # Learnable context tokens (similar to CoOp)
        self.context_tokens = nn.Parameter(torch.randn(context_length, self.clip_dim))
        
        # Learnable rank embeddings for ordinal regression
        # Key innovation: model numerical continuity
        self.rank_embeddings = nn.Parameter(torch.randn(num_classes, rank_embedding_dim))
        
        # Projection layer to map rank embeddings to CLIP text space
        self.rank_projection = nn.Linear(rank_embedding_dim, self.clip_dim)
        
        # Initialize parameters
        nn.init.normal_(self.context_tokens, std=0.02)
        nn.init.normal_(self.rank_embeddings, std=0.02)
        
        # Create base text template
        self.register_buffer("token_prefix", clip.tokenize("a photo of a")[0, :1])  # "a"
        self.register_buffer("token_suffix", clip.tokenize("a photo of a")[0, 1:])   # "photo of a"
        
    def create_ordinal_embeddings(self):
        """Create ordinal-aware text embeddings using learnable rank embeddings"""
        # This method is now simplified since we handle sequence construction in encode_ordinal_text
        return self.rank_embeddings
    
    def encode_ordinal_text(self):
        """Encode ordinal text using learnable embeddings"""
        # Create text embeddings that match CLIP's expected sequence length (77)
        batch_size = self.num_classes
        seq_len = 77  # CLIP's standard sequence length
        
        # Initialize with zeros
        text_embeddings = torch.zeros(batch_size, seq_len, self.clip_dim, device=self.context_tokens.device)
        
        # Project rank embeddings to CLIP space
        rank_features = self.rank_projection(self.rank_embeddings)  # [num_classes, clip_dim]
        
        # Fill in the sequence: [SOS] + context_tokens + rank_embedding + [EOS] + padding
        # Position 0: Start of sequence (learnable)
        text_embeddings[:, 0, :] = rank_features  # Use rank as first token
        
        # Positions 1 to context_length: Context tokens
        for i in range(self.context_length):
            if i + 1 < seq_len:
                text_embeddings[:, i + 1, :] = self.context_tokens[i].unsqueeze(0).expand(batch_size, -1)
        
        # Position context_length + 1: End of sequence token (use last context token)
        if self.context_length + 1 < seq_len:
            text_embeddings[:, self.context_length + 1, :] = self.context_tokens[-1].unsqueeze(0).expand(batch_size, -1)
        
        # Add positional embeddings
        x = text_embeddings + self.clip_model.positional_embedding.unsqueeze(0)
        x = x.permute(1, 0, 2)  # [seq_len, num_classes, clip_dim]
        
        # Pass through transformer
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # [num_classes, seq_len, clip_dim]
        
        # Take the end-of-sequence token (position context_length + 1)
        eot_position = min(self.context_length + 1, seq_len - 1)
        x = self.clip_model.ln_final(x[:, eot_position, :])  # [num_classes, clip_dim]
        
        # Project to final space
        text_features = x @ self.clip_model.text_projection
        
        return text_features
    
    def forward(self, images):
        """Forward pass for OrdinalCLIP"""
        batch_size = images.size(0)
        
        # Encode images using CLIP
        image_features = self.clip_model.encode_image(images)
        
        # Encode ordinal text prototypes
        text_features = self.encode_ordinal_text()  # [num_classes, clip_dim]
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity logits
        logits = image_features @ text_features.T  # [batch_size, num_classes]
        
        # Scale by CLIP's temperature
        logits = logits * self.clip_model.logit_scale.exp()
        
        return logits
    
    def get_age_predictions(self, logits):
        """Convert logits to age predictions using expected value"""
        # Softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Age range (adjust based on your dataset)
        ages = torch.arange(self.num_classes, device=logits.device).float()
        
        # Expected value (weighted average)
        predicted_ages = torch.sum(probs * ages.unsqueeze(0), dim=1)
        
        return predicted_ages

class OrdinalRankingLoss(nn.Module):
    """Ordinal ranking loss to enforce order constraints"""
    
    def __init__(self, num_classes, margin=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        
    def forward(self, logits, targets):
        """
        Ordinal ranking loss that enforces P(age <= target) > P(age <= target-1)
        """
        batch_size = logits.size(0)
        device = logits.device
        
        # Clamp targets to valid range
        targets = torch.clamp(targets, 0, self.num_classes - 1)
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)
        
        # Compute cumulative probabilities P(age <= k)
        cumulative_probs = torch.cumsum(probs, dim=1)
        
        # For each sample, enforce ordinal constraints
        total_loss = 0.0
        count = 0
        
        for i in range(batch_size):
            target = targets[i].long()
            
            # Skip boundary cases
            if target <= 0 or target >= self.num_classes - 1:
                continue
                
            # Ordinal constraint: P(age <= target) should be > P(age <= target-1)
            current_prob = cumulative_probs[i, target]
            prev_prob = cumulative_probs[i, target - 1]
            
            # Ranking loss with margin
            loss = F.relu(self.margin - (current_prob - prev_prob))
            total_loss += loss
            count += 1
        
        if count > 0:
            return total_loss / count
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)

def train_one_epoch(model, dataloader, criterion, ranking_loss, optimizer, device, epoch, config):
    model.train()
    running_loss = 0.0
    running_ranking_loss = 0.0
    running_mae = 0.0
    total = 0
    
    print(f"Epoch {epoch+1}/{config['num_epochs']}")
    print("----------")
    
    pbar = tqdm(total=len(dataloader), desc="Train", file=sys.stdout)
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(inputs)
        
        # Main classification loss
        main_loss = criterion(logits, targets.long())
        
        # Ordinal ranking loss
        rank_loss = ranking_loss(logits, targets)
        
        # Combined loss
        total_loss = main_loss + config['ranking_weight'] * rank_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Calculate metrics
        running_loss += main_loss.item() * inputs.size(0)
        running_ranking_loss += rank_loss.item() * inputs.size(0)
        
        # Age predictions using expected value
        predicted_ages = model.get_age_predictions(logits)
        mae = F.l1_loss(predicted_ages, targets.float())
        running_mae += mae.item() * inputs.size(0)
        
        total += inputs.size(0)
        
        # Update progress bar
        current_avg_loss = running_loss / total
        current_avg_ranking_loss = running_ranking_loss / total
        current_avg_mae = running_mae / total
        
        pbar.set_description(
            f"Train: loss={current_avg_loss:.4f}, rank_loss={current_avg_ranking_loss:.4f}, mae={current_avg_mae:.2f}"
        )
        pbar.update(1)
    
    pbar.close()
    
    epoch_loss = running_loss / total
    epoch_ranking_loss = running_ranking_loss / total
    epoch_mae = running_mae / total
    
    print(f"train Loss: {epoch_loss:.4f} Ranking Loss: {epoch_ranking_loss:.4f} MAE: {epoch_mae:.4f}")
    
    return epoch_loss, epoch_mae

def validate(model, dataloader, criterion, ranking_loss, device, config):
    model.eval()
    running_loss = 0.0
    running_ranking_loss = 0.0
    running_mae = 0.0
    total = 0
    
    pbar = tqdm(total=len(dataloader), desc="Val", file=sys.stdout)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            logits = model(inputs)
            
            # Losses
            main_loss = criterion(logits, targets.long())
            rank_loss = ranking_loss(logits, targets)
            
            # Metrics
            running_loss += main_loss.item() * inputs.size(0)
            running_ranking_loss += rank_loss.item() * inputs.size(0)
            
            # Age predictions
            predicted_ages = model.get_age_predictions(logits)
            mae = F.l1_loss(predicted_ages, targets.float())
            running_mae += mae.item() * inputs.size(0)
            
            total += inputs.size(0)
            
            current_avg_loss = running_loss / total
            current_avg_ranking_loss = running_ranking_loss / total
            current_avg_mae = running_mae / total
            
            pbar.set_description(
                f"Val: loss={current_avg_loss:.4f}, rank_loss={current_avg_ranking_loss:.4f}, mae={current_avg_mae:.2f}"
            )
            pbar.update(1)
    
    pbar.close()
    
    val_loss = running_loss / total
    val_ranking_loss = running_ranking_loss / total
    val_mae = running_mae / total
    
    print(f"val Loss: {val_loss:.4f} Ranking Loss: {val_ranking_loss:.4f} MAE: {val_mae:.4f}")
    
    return val_loss, val_mae

def main():
    
    # Configuration
    config = {
        # Data paths - UPDATE THESE TO YOUR PATHS
        # 'train_csv': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/train_annotations.csv',
        # 'val_csv': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/val_annotations.csv',
        # 'train_dir': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/train',
        # 'val_dir': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/val',


        'train_csv': '/home/meem/filtered/unified_age_dataset/train_annotations.csv',
        'val_csv': '/home/meem/filtered/unified_age_dataset/val_annotations.csv',
        'train_dir': '/home/meem/filtered/unified_age_dataset/train',
        'val_dir': '/home/meem/filtered/unified_age_dataset/val',
        
        # Output paths
        'output': './output_ordinalclip',
        'checkpoint_dir': './checkpoints_ordinalclip',
        
        # Training parameters
        'batch_size': 16,
        'num_epochs': 25,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        
        # OrdinalCLIP specific parameters
        'num_classes': 122,        # Age range 0-121 (UPDATE: was 101, now 122)
        'context_length': 4,       # Number of learnable context tokens
        'rank_embedding_dim': 512, # Dimension of rank embeddings
        'ranking_weight': 0.1,     # Weight for ordinal ranking loss
        'margin': 1.0,             # Margin for ranking loss
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
    
    # Data transforms - CLIP preprocessing
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
    
    # Create OrdinalCLIP model
    model = OrdinalCLIP(
        num_classes=config['num_classes'],
        context_length=config['context_length'],
        rank_embedding_dim=config['rank_embedding_dim']
    )
    model = model.to(device)

    # print(model)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _logger.info(f"Total parameters: {total_params:,}")
    _logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    
    # Loss functions
    criterion = nn.CrossEntropyLoss()
    ranking_loss = OrdinalRankingLoss(
        num_classes=config['num_classes'],
        margin=config['margin']
    )
    
    # Optimizer - only optimize learnable parameters
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['num_epochs'],
        eta_min=1e-6
    )
    
    # Training loop
    best_val_loss = float('inf')
    lr_history = []
    
    for epoch in range(config['num_epochs']):
        # ADD THESE LINES - Apply progressive unfreezing if needed
        new_optimizer, new_scheduler = apply_ordinalclip_unfreezing_strategy(model, epoch, config)
        if new_optimizer is not None:
            optimizer = new_optimizer
        if new_scheduler is not None:
            scheduler = new_scheduler
            
        # ADD THIS LINE - Print parameter breakdown at unfreezing milestones
        if epoch in [0, 8, 15]:
            print_trainable_parameters(model)
        
        # Train (UNCHANGED)
        train_loss, train_mae = train_one_epoch(
            model, train_loader, criterion, ranking_loss, optimizer, device, epoch, config
        )
        
        # Validate (UNCHANGED)
        val_loss, val_mae = validate(
            model, val_loader, criterion, ranking_loss, device, config
        )
        
        # Update learning rate (UNCHANGED)
        scheduler.step()
        lr_history.append(optimizer.param_groups[0]['lr'])
    
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(config['output'], "ordinalclip_best.pth")
            
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
        checkpoint_path = os.path.join(config['checkpoint_dir'], f"ordinalclip_epoch{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_mae': val_mae,
            'config': config,
        }, checkpoint_path)
    
    # Save final model
    final_model_path = os.path.join(config['output'], "ordinalclip_final.pth")
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
    plt.savefig(os.path.join(config['output'], 'ordinalclip_learning_rate_schedule.png'))
    plt.show()

if __name__ == "__main__":
    main()


# In[ ]:


from torchview import draw_graph

# For FaRL MLP

config = {
        # Data paths - UPDATE THESE TO YOUR PATHS
        'train_csv': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/train_annotations.csv',
        'val_csv': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/val_annotations.csv',
        'train_dir': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/train',
        'val_dir': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/val',
        
        # Output paths
        'output': './output_ordinalclip',
        'checkpoint_dir': './checkpoints_ordinalclip',
        
        # Training parameters
        'batch_size': 16,
        'num_epochs': 25,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        
        # OrdinalCLIP specific parameters
        'num_classes': 122,        # Age range 0-121 (UPDATE: was 101, now 122)
        'context_length': 4,       # Number of learnable context tokens
        'rank_embedding_dim': 512, # Dimension of rank embeddings
        'ranking_weight': 0.1,     # Weight for ordinal ranking loss
        'margin': 1.0,             # Margin for ranking loss
    }
    
# Create output directories
os.makedirs(config['output'], exist_ok=True)
os.makedirs(config['checkpoint_dir'], exist_ok=True)

# Set device and seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_logger.info(f"Using device: {device}")
torch.cuda.empty_cache()
torch.cuda.synchronize()

model = OrdinalCLIP(
        num_classes=config['num_classes'],
        context_length=config['context_length'],
        rank_embedding_dim=config['rank_embedding_dim']
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
model_graph.visual_graph.render('./output_ordinalclip/ordinalclip_model_architecture', format='png')

