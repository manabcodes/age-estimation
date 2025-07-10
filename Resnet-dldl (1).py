#!/usr/bin/env python
# coding: utf-8

# In[1]:


lr_history = []


# In[1]:


def report_model_complexity(model, input_size=(1, 3, 224, 224), output_file=None):
    """
    Report model complexity metrics including FLOPs, MACs, and parameters.
    
    Args:
        model: PyTorch model to analyze
        input_size: Input tensor size (batch_size, channels, height, width)
        output_file: Path to save the report (if None, only print to console)
    """
    # Create string buffer to store the report
    import io
    from contextlib import redirect_stdout
    
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        try:
            from ptflops import get_model_complexity_info
            
            macs, params = get_model_complexity_info(
                model, 
                input_size[1:],  # (C, H, W)
                as_strings=True,
                print_per_layer_stat=True,
                verbose=True
            )
            
            print(f"\nModel Complexity Analysis for {model.__class__.__name__}")
            print(f"Input size: {input_size[1:]} (C, H, W)")
            print(f"Computational complexity: {macs}")
            print(f"Number of parameters: {params}")
            
        except ImportError:
            try:
                from thop import profile
                
                # Create a dummy input with the specified size
                device = next(model.parameters()).device
                dummy_input = torch.randn(input_size).to(device)
                
                # Calculate FLOPs and parameters
                flops, params = profile(model, inputs=(dummy_input,))
                
                # Convert to more readable format
                flops_str = f"{flops / 1e9:.2f} GFLOPs"
                params_str = f"{params / 1e6:.2f} M"
                
                print(f"\nModel Complexity Analysis for {model.__class__.__name__}")
                print(f"Input size: {input_size[1:]} (C, H, W)")
                print(f"Computational complexity: {flops_str}")
                print(f"Number of parameters: {params_str}")
                
                # Add detailed per-module analysis
                print("\nPer-module complexity:")
                total_params = sum(p.numel() for p in model.parameters())
                for name, module in model.named_modules():
                    if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                        module_params = sum(p.numel() for p in module.parameters())
                        print(f"{name}: {module_params:,} params ({module_params/total_params*100:.2f}%)")
                
            except ImportError:
                print("\nCould not import ptflops or thop. Please install one of these packages:")
                print("pip install ptflops")
                print("pip install thop")
    
    # Get the report as a string
    report = buffer.getvalue()
    
    # Print to console
    # print(report)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Complexity report saved to {output_file}")
    
    return report

def detailed_model_analysis(model, input_size=(1, 3, 224, 224), output_file=None):
    """
    Provide detailed analysis of model architecture and efficiency.
    Saves the report to a file if output_file is specified.
    """
    # Create string buffer for the report
    import io
    from contextlib import redirect_stdout
    
    buffer = io.StringIO()
    
    with redirect_stdout(buffer):
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_size).to(device)
        
        # 1. Basic model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n==== Model Analysis: {model.__class__.__name__} ====")
        print(f"Date and time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        
        # 2. Memory usage estimation
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer_item in model.buffers():
            buffer_size += buffer_item.nelement() * buffer_item.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        print(f"Model memory footprint: {size_all_mb:.2f} MB")
        
        # 3. Try to measure inference time
        model.eval()  # Set to evaluation mode
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Measure
        torch.cuda.synchronize()
        start_time = time.time()
        iterations = 100
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(dummy_input)
        torch.cuda.synchronize()
        end_time = time.time()
        
        inference_time = (end_time - start_time) / iterations * 1000  # convert to ms
        print(f"Average inference time: {inference_time:.2f} ms (batch size: {input_size[0]})")
        print(f"Throughput: {input_size[0] / (inference_time / 1000):.2f} images/second")
        
        # 4. Module-by-module analysis
        print("\n---- Layer-by-layer Analysis ----")
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                module_params = sum(p.numel() for p in module.parameters())
                print(f"{name}: {module_params:,} params")
                
                # For convolution layers, add more details
                if isinstance(module, nn.Conv2d):
                    in_channels = module.in_channels
                    out_channels = module.out_channels
                    kernel_size = module.kernel_size
                    stride = module.stride
                    padding = module.padding
                    print(f"    in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}")
    
    # Get the report as a string
    report = buffer.getvalue()
    
    # Print to console
    # print(report)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Detailed analysis saved to {output_file}")
    
    return report


# In[3]:


#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
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

# Dual-head ResNet50 model for DLDL-v2
class DualHeadResNet(nn.Module):
    def __init__(self, num_classes=122, pretrained=True):
        super(DualHeadResNet, self).__init__()
        
        # Create the base ResNet50 model
        self.base_model = models.resnet50(pretrained=pretrained)
        
        # Get the feature dimension
        in_features = self.base_model.fc.in_features
        
        # Replace the classification layer with Identity
        self.base_model.fc = nn.Identity()
        
        # 1. Distribution head for label distribution learning
        self.distribution_head = nn.Linear(in_features, num_classes)
        
        # 2. Regression head for direct age prediction
        self.regression_head = nn.Linear(in_features, 1)
    
    def forward(self, x):
        # Get features from the base model
        features = self.base_model(x)
        
        # Forward through both heads
        distribution_logits = self.distribution_head(features)
        regression_output = self.regression_head(features)
        
        return distribution_logits, regression_output

def create_model(num_classes=122, pretrained=True):
    """Create and load dual-head ResNet50 model for DLDL-v2"""
    model = DualHeadResNet(num_classes=num_classes, pretrained=pretrained)
    return model

def apply_unfreezing_strategy(model, epoch, config):
    """Apply a more granular unfreezing strategy for ResNet"""
    unfreezing_milestones = {
        0: 'head_only',             # Only distribution_head and regression_head
        3: 'layer4_blocks',         # Unfreeze layer4 blocks gradually
        6: 'layer4_full',           # Entire layer4
        9: 'layer3_blocks',         # Unfreeze layer3 gradually
        12: 'layer3_full',          # Entire layer3
        15: 'layer2_full',          # Entire layer2
        18: 'layer1_full',          # Entire layer1
        21: 'full_model'            # Everything including stem
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
                
        elif strategy == 'layer4_blocks':
            # Gradually unfreeze layer4 from back to front (last blocks first)
            for i in range(len(model.base_model.layer4) - 1, len(model.base_model.layer4) // 2 - 1, -1):
                for param in model.base_model.layer4[i].parameters():
                    param.requires_grad = True
        
        elif strategy == 'layer4_full':
            # Unfreeze full layer4
            for param in model.base_model.layer4.parameters():
                param.requires_grad = True
        
        elif strategy == 'layer3_blocks':
            # Keep layer4 unfrozen and gradually unfreeze layer3 from back to front
            for param in model.base_model.layer4.parameters():
                param.requires_grad = True
            # Gradually unfreeze layer3 from back to front
            for i in range(len(model.base_model.layer3) - 1, len(model.base_model.layer3) // 2 - 1, -1):
                for param in model.base_model.layer3[i].parameters():
                    param.requires_grad = True
                    
        elif strategy == 'layer3_full':
            # Unfreeze layer4 and full layer3
            for param in model.base_model.layer4.parameters():
                param.requires_grad = True
            for param in model.base_model.layer3.parameters():
                param.requires_grad = True
        
        elif strategy == 'layer2_full':
            # Unfreeze layer4, layer3, and layer2
            for param in model.base_model.layer4.parameters():
                param.requires_grad = True
            for param in model.base_model.layer3.parameters():
                param.requires_grad = True
            for param in model.base_model.layer2.parameters():
                param.requires_grad = True
                
        elif strategy == 'layer1_full':
            # Unfreeze everything except stem and bn1
            for param in model.base_model.layer4.parameters():
                param.requires_grad = True
            for param in model.base_model.layer3.parameters():
                param.requires_grad = True
            for param in model.base_model.layer2.parameters():
                param.requires_grad = True
            for param in model.base_model.layer1.parameters():
                param.requires_grad = True
        
        elif strategy == 'full_model':
            # Unfreeze everything
            for param in model.parameters():
                param.requires_grad = True
        
        # Adjust learning rate based on unfreezing stage
        # Deeper unfreezing should use a smaller learning rate
        stage_divisor = {
            'head_only': 1,
            'layer4_blocks': 2,
            'layer4_full': 4,
            'layer3_blocks': 8,
            'layer3_full': 10,
            'layer2_full': 12,
            'layer1_full': 16,
            'full_model': 20
        }
        
        # Create a new optimizer with an adjusted learning rate
        lr = config['lr'] / stage_divisor[strategy]
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
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
        print(f'Learning rate: {lr:.6f}')
        
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
    pbar = tqdm(total=len(dataloader), desc="Train", file=sys.stdout)
    
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

def validate_dldlv2(model, dataloader, criterion, device, optimizer):
    model.eval()
    running_loss = 0.0
    running_kl_loss = 0.0
    running_l1_loss = 0.0
    running_distr_mae = 0.0
    running_reg_mae = 0.0
    total = 0
    start_time = time.time()
    
    # Create a progress bar using tqdm
    pbar = tqdm(total=len(dataloader), desc="Val", file=sys.stdout)
    
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

    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)
    
    # Return the best MAE (minimum of distribution-based and regression-based)
    best_mae = min(val_distr_mae, val_reg_mae)
    return val_loss, best_mae

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
    'train_csv': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/train_annotations.csv',
    'val_csv': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/val_annotations.csv',
    'train_dir': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/train',
    'val_dir': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/val',          # Update with your path
    'output': './output',
    'checkpoint_dir': './checkpoints_resnet50_dldlv2',
    'batch_size': 16,                              # Increased batch size for ResNet50
    'num_epochs': 30,                              # Increased to 30 epochs
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'save_freq': 50,                               # Save checkpoint every 50 batches
    'lambda_val': 1.0,                             # Weight between KL and L1 loss
    'sigma': 2.0,                                  # Sigma for the Gaussian label distributions
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
    
    # Create model - now using the dual-head ResNet50 architecture
    model = create_model(
        num_classes=122,  # 0-121 age range
        pretrained=True
    )
    model = model.to(device)

    complexity_report_path = os.path.join(config['output'], 'dldl-model_complexity.txt')
    detailed_report_path = os.path.join(config['output'], 'dldl-model_detailed_analysis.txt')
    
    report_model_complexity(model, output_file=complexity_report_path)
    detailed_model_analysis(model, output_file=detailed_report_path)
    
    # Print model structure
    # print(model)
    
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
        val_loss, val_mae = validate_dldlv2(model, val_loader, criterion, device, optimizer)
        
        # Track validation metrics
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint if it's the best model so far
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_loss = val_loss
            best_model_path = os.path.join(config['output'], "resnet50_dldlv2_best.pth")
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
        epoch_checkpoint_path = os.path.join(config['output'], f"resnet50_dldlv2_epoch{epoch+1}.pth")
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
        plt.savefig(os.path.join(config['output'], 'dldl-training_progress.png'))
        plt.close()
    
    # Save final model
    final_model_path = os.path.join(config['output'], "resnet50_dldlv2_final.pth")
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

    # from torchinfo import summary
    # summary_report_path = os.path.join(config['output'], 'model_summary.txt')
    # with open(summary_report_path, 'w') as f:
    #     with redirect_stdout(f):
    #         summary(model, input_size=input_size)

    # At the start of your script


    # When you want to profile:
    # PROFILING

    # Ensure model is in evaluation mode
    model.eval()
    
    # Clear cache for consistent measurements
    torch.cuda.empty_cache()
    
    # Create a representative input
    dummy_input = torch.randn((1, 3, 224, 224), device=device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Profile
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with record_function("model_inference"):
                _ = model(dummy_input)
    
    # Save the profile results
    prof_report_path = os.path.join(config['output'], 'dldl-profiler_results.txt')
    with open(prof_report_path, 'w') as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total"))
    print(f"Profiling results saved to {prof_report_path}")


# In[ ]:


# After training completes, plot the learning rate schedule
plt.figure(figsize=(10, 5))
plt.plot(lr_history)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.yscale('log')  # Log scale makes the drops more visible
plt.grid(True)
plt.savefig(os.path.join(config['output'], 'dldl-learning_rate_schedule.png'))
plt.show()


# In[2]:


get_ipython().getoutput('nvidia-smi')

