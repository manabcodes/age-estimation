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


# In[12]:


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


# In[ ]:


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


def create_model(num_classes=122, pretrained=True, model_name='resnet50'):
    """Create and load pre-trained ResNet model"""
    # Create the ResNet model
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Modify the classifier head for age classification
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model


def apply_unfreezing_strategy(model, epoch, config):
    """Apply a more granular unfreezing strategy for ResNet"""
    unfreezing_milestones = {
        0: 'head_only',             # Only fc layer
        5: 'layer4_blocks',         # Unfreeze layer4 blocks gradually
        10: 'layer4_full',          # Entire layer4
        15: 'layer3_blocks',        # Unfreeze layer3 gradually
        20: 'layer3_full',          # Entire layer3
        25: 'layer2_full',          # Entire layer2
        30: 'layer1_full',          # Entire layer1
        35: 'full_model'            # Everything including stem
    }
    
    # Check if we need to update unfreezing strategy
    if epoch in unfreezing_milestones:
        strategy = unfreezing_milestones[epoch]
        print(f"Updating unfreezing strategy to: {strategy}")
        
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
            
        # Apply the appropriate unfreezing strategy
        if strategy == 'head_only':
            # Just the final layer
            for param in model.fc.parameters():
                param.requires_grad = True
                
        elif strategy == 'layer4_blocks':
            # Unfreeze head and last few blocks of layer4
            for param in model.fc.parameters():
                param.requires_grad = True
            # Gradually unfreeze layer4 from back to front (last blocks first)
            for i in range(len(model.layer4) - 1, len(model.layer4) // 2 - 1, -1):
                for param in model.layer4[i].parameters():
                    param.requires_grad = True
        
        elif strategy == 'layer4_full':
            # Unfreeze head and full layer4
            for param in model.fc.parameters():
                param.requires_grad = True
            for param in model.layer4.parameters():
                param.requires_grad = True
        
        elif strategy == 'layer3_blocks':
            # Unfreeze head, layer4, and last few blocks of layer3
            for param in model.fc.parameters():
                param.requires_grad = True
            for param in model.layer4.parameters():
                param.requires_grad = True
            # Gradually unfreeze layer3 from back to front
            for i in range(len(model.layer3) - 1, len(model.layer3) // 2 - 1, -1):
                for param in model.layer3[i].parameters():
                    param.requires_grad = True
                    
        elif strategy == 'layer3_full':
            # Unfreeze head, layer4, and full layer3
            for param in model.fc.parameters():
                param.requires_grad = True
            for param in model.layer4.parameters():
                param.requires_grad = True
            for param in model.layer3.parameters():
                param.requires_grad = True
        
        elif strategy == 'layer2_full':
            # Unfreeze head, layer4, layer3, and layer2
            for param in model.fc.parameters():
                param.requires_grad = True
            for param in model.layer4.parameters():
                param.requires_grad = True
            for param in model.layer3.parameters():
                param.requires_grad = True
            for param in model.layer2.parameters():
                param.requires_grad = True
                
        elif strategy == 'layer1_full':
            # Unfreeze everything except stem and bn1
            for param in model.fc.parameters():
                param.requires_grad = True
            for param in model.layer4.parameters():
                param.requires_grad = True
            for param in model.layer3.parameters():
                param.requires_grad = True
            for param in model.layer2.parameters():
                param.requires_grad = True
            for param in model.layer1.parameters():
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
                    isinstance(criterion, UnimodalConcentratedLossWithWarmup)

    _num_epochs = config['num_epochs']
    # Use tqdm for progress bar
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{_num_epochs} Training', file=sys.stdout)
    # ncols=100)
    
    for batch_idx, (inputs, ages) in enumerate(pbar):
        inputs, ages = inputs.to(device), ages.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
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
    using_uc_loss = isinstance(criterion, UnimodalConcentratedLoss) or \
                   isinstance(criterion, UnimodalConcentratedLossWithWarmup)
    
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc='Validating', file=sys.stdout):
        # ncols=150):
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
    plt.savefig('unimodal_concentrated_resnet.png', dpi=300)
    plt.show()

# Configuration
config = {
    'model': 'resnet50',  # Changed from 'volo_d1' to 'resnet50'
    'train_csv': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/train_annotations.csv',
    'val_csv': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/val_annotations.csv',
    'train_dir': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/train',
    'val_dir': '/home/meem/backup/Age Datasets/UTKFace/crop_part1/val',
    'output': './output_resnet_unimodal',  # Changed output directory
    'checkpoint_dir': './checkpoints_resnet_unimodal',  # Changed checkpoint directory
    'batch_size': 16,  # Increased from 16 to 32 (ResNet can handle larger batches)
    'num_epochs': 50,  # Adjusted for new unfreezing schedule (3 epochs per stage)
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'save_freq': 50  # Save checkpoint every 50 batches
}


def main():
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
    
    # Data transforms - using standard ResNet preprocessing
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
        num_workers=0,  # Increased from 0 for better performance
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,  # Increased from 0 for better performance
        pin_memory=True
    )
    
    # Create model - using ResNet instead of VOLO
    model = create_model(
        num_classes=122,  # 0-121 age range
        pretrained=True,  # Use ImageNet pre-trained weights
        model_name=config['model']
    )
    model = model.to(device)

    complexity_report_path = os.path.join(config['output'], 'unimodal-model_complexity.txt')
    detailed_report_path = os.path.join(config['output'], 'unimodal-model_detailed_analysis.txt')
        
    report_model_complexity(model, output_file=complexity_report_path)
    detailed_model_analysis(model, output_file=detailed_report_path)
    
    # Print model summary
    print(f"Model: {config['model']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    criterion = UnimodalConcentratedLoss(lambda_value=100)  # Reduced lambda from 1000 to 100 for stability
    
    # Apply initial unfreezing strategy (head only)
    optimizer, scheduler = apply_unfreezing_strategy(model, 0, config)
    
    # Initial lr scheduler if not created by unfreezing strategy
    if scheduler is None:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    # Training loop
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    
    # Visualize the distributions
    visualize_unimodal_concentrated()
    
    # Training loop with unimodal-concentrated loss
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
        using_uc_loss = isinstance(criterion, UnimodalConcentratedLoss) or \
                        isinstance(criterion, UnimodalConcentratedLossWithWarmup)
        
        # If using UnimodalConcentratedLossWithWarmup, update the epoch
        if isinstance(criterion, UnimodalConcentratedLossWithWarmup):
            criterion.set_epoch(epoch)
        
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
        val_loss = val_results['loss']
        val_mae = val_results['mae']
        val_within_5 = val_results['within_5']
        print(f"Val Loss: {val_loss:.4f} - MAE: {val_mae:.4f} - Within 5 years: {val_within_5:.2f}%")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint if it's the best model so far (by loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(config['output'], f"{config['model']}_unimodal_best_loss.pth")
            
            # Determine current unfreezing stage
            current_strategy = None
            for milestone, strategy in {
                0: 'head_only', 3: 'layer4_blocks', 6: 'layer4_full', 
                9: 'layer3_blocks', 12: 'layer3_full', 15: 'layer2_full',
                18: 'layer1_full', 21: 'full_model'
            }.items():
                if milestone <= epoch:
                    current_strategy = strategy
            
            # Save checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'unfreezing_strategy': current_strategy,
                'using_uc_loss': using_uc_loss
            }, best_model_path)
            print(f"Saved best model (by loss) with loss {val_loss:.4f}")
        
        # Save checkpoint if it's the best model so far (by MAE)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_path = os.path.join(config['output'], f"{config['model']}_unimodal_best_mae.pth")
            
            # Determine current unfreezing stage
            current_strategy = None
            for milestone, strategy in {
                0: 'head_only', 3: 'layer4_blocks', 6: 'layer4_full', 
                9: 'layer3_blocks', 12: 'layer3_full', 15: 'layer2_full',
                18: 'layer1_full', 21: 'full_model'
            }.items():
                if milestone <= epoch:
                    current_strategy = strategy
            
            # Save checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'unfreezing_strategy': current_strategy,
                'using_uc_loss': using_uc_loss
            }, best_model_path)
            print(f"Saved best model (by MAE) with MAE {val_mae:.4f}")
        
        # Save checkpoint at the end of each epoch
        epoch_checkpoint_path = os.path.join(config['output'], f"{config['model']}_unimodal_epoch{epoch+1}.pth")
        
        # Determine current unfreezing stage
        current_strategy = None
        for milestone, strategy in {
            0: 'head_only', 3: 'layer4_blocks', 6: 'layer4_full', 
            9: 'layer3_blocks', 12: 'layer3_full', 15: 'layer2_full',
            18: 'layer1_full', 21: 'full_model'
        }.items():
            if milestone <= epoch:
                current_strategy = strategy
        
        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_mae': val_mae,
            'unfreezing_strategy': current_strategy,
            'using_uc_loss': using_uc_loss
        }, epoch_checkpoint_path)
        print(f"Saved checkpoint at end of epoch {epoch+1}")
    
    # Save final model
    final_model_path = os.path.join(config['output'], f"{config['model']}_unimodal_concentrated_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'unfreezing_strategy': 'full_model',  # Final model has everything unfrozen
        'config': config,
        'using_uc_loss': isinstance(criterion, UnimodalConcentratedLoss) or \
                        isinstance(criterion, UnimodalConcentratedLossWithWarmup)
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")

    
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
    prof_report_path = os.path.join(config['output'], 'unimodal-profiler_results.txt')
    with open(prof_report_path, 'w') as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total"))
    print(f"Profiling results saved to {prof_report_path}")

if __name__ == "__main__":
    main()

