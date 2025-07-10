#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import random
from datetime import datetime
import clip
import warnings


# In[2]:


# Generate timestamp for file naming
timestamp = datetime.now().strftime("-%Y-%m-%d_%H-%M-%S")

# Dataset class for age estimation
class AgeEstimationDataset(Dataset):
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
            
        return image, float(age), self.annotations.iloc[idx, 0]  # Return filename for visualization

# POE-FaRL Model Definition (copied from the POE-FaRL implementation)
class POEFaRLModel(nn.Module):
    """FaRL backbone with POE (Probabilistic Ordinal Embeddings) for age estimation"""
    
    def __init__(self, farl_model_path=None, embedding_dim=512, 
                 dropout_rate=0.1, freeze_backbone=True, max_t=50):
        super().__init__()
        
        # Load CLIP ViT-B/16 architecture
        self.clip_model, self.preprocess = clip.load("ViT-B/16", device="cpu")
        
        # Load FaRL weights if provided
        if farl_model_path and os.path.exists(farl_model_path):
            print(f"Loading FaRL weights from {farl_model_path}")
            farl_state = torch.load(farl_model_path, map_location="cpu")
            self.clip_model.load_state_dict(farl_state["state_dict"], strict=False)
        else:
            print("No FaRL weights provided, using CLIP weights only")
        
        # Extract the visual encoder (backbone)
        self.backbone = self.clip_model.visual
        
        # Get the actual feature dimension from the backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_features = self.backbone(dummy_input)
            backbone_dim = dummy_features.shape[-1]
            
        print(f"Detected backbone feature dimension: {backbone_dim}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("FaRL backbone frozen")
        
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
            # Deterministic forward pass (inference)
            age_pred = self.regression_head(mu)
            return age_pred.squeeze(), mu, log_var
    
    def predict_with_uncertainty(self, x, num_samples=100):
        """Generate predictions with uncertainty estimates"""
        self.eval()
        original_max_t = self.max_t
        self.max_t = num_samples
        
        with torch.no_grad():
            age_pred, mu, log_var = self.forward(x, use_sto=True, training=True)
            
            # Calculate mean prediction and uncertainty
            mean_pred = torch.mean(age_pred, dim=0)  # [batch]
            uncertainty = torch.var(age_pred, dim=0)  # [batch] - predictive variance
            
            # Also get aleatoric uncertainty from the model
            aleatoric_uncertainty = torch.mean(torch.exp(log_var), dim=1)  # [batch]
            
        self.max_t = original_max_t
        return mean_pred, uncertainty, aleatoric_uncertainty

# Create POE-FaRL model
def create_poe_farl_model(farl_model_path=None, embedding_dim=512, dropout_rate=0.1, max_t=50):
    model = POEFaRLModel(
        farl_model_path=farl_model_path,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
        freeze_backbone=True,
        max_t=max_t
    )
    return model

# Function to create scatter plot with uncertainty visualization
def create_poe_age_prediction_scatter(predictions, targets, uncertainties, results_dir):
    """Create a scatter plot showing predictions vs true ages with uncertainty"""
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with uncertainty as color
    scatter = plt.scatter(targets, predictions, c=uncertainties, cmap='viridis', 
                         alpha=0.6, s=20, edgecolor='none')
    
    # Add colorbar for uncertainty
    cbar = plt.colorbar(scatter)
    cbar.set_label('Predictive Uncertainty (variance)', fontsize=12)
    
    # Add reference lines
    max_val = max(np.max(targets), np.max(predictions))
    min_val = min(np.min(targets), np.min(predictions))
    
    # Perfect prediction line
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    # Linear regression line
    model = LinearRegression()
    X = targets.reshape(-1, 1)
    y = predictions
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    plt.plot([min_val, max_val], 
             [min_val * slope + intercept, max_val * slope + intercept], 
             'g-', linewidth=2, 
             label=f'Regression line (slope={slope:.2f})')
    
    # Error margin lines
    plt.plot([min_val, max_val], [min_val + 5, max_val + 5], 'k:', linewidth=1.5, label='+5 years')
    plt.plot([min_val, max_val], [min_val - 5, max_val - 5], 'k:', linewidth=1.5, label='-5 years')
    
    # Add statistics
    plt.text(0.02, 0.95, f"Total samples: {len(predictions)}", fontsize=10, transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8))
    
    r_squared = model.score(X, y)
    plt.text(0.02, 0.90, f"R² = {r_squared:.3f}", fontsize=10, transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8))
    
    mean_uncertainty = np.mean(uncertainties)
    plt.text(0.02, 0.85, f"Mean Uncertainty = {mean_uncertainty:.2f}", fontsize=10, transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Style the plot
    plt.grid(True, alpha=0.3)
    plt.xlabel('True Age', fontsize=14)
    plt.ylabel('Predicted Age', fontsize=14)
    plt.title('POE-FaRL Age Prediction Results with Uncertainty', fontsize=16)
    plt.legend(loc='upper left', fontsize=12)
    
    # Make sure we show the full data range
    buffer = 5
    plt.xlim(min_val - buffer, max_val + buffer)
    plt.ylim(min_val - buffer, max_val + buffer)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'poe_farl_age_prediction_scatter.png'), dpi=300)
    plt.close()

# Function to visualize uncertainty calibration
def plot_uncertainty_calibration(predictions, targets, uncertainties, results_dir):
    """Plot uncertainty calibration - how well uncertainty correlates with error"""
    errors = np.abs(predictions - targets)
    
    # Bin uncertainties and calculate average error in each bin
    n_bins = 10
    uncertainty_bins = np.linspace(np.min(uncertainties), np.max(uncertainties), n_bins + 1)
    bin_centers = (uncertainty_bins[:-1] + uncertainty_bins[1:]) / 2
    
    avg_errors = []
    avg_uncertainties = []
    
    for i in range(n_bins):
        mask = (uncertainties >= uncertainty_bins[i]) & (uncertainties < uncertainty_bins[i + 1])
        if np.sum(mask) > 0:
            avg_errors.append(np.mean(errors[mask]))
            avg_uncertainties.append(np.mean(uncertainties[mask]))
    
    plt.figure(figsize=(10, 6))
    plt.scatter(avg_uncertainties, avg_errors, s=100, alpha=0.7)
    
    # Add diagonal line for perfect calibration
    min_val = min(min(avg_uncertainties), min(avg_errors))
    max_val = max(max(avg_uncertainties), max(avg_errors))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect calibration')
    
    # Calculate correlation
    if len(avg_uncertainties) > 1:
        correlation = np.corrcoef(avg_uncertainties, avg_errors)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('Average Predicted Uncertainty', fontsize=14)
    plt.ylabel('Average Absolute Error', fontsize=14)
    plt.title('Uncertainty Calibration - POE-FaRL', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'poe_farl_uncertainty_calibration.png'), dpi=300)
    plt.close()

# Helper function to visualize regression predictions with uncertainty
def visualize_poe_predictions(filenames, true_ages, pred_ages, uncertainties, img_dir, output_dir):
    """Visualize predictions with uncertainty for a few samples"""
    n_samples = len(filenames)
    n_cols = min(5, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    # Limit the number of rows to 3 (15 samples max)
    n_rows = min(n_rows, 3)
    n_samples = min(n_samples, n_cols * n_rows)
    
    plt.figure(figsize=(n_cols * 2.5, n_rows * 2.5))
    
    for i, (filename, true_age, pred_age, uncertainty) in enumerate(zip(
        filenames[:n_samples], true_ages[:n_samples], pred_ages[:n_samples], uncertainties[:n_samples])):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Load and display the image
        img_path = os.path.join(img_dir, filename)
        img = Image.open(img_path).convert('RGB')
        plt.imshow(img)
        
        # Add true and predicted ages with uncertainty
        error = abs(true_age - pred_age)
        within_5 = "✓" if error <= 5 else "✗"
        color = 'green' if error <= 5 else 'red'
        
        # Include uncertainty in the title
        plt.title(f"True: {true_age:.1f}\nPred: {pred_age:.1f} ± {np.sqrt(uncertainty):.1f}\n{within_5}", 
                 color=color, fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'poe_farl_sample_predictions.png'))
    plt.close()

# Main test function for POE-FaRL
def test_poe_farl_model(model_path, test_csv, test_dir, output_dir='./poe_farl_test_results', 
                        farl_weights_path=None, num_samples_to_visualize=15, 
                        config=None, uncertainty_samples=100):
    """
    Test the POE-FaRL age estimation model and generate comprehensive visualizations
    
    Args:
        model_path: Path to the trained POE-FaRL checkpoint
        test_csv: Path to the CSV file with test annotations
        test_dir: Path to the directory containing test images
        output_dir: Directory to save results and visualizations
        farl_weights_path: Path to FaRL backbone weights (optional)
        num_samples_to_visualize: Number of random samples to visualize
        config: Model configuration dictionary (optional)
        uncertainty_samples: Number of samples for uncertainty estimation
        
    Returns:
        Dictionary with evaluation metrics and results
    """
    # Create output directory
    results_dir = f"{output_dir}{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model configuration if available
    if config is None:
        # Default configuration - should match training config
        config = {
            'embedding_dim': 512,
            'dropout_rate': 0.1,
            'max_t': 50
        }
    
    # Load model
    print(f"Creating POE-FaRL model...")
    model = create_poe_farl_model(
        farl_model_path=farl_weights_path,
        embedding_dim=config['embedding_dim'],
        dropout_rate=config['dropout_rate'],
        max_t=config['max_t']
    )
    
    print(f"Loading model weights from: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model from state_dict key")
            # Load config if available in checkpoint
            if 'config' in checkpoint:
                saved_config = checkpoint['config']
                print(f"Loaded config from checkpoint: {saved_config}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model from direct state_dict")
            
        model = model.to(device)
        model.eval()
        print("Model loaded successfully and set to evaluation mode")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    # Test transforms - CLIP preprocessing
    print("Setting up CLIP data preprocessing transforms...")
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                           std=[0.26862954, 0.26130258, 0.27577711]),  # CLIP normalization
    ])
    
    # Create dataset and dataloader
    print(f"Loading test dataset from {test_csv} with images in {test_dir}")
    try:
        test_dataset = AgeEstimationDataset(
            csv_file=test_csv,
            img_dir=test_dir,
            transform=test_transform
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=16, 
            shuffle=False, 
            num_workers=0,
            pin_memory=True
        )
        
        print(f"Test dataset loaded successfully with {len(test_dataset)} images")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Evaluation loop
    all_preds = []
    all_targets = []
    all_filenames = []
    all_errors = []
    all_uncertainties = []
    all_aleatoric_uncertainties = []
    
    criterion = nn.MSELoss()
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, ages, filenames in tqdm(test_loader, desc="Testing POE-FaRL"):
            inputs, targets = inputs.to(device), ages.to(device)
            
            # Get predictions with uncertainty
            mean_preds, uncertainties, aleatoric_uncertainties = model.predict_with_uncertainty(
                inputs, num_samples=uncertainty_samples
            )
            
            # Calculate loss using mean predictions
            loss = criterion(mean_preds, targets)
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate MAE for each sample
            mae = torch.abs(mean_preds - targets)
            
            # Save results
            all_preds.extend(mean_preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_filenames.extend(filenames)
            all_errors.extend(mae.cpu().numpy())
            all_uncertainties.extend(uncertainties.cpu().numpy())
            all_aleatoric_uncertainties.extend(aleatoric_uncertainties.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(all_preds)
    targets = np.array(all_targets)
    errors = np.array(all_errors)
    uncertainties = np.array(all_uncertainties)
    aleatoric_uncertainties = np.array(all_aleatoric_uncertainties)
    
    print("Testing completed successfully!")
    
    # Calculate metrics
    avg_loss = running_loss / len(test_loader.dataset)
    avg_mae = np.mean(errors)
    rmse = np.sqrt(np.mean((predictions - targets)**2))
    
    # Calculate accuracy within different thresholds
    within_1_year = np.mean(errors <= 1) * 100
    within_3_years = np.mean(errors <= 3) * 100
    within_5_years = np.mean(errors <= 5) * 100
    within_10_years = np.mean(errors <= 10) * 100
    
    # Calculate correlation
    correlation = np.corrcoef(targets, predictions)[0, 1]
    
    # Uncertainty metrics
    avg_uncertainty = np.mean(uncertainties)
    avg_aleatoric_uncertainty = np.mean(aleatoric_uncertainties)
    uncertainty_correlation = np.corrcoef(errors, uncertainties)[0, 1]
    
    # Print overall metrics
    print("\n=== POE-FARL TEST RESULTS ===")
    print(f"Average MSE Loss: {avg_loss:.4f}")
    print(f"Root Mean Square Error: {rmse:.2f} years")
    print(f"Mean Absolute Error: {avg_mae:.2f} years")
    print(f"Correlation (r): {correlation:.4f}")
    print(f"R²: {correlation**2:.4f}")
    print(f"Within 1 year: {within_1_year:.2f}%")
    print(f"Within 3 years: {within_3_years:.2f}%")
    print(f"Within 5 years: {within_5_years:.2f}%")
    print(f"Within 10 years: {within_10_years:.2f}%")
    print(f"\n=== UNCERTAINTY METRICS ===")
    print(f"Average Predictive Uncertainty: {avg_uncertainty:.2f}")
    print(f"Average Aleatoric Uncertainty: {avg_aleatoric_uncertainty:.2f}")
    print(f"Error-Uncertainty Correlation: {uncertainty_correlation:.4f}")
    
    # Calculate metrics by age group
    age_groups = [
        (0, 12, "Children (0-12)"),
        (13, 19, "Teenagers (13-19)"),
        (20, 35, "Young Adults (20-35)"),
        (36, 50, "Middle-aged (36-50)"),
        (51, 70, "Seniors (51-70)"),
        (71, 100, "Elderly (71+)")
    ]
    
    print("\n=== RESULTS BY AGE GROUP ===")
    
    # Store age group results for later visualization
    age_group_labels = []
    within_5_years_by_group = []
    mae_by_group = []
    uncertainty_by_group = []
    
    for start, end, name in age_groups:
        mask = (targets >= start) & (targets <= end)
        if np.sum(mask) > 0:
            group_mae = np.mean(errors[mask])
            group_rmse = np.sqrt(np.mean((predictions[mask] - targets[mask])**2))
            group_within_5 = np.mean(errors[mask] <= 5) * 100
            group_corr = np.corrcoef(targets[mask], predictions[mask])[0, 1] if np.sum(mask) > 1 else 0
            group_uncertainty = np.mean(uncertainties[mask])
            count = np.sum(mask)
            
            # Store for visualization
            age_group_labels.append(name)
            within_5_years_by_group.append(group_within_5)
            mae_by_group.append(group_mae)
            uncertainty_by_group.append(group_uncertainty)
            
            print(f"{name}: MAE: {group_mae:.2f} years, RMSE: {group_rmse:.2f} years, "
                  f"Within 5 years: {group_within_5:.2f}%, Uncertainty: {group_uncertainty:.2f}, "
                  f"r: {group_corr:.3f} (n={count})")
    
    # Create visualizations
    
    # 1. Age Prediction Scatter Plot with Uncertainty
    create_poe_age_prediction_scatter(predictions, targets, uncertainties, results_dir)
    
    # 2. Uncertainty Calibration Plot
    plot_uncertainty_calibration(predictions, targets, uncertainties, results_dir)
    
    # 3. Uncertainty vs Error Scatter Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(uncertainties, errors, alpha=0.5, s=20)
    plt.xlabel('Predictive Uncertainty', fontsize=14)
    plt.ylabel('Absolute Error (years)', fontsize=14)
    plt.title('Predictive Uncertainty vs Absolute Error - POE-FaRL', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Add correlation text
    plt.text(0.05, 0.95, f'Correlation: {uncertainty_correlation:.3f}', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'poe_farl_uncertainty_vs_error.png'), dpi=300)
    plt.close()
    
    # 4. Residuals Plot
    plt.figure(figsize=(10, 6))
    residuals = predictions - targets
    plt.scatter(targets, residuals, alpha=0.5, s=20)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.axhline(y=5, color='orange', linestyle=':', linewidth=1, label='+5 years')
    plt.axhline(y=-5, color='orange', linestyle=':', linewidth=1, label='-5 years')
    plt.xlabel('True Age', fontsize=14)
    plt.ylabel('Residuals (Predicted - True)', fontsize=14)
    plt.title('Residuals Plot - POE-FaRL Age Estimation', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'poe_farl_residuals_plot.png'), dpi=300)
    plt.close()
    
    # 5. Average Error by Age
    plt.figure(figsize=(12, 6))
    
    # Calculate average error for each age (bin by 2-year intervals)
    age_bins = np.arange(0, 101, 2)
    avg_errors_binned = []
    avg_uncertainties_binned = []
    age_centers = []
    
    for i in range(len(age_bins)-1):
        mask = (targets >= age_bins[i]) & (targets < age_bins[i+1])
        if np.sum(mask) > 0:
            avg_errors_binned.append(np.mean(errors[mask]))
            avg_uncertainties_binned.append(np.mean(uncertainties[mask]))
            age_centers.append((age_bins[i] + age_bins[i+1]) / 2)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot average error
    color = 'tab:red'
    ax1.set_xlabel('True Age', fontsize=14)
    ax1.set_ylabel('Average Error (years)', color=color, fontsize=14)
    line1 = ax1.plot(age_centers, avg_errors_binned, 'o-', color=color, linewidth=2, markersize=4, label='MAE')
    ax1.axhline(y=avg_mae, color=color, linestyle='--', alpha=0.7, label=f'Overall MAE: {avg_mae:.2f}')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Plot average uncertainty on secondary y-axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Average Uncertainty', color=color, fontsize=14)
    line2 = ax2.plot(age_centers, avg_uncertainties_binned, 's-', color=color, linewidth=2, markersize=4, label='Uncertainty')
    ax2.axhline(y=avg_uncertainty, color=color, linestyle='--', alpha=0.7, label=f'Overall Uncertainty: {avg_uncertainty:.2f}')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Average Error and Uncertainty by Age (2-year bins)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'poe_farl_avg_error_uncertainty_by_age.png'), dpi=300)
    plt.close()
    
    # 6. Cumulative Error Distribution
    plt.figure(figsize=(10, 6))
    error_thresholds = np.arange(0, 21)  # 0 to 20 years
    cumulative_pct = [np.mean(errors <= threshold) * 100 for threshold in error_thresholds]
    
    plt.plot(error_thresholds, cumulative_pct, marker='o', markersize=6, linewidth=2)
    plt.axhline(y=90, color='r', linestyle='--', alpha=0.7, label='90% threshold')
    plt.axvline(x=5, color='g', linestyle='--', alpha=0.7, label='5-year threshold')
    
    # Mark the percentage of predictions within 5 years
    plt.plot(5, within_5_years, 'ro', markersize=8)
    plt.annotate(f"{within_5_years:.1f}%", 
                 xy=(5, within_5_years), 
                 xytext=(7, within_5_years - 5),
                 arrowprops=dict(arrowstyle="->", color='r'))
    
    plt.xlabel('Error Threshold (years)', fontsize=14)
    plt.ylabel('Percentage of Predictions (%)', fontsize=14)
    plt.title('Cumulative Error Distribution - POE-FaRL', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'poe_farl_cumulative_error_distribution.png'), dpi=300)
    plt.close()
    
    # 7. Error Distribution Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=avg_mae, color='r', linestyle='--', linewidth=2, label=f'Mean MAE: {avg_mae:.2f}')
    plt.axvline(x=np.median(errors), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.2f}')
    plt.xlabel('Absolute Error (years)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Distribution of Prediction Errors - POE-FaRL', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'poe_farl_error_distribution.png'), dpi=300)
    plt.close()
    
    # 8. Age Group Performance Comparison
    plt.figure(figsize=(15, 10))
    
    # Create subplots for MAE, Within 5 years, and Uncertainty
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # MAE by age group
    bars1 = ax1.bar(range(len(age_group_labels)), mae_by_group, color='#3498db', alpha=0.7)
    ax1.axhline(y=avg_mae, color='r', linestyle='--', alpha=0.7, label=f'Overall MAE: {avg_mae:.2f}')
    ax1.set_xticks(range(len(age_group_labels)))
    ax1.set_xticklabels(age_group_labels, rotation=45, ha='right')
    ax1.set_ylabel('Mean Absolute Error (years)')
    ax1.set_title('MAE by Age Group - POE-FaRL')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add data labels
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{mae_by_group[i]:.1f}',
                ha='center', va='bottom')
    
    # Within 5 years by age group
    bars2 = ax2.bar(range(len(age_group_labels)), within_5_years_by_group, color='#2ecc71', alpha=0.7)
    ax2.axhline(y=within_5_years, color='r', linestyle='--', alpha=0.7, label=f'Overall: {within_5_years:.1f}%')
    ax2.set_xticks(range(len(age_group_labels)))
    ax2.set_xticklabels(age_group_labels, rotation=45, ha='right')
    ax2.set_ylabel('Percentage within 5 years (%)')
    ax2.set_title('Accuracy within 5 Years by Age Group - POE-FaRL')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add data labels
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{within_5_years_by_group[i]:.1f}%',
                ha='center', va='bottom')
    
    # Uncertainty by age group
    bars3 = ax3.bar(range(len(age_group_labels)), uncertainty_by_group, color='#e74c3c', alpha=0.7)
    ax3.axhline(y=avg_uncertainty, color='r', linestyle='--', alpha=0.7, label=f'Overall: {avg_uncertainty:.2f}')
    ax3.set_xticks(range(len(age_group_labels)))
    ax3.set_xticklabels(age_group_labels, rotation=45, ha='right')
    ax3.set_ylabel('Average Uncertainty')
    ax3.set_title('Uncertainty by Age Group - POE-FaRL')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Add data labels
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{uncertainty_by_group[i]:.2f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'poe_farl_age_group_performance.png'), dpi=300)
    plt.close()
    
    # 9. Uncertainty Distribution Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(uncertainties, bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(x=avg_uncertainty, color='r', linestyle='--', linewidth=2, label=f'Mean: {avg_uncertainty:.2f}')
    plt.axvline(x=np.median(uncertainties), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(uncertainties):.2f}')
    plt.xlabel('Predictive Uncertainty', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Distribution of Predictive Uncertainties - POE-FaRL', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'poe_farl_uncertainty_distribution.png'), dpi=300)
    plt.close()
    
    # 10. Visualize sample predictions with uncertainty
    if num_samples_to_visualize > 0:
        # Sample random indices, preferring high uncertainty samples for visualization
        uncertainty_indices = np.argsort(uncertainties)[::-1]  # Sort by uncertainty (high to low)
        high_uncertainty_indices = uncertainty_indices[:min(num_samples_to_visualize//2, len(uncertainty_indices))]
        
        # Also sample some random indices
        remaining_samples = num_samples_to_visualize - len(high_uncertainty_indices)
        if remaining_samples > 0:
            random_indices = np.random.choice(
                [i for i in range(len(all_preds)) if i not in high_uncertainty_indices], 
                min(remaining_samples, len(all_preds) - len(high_uncertainty_indices)), 
                replace=False
            )
            selected_indices = np.concatenate([high_uncertainty_indices, random_indices])
        else:
            selected_indices = high_uncertainty_indices
        
        visualize_poe_predictions(
            [all_filenames[i] for i in selected_indices],
            [targets[i] for i in selected_indices],
            [predictions[i] for i in selected_indices],
            [uncertainties[i] for i in selected_indices],
            test_loader.dataset.img_dir,
            results_dir
        )
    
    # Save detailed results to CSV
    results_df = pd.DataFrame({
        'filename': all_filenames,
        'true_age': targets,
        'predicted_age': predictions,
        'absolute_error': errors,
        'predictive_uncertainty': uncertainties,
        'aleatoric_uncertainty': aleatoric_uncertainties,
        'within_5_years': errors <= 5
    })
    
    results_df.to_csv(os.path.join(results_dir, 'poe_farl_results.csv'), index=False)
    print(f"\nSaved detailed results to {results_dir}/poe_farl_results.csv")
    
    # Write summary to text file
    with open(os.path.join(results_dir, 'poe_farl_summary.txt'), 'w') as f:
        f.write("=== POE-FARL MODEL EVALUATION SUMMARY ===\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"FaRL weights: {farl_weights_path or 'CLIP weights only'}\n")
        f.write(f"Test dataset: {test_csv}\n")
        f.write(f"Number of test samples: {len(test_dataset)}\n")
        f.write(f"Uncertainty samples: {uncertainty_samples}\n\n")
        
        f.write("=== OVERALL METRICS ===\n")
        f.write(f"Mean Absolute Error: {avg_mae:.2f} years\n")
        f.write(f"Root Mean Square Error: {rmse:.2f} years\n")
        f.write(f"Correlation (r): {correlation:.4f}\n")
        f.write(f"R-squared (r²): {correlation**2:.4f}\n")
        f.write(f"Average MSE Loss: {avg_loss:.4f}\n\n")
        
        f.write("=== ERROR THRESHOLDS ===\n")
        f.write(f"Within 1 year: {within_1_year:.2f}%\n")
        f.write(f"Within 3 years: {within_3_years:.2f}%\n")
        f.write(f"Within 5 years: {within_5_years:.2f}%\n")
        f.write(f"Within 10 years: {within_10_years:.2f}%\n\n")
        
        f.write("=== UNCERTAINTY METRICS ===\n")
        f.write(f"Average Predictive Uncertainty: {avg_uncertainty:.2f}\n")
        f.write(f"Average Aleatoric Uncertainty: {avg_aleatoric_uncertainty:.2f}\n")
        f.write(f"Error-Uncertainty Correlation: {uncertainty_correlation:.4f}\n\n")
        
        # Find error threshold for 90% accuracy
        error_thresholds = np.arange(0, 21)
        cumulative_pct = [np.mean(errors <= threshold) * 100 for threshold in error_thresholds]
        threshold_90pct = next((i for i, pct in enumerate(cumulative_pct) if pct >= 90), None)
        if threshold_90pct is not None:
            f.write(f"Error threshold for 90% accuracy: {threshold_90pct} years\n\n")
        
        # Calculate bias
        avg_bias = np.mean(predictions - targets)
        f.write(f"Average bias (predicted - true): {avg_bias:.2f} years\n\n")
        
        f.write("=== RESULTS BY AGE GROUP ===\n")
        for i, (start, end, name) in enumerate(age_groups):
            if i < len(mae_by_group):
                mask = (targets >= start) & (targets <= end)
                count = np.sum(mask)
                f.write(f"{name}:\n")
                f.write(f"  - Sample count: {count}\n")
                f.write(f"  - MAE: {mae_by_group[i]:.2f} years\n")
                f.write(f"  - Within 5 years: {within_5_years_by_group[i]:.2f}%\n")
                f.write(f"  - Average Uncertainty: {uncertainty_by_group[i]:.2f}\n\n")
        
        # Find worst predictions
        worst_indices = np.argsort(errors)[-10:][::-1]
        f.write("=== WORST PREDICTIONS ===\n")
        for idx in worst_indices:
            f.write(f"File: {all_filenames[idx]}, True: {targets[idx]:.1f}, "
                   f"Predicted: {predictions[idx]:.1f}, Error: {errors[idx]:.1f} years, "
                   f"Uncertainty: {uncertainties[idx]:.2f}\n")
        
        # Find highest uncertainty predictions
        highest_uncertainty_indices = np.argsort(uncertainties)[-10:][::-1]
        f.write("\n=== HIGHEST UNCERTAINTY PREDICTIONS ===\n")
        for idx in highest_uncertainty_indices:
            f.write(f"File: {all_filenames[idx]}, True: {targets[idx]:.1f}, "
                   f"Predicted: {predictions[idx]:.1f}, Error: {errors[idx]:.1f} years, "
                   f"Uncertainty: {uncertainties[idx]:.2f}\n")
        
        f.write(f"\n=== TEST ENVIRONMENT ===\n")
        f.write(f"Test date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n")
        if torch.cuda.is_available():
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    return {
        'mae': avg_mae,
        'rmse': rmse,
        'correlation': correlation,
        'r_squared': correlation**2,
        'within_5_years': within_5_years,
        'avg_uncertainty': avg_uncertainty,
        'uncertainty_correlation': uncertainty_correlation,
        'results_dir': results_dir
    }


# In[3]:


if __name__ == "__main__":
    """Main execution function"""
    import argparse
    
    # Handle Jupyter notebook vs script execution
    if any(arg.startswith('-f') for arg in sys.argv):
        # Jupyter mode - use default arguments
        class Args:
            model_path = './output_poe_farl/poe_farl_best.pth'  # Update this path
            test_csv = '/home/meem/backup/Age Datasets/test_annotations.csv'
            test_dir = '/home/meem/backup/Age Datasets/test'
            output_dir = './poe_farl_test_results'
            farl_weights = None  # Path to FaRL backbone weights (optional)
            num_samples = 15
            uncertainty_samples = 100
            embedding_dim = 512
            dropout_rate = 0.1
            max_t = 50
            seed = 42
        args = Args()
    else:
        # Script mode - parse command line arguments
        parser = argparse.ArgumentParser(description='Test POE-FaRL age estimation model')
        parser.add_argument('--model_path', type=str, default='./output_poe_farl/poe_farl_best.pth',
                            help='Path to the trained POE-FaRL checkpoint')
        parser.add_argument('--test_csv', type=str, default='/home/meem/backup/Age Datasets/test_annotations.csv',
                            help='Path to CSV file with test annotations')
        parser.add_argument('--test_dir', type=str, default='/home/meem/backup/Age Datasets/test',
                            help='Path to directory containing test images')
        parser.add_argument('--output_dir', type=str, default='./poe_farl_test_results',
                            help='Directory to save test results and visualizations')
        parser.add_argument('--farl_weights', type=str, default=None,
                            help='Path to FaRL backbone weights (optional)')
        parser.add_argument('--num_samples', type=int, default=15,
                            help='Number of random samples to visualize')
        parser.add_argument('--uncertainty_samples', type=int, default=100,
                            help='Number of samples for uncertainty estimation')
        parser.add_argument('--embedding_dim', type=int, default=512,
                            help='Embedding dimension (should match training config)')
        parser.add_argument('--dropout_rate', type=float, default=0.1,
                            help='Dropout rate (should match training config)')
        parser.add_argument('--max_t', type=int, default=50,
                            help='Max samples during training (should match training config)')
        parser.add_argument('--seed', type=int, default=42,
                            help='Random seed for reproducibility')
        
        args = parser.parse_args()
    
    # Set up for reproducibility
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("\n===============================================")
    print("     POE-FaRL Age Estimation Test Tool        ")
    print("===============================================\n")
    print(f"Model path: {args.model_path}")
    print(f"Test data: {args.test_csv}")
    print(f"Image directory: {args.test_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"FaRL weights: {args.farl_weights or 'CLIP weights only'}")
    print(f"Uncertainty samples: {args.uncertainty_samples}")
    print(f"Model config: embedding_dim={args.embedding_dim}, dropout={args.dropout_rate}, max_t={args.max_t}")
    print(f"Random seed: {args.seed}")
    print("===============================================\n")
    
    # Check if files exist
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file not found: {args.model_path}")
        exit(1)
    if not os.path.exists(args.test_csv):
        print(f"ERROR: Test CSV file not found: {args.test_csv}")
        exit(1)
    if not os.path.exists(args.test_dir):
        print(f"ERROR: Test directory not found: {args.test_dir}")
        exit(1)
    if args.farl_weights and not os.path.exists(args.farl_weights):
        print(f"ERROR: FaRL weights file not found: {args.farl_weights}")
        exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model config
    config = {
        'embedding_dim': args.embedding_dim,
        'dropout_rate': args.dropout_rate,
        'max_t': args.max_t
    }
    
    # Run test
    try:
        print("Starting POE-FaRL test process...")
        results = test_poe_farl_model(
            model_path=args.model_path,
            test_csv=args.test_csv,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            farl_weights_path=args.farl_weights,
            num_samples_to_visualize=args.num_samples,
            config=config,
            uncertainty_samples=args.uncertainty_samples
        )
        
        # Print final summary
        print("\n===============================================")
        print("              FINAL SUMMARY                    ")
        print("===============================================")
        print(f"Model: {args.model_path}")
        print(f"Mean Absolute Error: {results['mae']:.2f} years")
        print(f"RMSE: {results['rmse']:.2f} years")
        print(f"Correlation (r): {results['correlation']:.4f}")
        print(f"R-squared: {results['r_squared']:.4f}")
        print(f"Percentage within 5 years: {results['within_5_years']:.2f}%")
        print(f"Average Uncertainty: {results['avg_uncertainty']:.2f}")
        print(f"Error-Uncertainty Correlation: {results['uncertainty_correlation']:.4f}")
        print(f"All results saved to: {results['results_dir']}")
        print("===============================================\n")
        
    except Exception as e:
        import traceback
        print(f"\nERROR: Test process failed with error:\n{e}")
        print("\nStack trace:")
        traceback.print_exc()
        print("\nPlease check the paths and parameters and try again.")
        exit(1)

