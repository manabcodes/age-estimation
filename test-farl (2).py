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
# warnings.filterwarnings('ignore')


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

# FaRL MLP Model Definition
class FaRLMLP(nn.Module):
    """FaRL backbone with MLP head for age estimation (regression mode)"""
    
    def __init__(self, farl_model_path=None, hidden_dims=[512], 
                 dropout_rate=0.1, freeze_backbone=True):
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
            feature_dim = dummy_features.shape[-1]
            
        print(f"Detected feature dimension: {feature_dim}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("FaRL backbone frozen")
        
        # Build simple MLP head for regression
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

# Create FaRL MLP model
def create_farl_model(farl_model_path=None, hidden_dims=[512], dropout_rate=0.1):
    model = FaRLMLP(
        farl_model_path=farl_model_path,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        freeze_backbone=True
    )
    return model

# Function to create scatter plot of predictions vs true ages
def create_complete_age_prediction_scatter(predictions, targets, results_dir):
    """Create a scatter plot showing ALL data points of predicted vs true ages"""
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with smaller, more transparent points
    plt.scatter(targets, predictions, alpha=0.3, s=20, color='#3498db', edgecolor='none')
    
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
    
    # Add count information
    plt.text(0.02, 0.95, f"Total samples: {len(predictions)}", fontsize=10, transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Calculate R² and add to plot
    r_squared = model.score(X, y)
    plt.text(0.02, 0.90, f"R² = {r_squared:.3f}", fontsize=10, transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Style the plot
    plt.grid(True, alpha=0.3)
    plt.xlabel('True Age', fontsize=14)
    plt.ylabel('Predicted Age', fontsize=14)
    plt.title('FaRL MLP Age Prediction Results', fontsize=16)
    plt.legend(loc='upper left', fontsize=12)
    
    # Make sure we show the full data range
    buffer = 5
    plt.xlim(min_val - buffer, max_val + buffer)
    plt.ylim(min_val - buffer, max_val + buffer)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'farl_age_prediction_scatter.png'), dpi=300)
    plt.close()

# Helper function to visualize regression predictions
def visualize_regression_predictions(filenames, true_ages, pred_ages, img_dir, output_dir):
    """Visualize predictions for a few samples"""
    n_samples = len(filenames)
    n_cols = min(5, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    # Limit the number of rows to 3 (15 samples max)
    n_rows = min(n_rows, 3)
    n_samples = min(n_samples, n_cols * n_rows)
    
    plt.figure(figsize=(n_cols * 2, n_rows * 2))
    
    for i, (filename, true_age, pred_age) in enumerate(zip(filenames[:n_samples], true_ages[:n_samples], pred_ages[:n_samples])):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Load and display the image
        img_path = os.path.join(img_dir, filename)
        img = Image.open(img_path).convert('RGB')
        plt.imshow(img)
        
        # Add true and predicted ages as text
        error = abs(true_age - pred_age)
        within_5 = "✓" if error <= 5 else "✗"
        color = 'green' if error <= 5 else 'red'
        plt.title(f"True: {true_age:.1f}\nPred: {pred_age:.1f}\n{within_5}", color=color, fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'farl_sample_predictions.png'))
    plt.close()

# Main test function
def test_farl_model(model_path, test_csv, test_dir, output_dir='./farl_test_results', 
                    farl_weights_path=None, num_samples_to_visualize=15):
    """
    Test the FaRL MLP age estimation model and generate comprehensive visualizations
    
    Args:
        model_path: Path to the trained FaRL MLP checkpoint
        test_csv: Path to the CSV file with test annotations
        test_dir: Path to the directory containing test images
        output_dir: Directory to save results and visualizations
        farl_weights_path: Path to FaRL backbone weights (optional)
        num_samples_to_visualize: Number of random samples to visualize
        
    Returns:
        Dictionary with evaluation metrics and results
    """
    # Create output directory
    results_dir = f"{output_dir}{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Creating FaRL MLP model...")
    model = create_farl_model(
        farl_model_path=farl_weights_path,
        hidden_dims=[512],
        dropout_rate=0.1
    )
    
    print(f"Loading model weights from: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model from state_dict key")
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
    
    criterion = nn.MSELoss()  # Use MSE for regression
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, ages, filenames in tqdm(test_loader, desc="Testing FaRL MLP"):
            inputs, targets = inputs.to(device), ages.to(device)
            
            # Forward pass - FaRL outputs direct age predictions
            age_predictions = model(inputs)
            
            # Calculate loss
            loss = criterion(age_predictions, targets)
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate MAE for each sample
            mae = torch.abs(age_predictions - targets)
            
            # Save results
            all_preds.extend(age_predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_filenames.extend(filenames)
            all_errors.extend(mae.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(all_preds)
    targets = np.array(all_targets)
    errors = np.array(all_errors)
        
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
    
    # Print overall metrics
    print("\n=== FARL MLP TEST RESULTS ===")
    print(f"Average MSE Loss: {avg_loss:.4f}")
    print(f"Root Mean Square Error: {rmse:.2f} years")
    print(f"Mean Absolute Error: {avg_mae:.2f} years")
    print(f"Correlation (r): {correlation:.4f}")
    print(f"R²: {correlation**2:.4f}")
    print(f"Within 1 year: {within_1_year:.2f}%")
    print(f"Within 3 years: {within_3_years:.2f}%")
    print(f"Within 5 years: {within_5_years:.2f}%")
    print(f"Within 10 years: {within_10_years:.2f}%")
    
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
    
    for start, end, name in age_groups:
        mask = (targets >= start) & (targets <= end)
        if np.sum(mask) > 0:
            group_mae = np.mean(errors[mask])
            group_rmse = np.sqrt(np.mean((predictions[mask] - targets[mask])**2))
            group_within_5 = np.mean(errors[mask] <= 5) * 100
            group_corr = np.corrcoef(targets[mask], predictions[mask])[0, 1] if np.sum(mask) > 1 else 0
            count = np.sum(mask)
            
            # Store for visualization
            age_group_labels.append(name)
            within_5_years_by_group.append(group_within_5)
            mae_by_group.append(group_mae)
            
            print(f"{name}: MAE: {group_mae:.2f} years, RMSE: {group_rmse:.2f} years, Within 5 years: {group_within_5:.2f}%, r: {group_corr:.3f} (n={count})")
    
    # Create visualizations
    
    # 1. Complete Age Prediction Scatter Plot
    create_complete_age_prediction_scatter(predictions, targets, results_dir)
    
    # 2. Residuals Plot
    plt.figure(figsize=(10, 6))
    residuals = predictions - targets
    plt.scatter(targets, residuals, alpha=0.5, s=20)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.axhline(y=5, color='orange', linestyle=':', linewidth=1, label='+5 years')
    plt.axhline(y=-5, color='orange', linestyle=':', linewidth=1, label='-5 years')
    plt.xlabel('True Age', fontsize=14)
    plt.ylabel('Residuals (Predicted - True)', fontsize=14)
    plt.title('Residuals Plot - FaRL MLP Age Estimation', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'farl_residuals_plot.png'), dpi=300)
    plt.close()
    
    # 3. Average Error by Age
    plt.figure(figsize=(12, 6))
    
    # Calculate average error for each age (bin by 2-year intervals for smoother plot)
    age_bins = np.arange(0, 101, 2)
    avg_errors_binned = []
    age_centers = []
    
    for i in range(len(age_bins)-1):
        mask = (targets >= age_bins[i]) & (targets < age_bins[i+1])
        if np.sum(mask) > 0:
            avg_errors_binned.append(np.mean(errors[mask]))
            age_centers.append((age_bins[i] + age_bins[i+1]) / 2)
    
    plt.plot(age_centers, avg_errors_binned, 'b-', linewidth=2, marker='o', markersize=4)
    plt.axhline(y=avg_mae, color='r', linestyle='--', alpha=0.7, label=f'Overall MAE: {avg_mae:.2f}')
    plt.xlabel('True Age', fontsize=14)
    plt.ylabel('Average Error (years)', fontsize=14)
    plt.title('Average Prediction Error by Age (2-year bins)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'farl_avg_error_by_age.png'), dpi=300)
    plt.close()
    
    # 4. Cumulative Error Distribution
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
    plt.title('Cumulative Error Distribution - FaRL MLP', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'farl_cumulative_error_distribution.png'), dpi=300)
    plt.close()
    
    # 5. Error Distribution Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=avg_mae, color='r', linestyle='--', linewidth=2, label=f'Mean MAE: {avg_mae:.2f}')
    plt.axvline(x=np.median(errors), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.2f}')
    plt.xlabel('Absolute Error (years)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Distribution of Prediction Errors - FaRL MLP', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'farl_error_distribution.png'), dpi=300)
    plt.close()
    
    # 6. Age Group Performance Comparison
    plt.figure(figsize=(12, 8))
    
    # Create subplot for MAE and Within 5 years
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # MAE by age group
    bars1 = ax1.bar(range(len(age_group_labels)), mae_by_group, color='#3498db', alpha=0.7)
    ax1.axhline(y=avg_mae, color='r', linestyle='--', alpha=0.7, label=f'Overall MAE: {avg_mae:.2f}')
    ax1.set_xticks(range(len(age_group_labels)))
    ax1.set_xticklabels(age_group_labels, rotation=45, ha='right')
    ax1.set_ylabel('Mean Absolute Error (years)')
    ax1.set_title('MAE by Age Group - FaRL MLP')
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
    ax2.set_title('Accuracy within 5 Years by Age Group - FaRL MLP')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add data labels
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{within_5_years_by_group[i]:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'farl_age_group_performance.png'), dpi=300)
    plt.close()
    
    # 7. Visualize sample predictions
    if num_samples_to_visualize > 0:
        # Sample random indices
        random_indices = np.random.choice(len(all_preds), min(num_samples_to_visualize, len(all_preds)), replace=False)
        visualize_regression_predictions(
            [all_filenames[i] for i in random_indices],
            [targets[i] for i in random_indices],
            [predictions[i] for i in random_indices],
            test_loader.dataset.img_dir,
            results_dir
        )
    
    # Save detailed results to CSV
    results_df = pd.DataFrame({
        'filename': all_filenames,
        'true_age': targets,
        'predicted_age': predictions,
        'absolute_error': errors,
        'within_5_years': errors <= 5
    })
    
    results_df.to_csv(os.path.join(results_dir, 'farl_results.csv'), index=False)
    print(f"\nSaved detailed results to {results_dir}/farl_results.csv")
    
    # Write summary to text file
    with open(os.path.join(results_dir, 'farl_summary.txt'), 'w') as f:
        f.write("=== FARL MLP MODEL EVALUATION SUMMARY ===\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"FaRL weights: {farl_weights_path or 'CLIP weights only'}\n")
        f.write(f"Test dataset: {test_csv}\n")
        f.write(f"Number of test samples: {len(test_dataset)}\n\n")
        
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
                f.write(f"  - Within 5 years: {within_5_years_by_group[i]:.2f}%\n\n")
        
        # Find worst predictions
        worst_indices = np.argsort(errors)[-10:][::-1]
        f.write("=== WORST PREDICTIONS ===\n")
        for idx in worst_indices:
            f.write(f"File: {all_filenames[idx]}, True: {targets[idx]:.1f}, Predicted: {predictions[idx]:.1f}, Error: {errors[idx]:.1f} years\n")
        
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
        'results_dir': results_dir
    }

if __name__ == "__main__":
    """Main execution function"""
    import argparse
    
    # Handle Jupyter notebook vs script execution
    if any(arg.startswith('-f') for arg in sys.argv):
        # Jupyter mode - use default arguments
        class Args:
            model_path = './output_farl/farl_mlp_final.pth'  # Update this path
            test_csv = '/home/meem/backup/Age Datasets/UTKFace/crop_part1/test_annotations.csv'
            test_dir = '/home/meem/backup/Age Datasets/UTKFace/crop_part1/test'
            output_dir = './farl_test_results'
            farl_weights = None  # Path to FaRL backbone weights (optional)
            num_samples = 15
            seed = 42
        args = Args()
    else:
        # Script mode - parse command line arguments
        parser = argparse.ArgumentParser(description='Test FaRL MLP age estimation model')
        parser.add_argument('--model_path', type=str, default='./output_farl/farl_mlp_best.pth',
                            help='Path to the trained FaRL MLP checkpoint')
        parser.add_argument('--test_csv', type=str, default='/home/meem/backup/Age Datasets/UTKFace/crop_part1/test_annotations.csv',
                            help='Path to CSV file with test annotations')
        parser.add_argument('--test_dir', type=str, default='/home/meem/backup/Age Datasets/UTKFace/crop_part1/test',
                            help='Path to directory containing test images')
        parser.add_argument('--output_dir', type=str, default='./farl_test_results',
                            help='Directory to save test results and visualizations')
        parser.add_argument('--farl_weights', type=str, default=None,
                            help='Path to FaRL backbone weights (optional)')
        parser.add_argument('--num_samples', type=int, default=15,
                            help='Number of random samples to visualize')
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
    print("     FaRL MLP Age Estimation Test Tool        ")
    print("===============================================\n")
    print(f"Model path: {args.model_path}")
    print(f"Test data: {args.test_csv}")
    print(f"Image directory: {args.test_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"FaRL weights: {args.farl_weights or 'CLIP weights only'}")
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
    
    # Run test
    try:
        print("Starting FaRL MLP test process...")
        results = test_farl_model(
            model_path=args.model_path,
            test_csv=args.test_csv,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            farl_weights_path=args.farl_weights,
            num_samples_to_visualize=args.num_samples
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
        print(f"All results saved to: {results['results_dir']}")
        print("===============================================\n")
        
    except Exception as e:
        import traceback
        print(f"\nERROR: Test process failed with error:\n{e}")
        print("\nStack trace:")
        traceback.print_exc()
        print("\nPlease check the paths and parameters and try again.")
        exit(1)

