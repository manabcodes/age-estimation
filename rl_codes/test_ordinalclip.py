#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import os
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


# In[3]:


# Generate timestamp for file naming
timestamp = datetime.now().strftime("-%Y-%m-%d_%H-%M-%S")

# Dataset class for age classification
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
            
        # For ordinal regression, return age as a float
        age = float(age)
        
        return image, age, self.annotations.iloc[idx, 0]  # Return filename for visualization

# OrdinalCLIP Model Definition (copied from your training script)
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

# Function to create scatter plot of predictions vs true ages
def create_complete_age_prediction_scatter(predictions, targets, results_dir):
    """
    Create a scatter plot showing ALL data points of predicted vs true ages
    """
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with smaller, more transparent points to avoid overplotting
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
    
    # Style the plot
    plt.grid(True, alpha=0.3)
    plt.xlabel('True Age', fontsize=14)
    plt.ylabel('Predicted Age', fontsize=14)
    plt.title('OrdinalCLIP Age Prediction Results (All Data Points)', fontsize=16)
    plt.legend(loc='upper left', fontsize=12)
    
    # Make sure we show the full data range
    buffer = 5
    plt.xlim(min_val - buffer, max_val + buffer)
    plt.ylim(min_val - buffer, max_val + buffer)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'ordinalclip_age_prediction_scatter.png'), dpi=300)
    plt.close()

# Helper function to visualize ordinal predictions
def visualize_ordinal_predictions(filenames, true_ages, pred_ages, img_dir, output_dir):
    """
    Visualize predictions for a few samples with continuous age predictions
    """
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
    plt.savefig(os.path.join(output_dir, 'ordinalclip_sample_predictions.png'))
    plt.close()

# Function to visualize probability distributions from OrdinalCLIP
def visualize_ordinal_distributions(probabilities, targets, filenames, img_dir, results_dir, num_samples=10):
    """
    Visualize ordinal probability distributions for selected samples
    """
    # Create directory for individual visualizations
    indiv_dir = os.path.join(results_dir, 'ordinal_distributions')
    os.makedirs(indiv_dir, exist_ok=True)
    
    # Select random samples to visualize
    indices = np.random.choice(len(targets), min(num_samples, len(targets)), replace=False)
    
    # Create a figure for each sample
    for i, idx in enumerate(indices):
        probs = probabilities[idx]
        true_age = targets[idx]
        filename = filenames[idx]
        
        # Calculate statistics for the distribution
        age_values = np.arange(len(probs))
        mean_age = np.sum(age_values * probs)
        variance = np.sum(probs * (age_values - mean_age)**2)
        std_dev = np.sqrt(variance)
        
        # Load image
        img_path = os.path.join(img_dir, filename)
        image = Image.open(img_path).convert('RGB')
        
        # Create a figure with both the image and the distribution
        fig = plt.figure(figsize=(12, 5))
        
        # Display the image
        ax1 = fig.add_subplot(121)
        ax1.imshow(image)
        ax1.set_title(f"True Age: {true_age:.1f}, Predicted: {mean_age:.1f}")
        ax1.axis('off')
        
        # Plot the distribution
        ax2 = fig.add_subplot(122)
        
        # Bar chart of actual probabilities
        bars = ax2.bar(age_values, probs, alpha=0.7, width=1, color='#3498db')
        
        # Add markers for true age and predicted age
        ax2.axvline(x=true_age, color='g', linestyle='--', linewidth=2, label=f'True Age: {true_age:.1f}')
        ax2.axvline(x=mean_age, color='r', linestyle='-', linewidth=2, label=f'Predicted Age: {mean_age:.1f}')
        
        # Add distribution statistics as text in the plot
        text_str = f"μ = {mean_age:.2f}\nσ = {std_dev:.2f}\nMAE = {abs(mean_age - true_age):.2f}"
        ax2.text(0.05, 0.95, text_str, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Show only a relevant window around the prediction
        window = int(max(30, 6*std_dev))
        center = int(mean_age)
        ax2.set_xlim(max(0, center - window//2), min(len(probs), center + window//2))
        
        ax2.set_xlabel('Age')
        ax2.set_ylabel('Probability')
        ax2.set_title(f'OrdinalCLIP Age Distribution (σ = {std_dev:.2f})')
        ax2.legend()
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(indiv_dir, f'ordinal_distribution_{i}_{filename}.png'), dpi=200)
        plt.close()
    
    print(f"Saved {num_samples} ordinal distribution visualizations to {indiv_dir}")

# Main test function for OrdinalCLIP
def test_ordinalclip_model(model_path, test_csv, test_dir, output_dir='./ordinalclip_test_results', 
                          num_classes=122, num_samples_to_visualize=15):
    """
    Test the OrdinalCLIP age estimation model and generate comprehensive visualizations and reports
    """
    # Create output directory
    results_dir = f"{output_dir}{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Creating OrdinalCLIP model with {num_classes} output classes...")
    model = OrdinalCLIP(
        num_classes=num_classes,
        context_length=4,
        rank_embedding_dim=512
    )
    
    print(f"Loading model weights from: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model from state_dict key")
            if 'config' in checkpoint:
                print(f"Model config: {checkpoint['config']}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model from direct state_dict")
            
        model = model.to(device)
        model.eval()
        print("OrdinalCLIP model loaded successfully and set to evaluation mode")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    # Test transforms - using CLIP preprocessing
    print("Setting up CLIP data preprocessing transforms...")
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                           std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    
    # Create dataset and dataloader
    print(f"Loading test dataset from {test_csv} with images in {test_dir}")
    try:
        test_dataset = AgeClassificationDataset(
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
    all_probabilities = []
    
    with torch.no_grad():
        for inputs, ages, filenames in tqdm(test_loader, desc="Testing OrdinalCLIP"):
            inputs, targets = inputs.to(device), ages.to(device)
            
            # Forward pass
            logits = model(inputs)
            
            # Get age predictions and probability distributions
            predicted_ages = model.get_age_predictions(logits)
            probs = F.softmax(logits, dim=1)
            
            # Calculate MAE for each sample
            mae = torch.abs(predicted_ages - targets)
            
            # Save results
            all_preds.extend(predicted_ages.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_filenames.extend(filenames)
            all_errors.extend(mae.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(all_preds)
    targets = np.array(all_targets)
    errors = np.array(all_errors)
    probabilities = np.array(all_probabilities)
        
    print("OrdinalCLIP testing completed successfully!")
    
    # Calculate metrics
    avg_mae = np.mean(errors)
    
    # Calculate accuracy within different thresholds
    within_1_year = np.mean(errors <= 1) * 100
    within_3_years = np.mean(errors <= 3) * 100
    within_5_years = np.mean(errors <= 5) * 100
    within_10_years = np.mean(errors <= 10) * 100
    
    # Print overall metrics
    print("\n=== ORDINALCLIP TEST RESULTS ===")
    print(f"Mean Absolute Error: {avg_mae:.2f} years")
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
    for start, end, name in age_groups:
        mask = (targets >= start) & (targets <= end)
        if np.sum(mask) > 0:
            group_mae = np.mean(errors[mask])
            group_within_5 = np.mean(errors[mask] <= 5) * 100
            count = np.sum(mask)
            print(f"{name}: MAE: {group_mae:.2f} years, Within 5 years: {group_within_5:.2f}% (n={count})")
    
    # Create visualizations
    
    # 1. Age Prediction Scatter Plot
    create_complete_age_prediction_scatter(predictions, targets, results_dir)
    
    # 2. Cumulative Error Distribution
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
    plt.title('OrdinalCLIP Cumulative Error Distribution', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'ordinalclip_cumulative_error_distribution.png'))
    plt.close()
    
    # 3. Visualize sample predictions
    if num_samples_to_visualize > 0:
        random_indices = np.random.choice(len(all_preds), min(num_samples_to_visualize, len(all_preds)), replace=False)
        visualize_ordinal_predictions(
            [all_filenames[i] for i in random_indices],
            [targets[i] for i in random_indices],
            [predictions[i] for i in random_indices],
            test_loader.dataset.img_dir,
            results_dir
        )
    
    # 4. Visualize ordinal distributions
    visualize_ordinal_distributions(
        probabilities=probabilities,
        targets=targets,
        filenames=all_filenames,
        img_dir=test_dir,
        results_dir=results_dir,
        num_samples=15
    )
    
    # 5. Average probability distribution analysis
    print("\nAnalyzing ordinal distributions...")
    
    # Calculate distribution statistics
    means = []
    stds = []
    
    for i, probs in enumerate(probabilities):
        age_values = np.arange(len(probs))
        mean_age = np.sum(age_values * probs)
        variance = np.sum(probs * (age_values - mean_age)**2)
        std_dev = np.sqrt(variance)
        
        means.append(mean_age)
        stds.append(std_dev)
    
    means = np.array(means)
    stds = np.array(stds)
    
    # Plot distribution of standard deviations
    plt.figure(figsize=(10, 6))
    plt.hist(stds, bins=30, alpha=0.7, color='#3498db')
    plt.axvline(x=np.mean(stds), color='r', linestyle='--', 
                label=f'Mean σ: {np.mean(stds):.2f}')
    
    plt.xlabel('Standard Deviation (σ)')
    plt.ylabel('Frequency')
    plt.title('OrdinalCLIP: Distribution of Age Prediction Standard Deviations')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'ordinalclip_std_dev_distribution.png'), dpi=300)
    plt.close()
    
    # Correlation between distribution width and error
    corr = np.corrcoef(stds, errors)[0, 1]
    print(f"Correlation between distribution width and error: {corr:.4f}")
    
    # Save detailed results to CSV
    results_df = pd.DataFrame({
        'filename': all_filenames,
        'true_age': targets,
        'predicted_age': predictions,
        'absolute_error': errors,
        'distribution_std': stds,
        'within_5_years': errors <= 5
    })
    
    results_df.to_csv(os.path.join(results_dir, 'ordinalclip_results.csv'), index=False)
    print(f"\nSaved detailed results to {results_dir}/ordinalclip_results.csv")
    
    # Write summary to text file
    with open(os.path.join(results_dir, 'ordinalclip_summary.txt'), 'w') as f:
        f.write("=== ORDINALCLIP MODEL EVALUATION SUMMARY ===\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test dataset: {test_csv}\n")
        f.write(f"Number of test samples: {len(test_dataset)}\n")
        f.write(f"Number of age classes: {num_classes}\n\n")
        
        f.write("=== OVERALL METRICS ===\n")
        f.write(f"Mean Absolute Error: {avg_mae:.2f} years\n\n")
        
        f.write("=== ERROR THRESHOLDS ===\n")
        f.write(f"Within 1 year: {within_1_year:.2f}%\n")
        f.write(f"Within 3 years: {within_3_years:.2f}%\n")
        f.write(f"Within 5 years: {within_5_years:.2f}%\n")
        f.write(f"Within 10 years: {within_10_years:.2f}%\n\n")
        
        # Calculate correlation between true age and predicted age
        correlation = np.corrcoef(targets, predictions)[0, 1]
        f.write(f"Correlation between true and predicted age: {correlation:.4f}\n")
        f.write(f"Average distribution standard deviation: {np.mean(stds):.2f} years\n")
        f.write(f"Correlation between distribution width and error: {corr:.4f}\n\n")
        
        f.write("=== RESULTS BY AGE GROUP ===\n")
        for start, end, name in age_groups:
            mask = (targets >= start) & (targets <= end)
            if np.sum(mask) > 0:
                group_mae = np.mean(errors[mask])
                group_within_5 = np.mean(errors[mask] <= 5) * 100
                group_within_10 = np.mean(errors[mask] <= 10) * 100
                count = np.sum(mask)
                f.write(f"{name}:\n")
                f.write(f"  - Sample count: {count}\n")
                f.write(f"  - MAE: {group_mae:.2f} years\n")
                f.write(f"  - Within 5 years: {group_within_5:.2f}%\n")
                f.write(f"  - Within 10 years: {group_within_10:.2f}%\n\n")
        
        # Find bias in predictions
        avg_bias = np.mean(predictions - targets)
        f.write(f"Average bias (predicted - true): {avg_bias:.2f} years\n\n")
        
        f.write("=== ORDINAL REGRESSION ANALYSIS ===\n")
        f.write(f"Prediction method: Expected value from softmax distribution\n")
        f.write(f"Continuous age predictions: Yes\n")
        f.write(f"Uses ordinal constraints: Yes (learnable rank embeddings)\n")
        f.write(f"Language-guided: Yes (CLIP text encoder)\n\n")
        
        f.write("=== VISUALIZATIONS GENERATED ===\n")
        f.write("- ordinalclip_age_prediction_scatter.png: Scatter plot of predicted vs true ages\n")
        f.write("- ordinalclip_cumulative_error_distribution.png: Cumulative error distribution\n")
        f.write("- ordinalclip_sample_predictions.png: Sample prediction visualizations\n")
        f.write("- ordinalclip_std_dev_distribution.png: Distribution of prediction uncertainties\n")
        f.write("- ordinal_distributions/: Individual probability distribution visualizations\n")
    
    print(f"\nAll results saved to: {results_dir}")
    return {
        'mae': avg_mae,
        'within_5_years': within_5_years,
        'results_dir': results_dir,
        'correlation': correlation
    }

# Jupyter notebook execution
if __name__ == "__main__":
    # Set up paths for OrdinalCLIP testing
    model_path = './output_ordinalclip/ordinalclip_best.pth'  # Update this path
    # test_csv = '/home/meem/backup/Age Datasets/test_annotations.csv'
    # test_dir = '/home/meem/backup/Age Datasets/test'
    test_csv = '/home/meem/backup/filtered/unified_age_dataset/test_annotations.csv'
    test_dir = '/home/meem/backup/filtered/unified_age_dataset/test'
    output_dir = './ordinalclip_test_results'
    num_classes = 122  # Update based on your training config
    num_samples = 15
    
    # Set up for reproducibility
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print("\n===============================================")
    print("     OrdinalCLIP Age Estimation Testing Tool     ")
    print("===============================================\n")
    print(f"Model path: {model_path}")
    print(f"Test data: {test_csv}")
    print(f"Image directory: {test_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of age classes: {num_classes}")
    print(f"Random seed: {seed}")
    print("===============================================\n")
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        print("Please update the model_path variable to point to your trained OrdinalCLIP model.")
    elif not os.path.exists(test_csv):
        print(f"ERROR: Test CSV file not found: {test_csv}")
        print("Please update the test_csv variable to point to your test annotations.")
    elif not os.path.exists(test_dir):
        print(f"ERROR: Test directory not found: {test_dir}")
        print("Please update the test_dir variable to point to your test images.")
    else:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Run test
        try:
            print("Starting OrdinalCLIP test process...")
            results = test_ordinalclip_model(
                model_path=model_path,
                test_csv=test_csv,
                test_dir=test_dir,
                output_dir=output_dir,
                num_classes=num_classes,
                num_samples_to_visualize=num_samples
            )
            
            # Print final summary
            print("\n===============================================")
            print("              ORDINALCLIP FINAL SUMMARY                ")
            print("===============================================")
            print(f"Model: {model_path}")
            print(f"Mean Absolute Error: {results['mae']:.2f} years")
            print(f"Percentage within 5 years: {results['within_5_years']:.2f}%")
            print(f"Age prediction correlation: {results['correlation']:.4f}")
            print(f"All results saved to: {results['results_dir']}")
            print("===============================================\n")
            
        except Exception as e:
            import traceback
            print(f"\nERROR: OrdinalCLIP test process failed with error:\n{e}")
            print("\nStack trace:")
            traceback.print_exc()
            print("\nPlease check the paths and parameters and try again.")
            print("\nCommon issues:")
            print("1. Make sure the model_path points to a valid OrdinalCLIP checkpoint")
            print("2. Verify that num_classes matches your training configuration")
            print("3. Ensure test_csv and test_dir paths are correct")
            print("4. Check that you have sufficient GPU memory if using CUDA")

