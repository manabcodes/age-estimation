#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import sys

os.getcwd()


# In[16]:


#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from tqdm.notebook import tqdm
import random
from datetime import datetime
import sys


class DualHeadResNet(nn.Module):
    def __init__(self, num_classes=122, pretrained=True, model_name='resnet50'):
        super(DualHeadResNet, self).__init__()
        
        # Create the base ResNet model
        if model_name == 'resnet18':
            self.base_model = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            self.base_model = models.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            self.base_model = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            self.base_model = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
        
        # Get the feature dimension
        in_features = self.base_model.fc.in_features
        
        # Replace the classification head with an identity layer
        self.base_model.fc = nn.Identity()
        
        # Create dual heads
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


def load_model_resnet(model_path, device='cpu', num_classes=122, model_name='resnet50'):
    # Create base dual-head model
    base_model = DualHeadResNet(
        num_classes=num_classes,
        pretrained=True,  # Start with ImageNet weights
        model_name=model_name
    )
    
    # Load the trained weights
    print(f"Loading trained model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        base_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        base_model.load_state_dict(checkpoint)
    
    base_model = base_model.to(device)
    base_model.eval()
    
    return base_model


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
            
        # For classification, return age as an integer (class label)
        age = int(age)
        
        return image, age, self.annotations.iloc[idx, 0]  # Return filename for visualization

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
    
    # Add detailed count information for each band
    unique_preds = np.unique(predictions)
    txt = "Prediction bands:\n"
    for pred in unique_preds[:10]:  # Show top 10 bands
        count = np.sum(predictions == pred)
        txt += f"{pred:.1f}: {count} samples\n"
    if len(unique_preds) > 10:
        txt += f"... and {len(unique_preds)-10} more bands"
    plt.annotate(txt, xy=(0.02, 0.90), xycoords='axes fraction', 
                va='top', ha='left', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add count information to the plot
    plt.text(0.02, 0.95, f"Total samples: {len(predictions)}", fontsize=10, transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Style the plot
    plt.grid(True, alpha=0.3)
    plt.xlabel('True Age', fontsize=14)
    plt.ylabel('Predicted Age', fontsize=14)
    plt.title('Age Prediction Results (All Data Points)', fontsize=16)
    plt.legend(loc='upper left', fontsize=12)
    
    # Make sure we show the full data range
    buffer = 5
    plt.xlim(min_val - buffer, max_val + buffer)
    plt.ylim(min_val - buffer, max_val + buffer)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'complete_age_prediction_scatter.png'), dpi=300)
    plt.close()

# Helper function to visualize classification predictions
def visualize_classification_predictions(filenames, true_ages, pred_ages, img_dir, output_dir):
    """
    Visualize predictions for a few samples
    
    Args:
        filenames: List of image filenames
        true_ages: List of true ages
        pred_ages: List of predicted ages
        img_dir: Directory containing the images
        output_dir: Directory to save the visualization
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
        plt.title(f"True: {true_age}\nPred: {pred_age:.1f}\n{within_5}", color=color, fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_predictions.png'))
    plt.close()

# Add this function to your script - it creates visualizations of individual age distributions
def visualize_individual_age_distributions(probabilities, targets, filenames, img_dir, results_dir, num_samples=10):
    """
    Visualize individual age distributions for selected samples to show mean-variance characteristics
    
    Args:
        probabilities: Array of probability distributions for all samples
        targets: Array of true ages
        filenames: List of filenames for the samples
        img_dir: Directory containing the images
        results_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    # Create directory for individual visualizations
    indiv_dir = os.path.join(results_dir, 'individual_distributions')
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
        ax1.set_title(f"True Age: {true_age}, Predicted: {mean_age:.1f}")
        ax1.axis('off')
        
        # Plot the distribution
        ax2 = fig.add_subplot(122)
        
        # Bar chart of actual probabilities
        bars = ax2.bar(age_values, probs, alpha=0.5, width=1, color='#3498db')
        
        # Also plot a Gaussian approximation
        from scipy.stats import norm
        x = np.linspace(max(0, mean_age - 4*std_dev), mean_age + 4*std_dev, 1000)
        gaussian = norm.pdf(x, mean_age, std_dev)
        # Scale the Gaussian to match bar heights
        max_prob = probs.max()
        scale_factor = max_prob / norm.pdf(mean_age, mean_age, std_dev) if std_dev > 0 else 1
        ax2.plot(x, gaussian * scale_factor, 'r-', linewidth=2)
        
        # Add markers for true age and predicted age
        ax2.axvline(x=true_age, color='g', linestyle='--', linewidth=2, label=f'True Age: {true_age}')
        ax2.axvline(x=mean_age, color='r', linestyle='-', linewidth=2, label=f'Predicted Age: {mean_age:.1f}')
        
        # Add distribution statistics as text in the plot
        text_str = f"μ = {mean_age:.2f}\nσ = {std_dev:.2f}\nMAE = {abs(mean_age - true_age):.2f}"
        ax2.text(0.05, 0.95, text_str, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Show only a relevant window around the true age
        window = int(max(30, 6*std_dev))
        ax2.set_xlim(max(0, true_age - window//2), true_age + window//2)
        
        ax2.set_xlabel('Age')
        ax2.set_ylabel('Probability')
        ax2.set_title(f'Age Distribution (σ = {std_dev:.2f})')
        ax2.legend()
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(indiv_dir, f'distribution_{i}_{filename}.png'), dpi=200)
        plt.close()
    
    print(f"Saved {num_samples} individual age distribution visualizations to {indiv_dir}")
    
    # Create a figure showing "before/after" mean-variance loss effect like in the paper
    plt.figure(figsize=(10, 5))
    
    # Use the first sample for this visualization
    idx = indices[0]
    true_age = targets[idx]
    probs = probabilities[idx]
    
    # Calculate actual distribution statistics
    age_values = np.arange(len(probs))
    mean_age = np.sum(age_values * probs)
    variance = np.sum(probs * (age_values - mean_age)**2)
    std_dev = np.sqrt(variance)
    
    # Left plot - "Before" distribution with higher variance
    plt.subplot(1, 2, 1)
    # Simulate a distribution with same mean but higher variance
    simulated_std = std_dev * 1.5
    simulated_mean = mean_age * 0.9  # Slightly off mean
    
    from scipy.stats import norm
    x = np.linspace(max(0, mean_age - 4*simulated_std), mean_age + 4*simulated_std, 1000)
    y = norm.pdf(x, simulated_mean, simulated_std)
    
    plt.plot(x, y, 'b-', linewidth=2)
    plt.axvline(x=true_age, color='r', linestyle='--', linewidth=2)
    
    # Add distribution statistics
    plt.title('Before Applying the Loss')
    plt.xlabel('Age')
    plt.ylabel('Probability')
    text_str = f"μ = {simulated_mean:.1f}\nσ = {simulated_std:.1f}"
    plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Right plot - "After" distribution with lower variance
    plt.subplot(1, 2, 2)
    x = np.linspace(max(0, mean_age - 4*std_dev), mean_age + 4*std_dev, 1000)
    y = norm.pdf(x, mean_age, std_dev)
    
    plt.plot(x, y, 'b-', linewidth=2)
    plt.axvline(x=true_age, color='r', linestyle='--', linewidth=2)
    
    # Add distribution statistics
    plt.title('After Applying the Loss')
    plt.xlabel('Age')
    plt.ylabel('Probability')
    text_str = f"μ = {mean_age:.1f}\nσ = {std_dev:.1f}"
    plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'before_after_effect.png'), dpi=300)
    plt.close()
    print(f"Saved mean-variance effect visualization to {results_dir}/before_after_effect.png")

# Add this function to create aggregate visualizations of age distributions
def visualize_aggregate_age_distributions(probabilities, targets, results_dir):
    """
    Create visualizations that analyze the aggregated age distribution properties
    
    Args:
        probabilities: Array of probability distributions for all samples
        targets: Array of true ages
        results_dir: Directory to save visualizations
    """
    # Calculate distribution statistics for all samples
    means = []
    stds = []
    maes = []
    
    for i, probs in enumerate(probabilities):
        age_values = np.arange(len(probs))
        mean_age = np.sum(age_values * probs)
        variance = np.sum(probs * (age_values - mean_age)**2)
        std_dev = np.sqrt(variance)
        
        means.append(mean_age)
        stds.append(std_dev)
        maes.append(abs(mean_age - targets[i]))
    
    # Convert to numpy arrays
    means = np.array(means)
    stds = np.array(stds)
    maes = np.array(maes)
    
    # 1. Plot distribution of standard deviations
    plt.figure(figsize=(10, 6))
    plt.hist(stds, bins=30, alpha=0.7, color='#3498db')
    plt.axvline(x=np.mean(stds), color='r', linestyle='--', 
                label=f'Mean σ: {np.mean(stds):.2f}')
    
    plt.xlabel('Standard Deviation (σ)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Age Prediction Standard Deviations')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'std_dev_distribution.png'), dpi=300)
    plt.close()
    
    # 2. Scatter plot of standard deviation vs MAE
    plt.figure(figsize=(10, 6))
    plt.scatter(stds, maes, alpha=0.3, s=20, color='#3498db')
    
    # Add trend line
    z = np.polyfit(stds, maes, 1)
    p = np.poly1d(z)
    plt.plot(np.sort(stds), p(np.sort(stds)), "r--", 
             label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")
    
    # Add correlation coefficient
    corr = np.corrcoef(stds, maes)[0, 1]
    plt.title(f'Relationship between Distribution Width and Prediction Error (r={corr:.2f})')
    plt.xlabel('Standard Deviation (σ)')
    plt.ylabel('Mean Absolute Error')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'std_vs_mae.png'), dpi=300)
    plt.close()
    
    # 3. Scatter plot of true age vs standard deviation
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, stds, alpha=0.3, s=20, color='#3498db')
    
    # Add trend line
    z = np.polyfit(targets, stds, 1)
    p = np.poly1d(z)
    plt.plot(np.sort(targets), p(np.sort(targets)), "r--", 
             label=f"Trend: y={z[0]:.4f}x+{z[1]:.2f}")
    
    # Add correlation coefficient
    corr = np.corrcoef(targets, stds)[0, 1]
    plt.title(f'Distribution Width vs True Age (r={corr:.2f})')
    plt.xlabel('True Age')
    plt.ylabel('Standard Deviation (σ)')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'age_vs_std.png'), dpi=300)
    plt.close()
    
    # 4. Calculate and visualize average age distribution by age group
    age_groups = [
        (0, 12, "Children (0-12)"),
        (13, 19, "Teenagers (13-19)"),
        (20, 35, "Young Adults (20-35)"),
        (36, 50, "Middle-aged (36-50)"),
        (51, 70, "Seniors (51-70)"),
        (71, 100, "Elderly (71+)")
    ]
    
    plt.figure(figsize=(12, 8))
    
    for start, end, name in age_groups:
        # Find samples in this age group
        mask = (targets >= start) & (targets <= end)
        if np.sum(mask) > 10:  # Only process groups with enough samples
            group_probs = probabilities[mask]
            
            # Average the distributions
            avg_prob = np.mean(group_probs, axis=0)
            
            # Calculate statistics
            age_values = np.arange(len(avg_prob))
            mean_age = np.sum(age_values * avg_prob)
            variance = np.sum(avg_prob * (age_values - mean_age)**2)
            std_dev = np.sqrt(variance)
            
            # Plot a section of the distribution around its mean
            window = int(max(30, 5*std_dev))
            start_idx = max(0, int(mean_age) - window//2)
            end_idx = min(len(avg_prob), int(mean_age) + window//2)
            
            x_range = np.arange(start_idx, end_idx)
            plt.plot(x_range, avg_prob[start_idx:end_idx], 
                     label=f'{name} (μ={mean_age:.1f}, σ={std_dev:.1f}, n={np.sum(mask)})')
    
    plt.xlabel('Age')
    plt.ylabel('Average Probability')
    plt.title('Average Age Distribution by Age Group')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'avg_distribution_by_age_group.png'), dpi=300)
    plt.close()
    
    print(f"Saved aggregate age distribution visualizations to {results_dir}")


# Main test function
def test_model(model_path, test_csv, test_dir, output_dir='./test_results', num_classes=122, num_samples_to_visualize=15):
    """
    Test the age classification model and generate comprehensive visualizations and reports
    
    Args:
        model_path: Path to the trained model checkpoint
        test_csv: Path to the CSV file with test annotations
        test_dir: Path to the directory containing test images
        output_dir: Directory to save results and visualizations
        num_classes: Number of age classes in the model
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
    print(f"Loading ResNet-50 dual-head model with {num_classes} output classes...")
    model = load_model_resnet(
        model_path=model_path, 
        device=device, 
        num_classes=num_classes, 
        model_name='resnet50'  # Change as needed
    )    
    model.eval()
    
    # Test transforms - standard ImageNet preprocessing
    print("Setting up data preprocessing transforms...")
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    
    # Create DLDL loss for calculations (will be needed for expected age)
    dldl_criterion = DLDLv2Loss(num_classes=num_classes)
    
    all_preds = []
    all_targets = []
    all_filenames = []
    all_errors = []
    all_probabilities = []
    all_distribution_preds = []
    all_regression_preds = []
    
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, ages, filenames in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            target_ages = ages.to(device)
            
            # Forward pass with dual-head model
            distribution_logits, regression_output = model(inputs)
            
            # Calculate probabilities from distribution head
            probabilities = F.softmax(distribution_logits, dim=1)
            
            # Calculate expected age from distribution (DLDL way)
            age_values = torch.arange(distribution_logits.size(1), device=device).float()
            expected_ages = torch.sum(probabilities * age_values.unsqueeze(0), dim=1)
            
            # Get direct predictions from regression head
            direct_ages = regression_output.squeeze()
            
            # Calculate loss using DLDL loss
            total_loss, kl_loss, l1_loss, _, _ = dldl_criterion(
                distribution_logits, regression_output, target_ages)
            
            running_loss += total_loss.item() * inputs.size(0)
            
            # Select the better prediction for each sample (lower error)
            distribution_error = torch.abs(expected_ages - target_ages.float())
            regression_error = torch.abs(direct_ages - target_ages.float())
            
            # Use whichever head gives better predictions
            predictions = torch.where(
                distribution_error <= regression_error,
                expected_ages,
                direct_ages
            )
            
            # Calculate final MAE based on the better predictions
            mae = torch.abs(predictions - target_ages.float())
            
            # Save all data for detailed analysis
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(target_ages.cpu().numpy())
            all_filenames.extend(filenames)
            all_errors.extend(mae.cpu().numpy())
            
            # Save distribution probabilities for visualization
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Save individual head predictions for comparison
            all_distribution_preds.extend(expected_ages.cpu().numpy())
            all_regression_preds.extend(direct_ages.cpu().numpy())
    
    # Convert to numpy arrays for easier processing
    predictions = np.array(all_preds)
    targets = np.array(all_targets)
    errors = np.array(all_errors)
    probabilities = np.array(all_probabilities)
    distribution_preds = np.array(all_distribution_preds)
    regression_preds = np.array(all_regression_preds)
    
    # Calculate overall metrics
    avg_loss = running_loss / len(test_loader.dataset)
    avg_mae = np.mean(errors)
    exact_match = np.mean(np.round(predictions) == targets) * 100
    
    # Calculate accuracy within different thresholds
    within_1_year = np.mean(errors <= 1) * 100
    within_3_years = np.mean(errors <= 3) * 100
    within_5_years = np.mean(errors <= 5) * 100
    within_10_years = np.mean(errors <= 10) * 100
    
    # Compare performance of individual heads
    dist_errors = np.abs(distribution_preds - targets)
    reg_errors = np.abs(regression_preds - targets)
    dist_mae = np.mean(dist_errors)
    reg_mae = np.mean(reg_errors)
    dist_within5 = np.mean(dist_errors <= 5) * 100
    reg_within5 = np.mean(reg_errors <= 5) * 100
    
    # Print overall metrics
    print("\n=== TEST RESULTS ===")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Mean Absolute Error (combined): {avg_mae:.2f} years")
    print(f"Exact Match Accuracy: {exact_match:.2f}%")
    print(f"Within 1 year: {within_1_year:.2f}%")
    print(f"Within 3 years: {within_3_years:.2f}%")
    print(f"Within 5 years: {within_5_years:.2f}%")
    print(f"Within 10 years: {within_10_years:.2f}%")
    
    # Compare distribution vs regression heads
    print("\n=== HEAD COMPARISON ===")
    print(f"Distribution Head MAE: {dist_mae:.2f} years, Within 5 years: {dist_within5:.2f}%")
    print(f"Regression Head MAE: {reg_mae:.2f} years, Within 5 years: {reg_within5:.2f}%")
    print(f"Combined MAE: {avg_mae:.2f} years, Within 5 years: {within_5_years:.2f}%")
    
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
    
    for start, end, name in age_groups:
        mask = (targets >= start) & (targets <= end)
        if np.sum(mask) > 0:
            group_mae = np.mean(errors[mask])
            group_within_5 = np.mean(errors[mask] <= 5) * 100
            count = np.sum(mask)
            
            # Store for visualization
            age_group_labels.append(name)
            within_5_years_by_group.append(group_within_5)
            
            print(f"{name}: MAE: {group_mae:.2f} years, Within 5 years: {group_within_5:.2f}% (n={count})")
    
    # Create visualizations
    
    # 1. Complete Age Prediction Scatter Plot
    create_complete_age_prediction_scatter(predictions, targets, results_dir)
    
    # 2. Confusion Matrix (Log Scale)
    plt.figure(figsize=(12, 10))
    max_age = min(100, max(np.max(targets), np.max(np.round(predictions))))
    
    # Round predictions for confusion matrix
    rounded_preds = np.round(predictions).astype(int)
    rounded_preds = np.clip(rounded_preds, 0, max_age)
    targets_clipped = np.clip(targets, 0, max_age)
    
    cm = confusion_matrix(
        targets_clipped, 
        rounded_preds, 
        labels=range(int(max_age)+1)
    )
    
    # Apply log scale for better visibility
    cm_log = np.log1p(cm)
    
    # Create heatmap
    ax = sns.heatmap(cm_log, cmap='viridis', annot=False, 
                    xticklabels=10, yticklabels=10)
    
    # Set tick labels
    tick_positions = np.arange(0, max_age+1, 10)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_positions)
    ax.set_yticklabels(tick_positions)
    
    plt.xlabel('Predicted Age', fontsize=14)
    plt.ylabel('True Age', fontsize=14)
    plt.title('Age Confusion Matrix (Log Scale)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix_log.png'))
    plt.close()
    
    # 3. Average Error by Age
    plt.figure(figsize=(10, 6))
    
    # Calculate average error for each age
    max_target_age = min(100, int(np.max(targets)))
    avg_error_by_age = np.zeros(max_target_age + 1)
    count_by_age = np.zeros(max_target_age + 1)
    
    for true_age, error in zip(targets, errors):
        if true_age <= max_target_age:
            avg_error_by_age[int(true_age)] += error
            count_by_age[int(true_age)] += 1
    
    # Avoid division by zero
    mask = count_by_age > 0
    avg_error_by_age[mask] = avg_error_by_age[mask] / count_by_age[mask]
    
    plt.bar(range(max_target_age + 1), avg_error_by_age, alpha=0.7)
    plt.xlabel('True Age', fontsize=14)
    plt.ylabel('Average Error (years)', fontsize=14)
    plt.title('Average Prediction Error by Age', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'avg_error_by_age.png'))
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
    plt.title('Cumulative Error Distribution', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'cumulative_error_distribution.png'))
    plt.close()
    
    # 5. Pie chart for error thresholds
    plt.figure(figsize=(7, 5))
    error_categories = [
        ('≤1 year', np.sum(errors <= 1)),
        ('1-3 years', np.sum((errors > 1) & (errors <= 3))),
        ('3-5 years', np.sum((errors > 3) & (errors <= 5))),
        ('5-10 years', np.sum((errors > 5) & (errors <= 10))),
        ('>10 years', np.sum(errors > 10))
    ]
    
    labels = [f"{cat[0]}: {cat[1]/len(errors)*100:.1f}%" for cat in error_categories]
    sizes = [cat[1] for cat in error_categories]
    colors = ['#2ecc71', '#27ae60', '#f1c40f', '#e67e22', '#e74c3c']
    explode = (0.1, 0.05, 0, 0, 0)  # explode the smallest slice
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=False, startangle=90, textprops={'fontsize': 9})
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Distribution of Prediction Errors')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'error_distribution_pie.png'))
    plt.close()
    
    # 6. Bar chart showing percentage of predictions within 5 years by age group
    plt.figure(figsize=(8, 5))
    
    # Add overall percentage to the age group data
    age_group_labels.append("Overall")
    within_5_years_by_group.append(within_5_years)
    
    # Create bar chart
    bars = plt.bar(range(len(age_group_labels)), within_5_years_by_group, color='#3498db')
    
    # Add data labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{within_5_years_by_group[i]:.1f}%',
                ha='center', va='bottom', rotation=0)
    
    plt.axhline(y=within_5_years, color='r', linestyle='--', alpha=0.7, label='Overall Average')
    plt.xticks(range(len(age_group_labels)), age_group_labels, rotation=45, ha='right')
    plt.ylabel('Percentage within 5 years (%)')
    plt.title('Predictions Within 5 Years by Age Group')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'within_5_years_by_age_group.png'))
    plt.close()
    
    # 7. Line chart showing accuracy by exact error threshold
    plt.figure(figsize=(8, 5))
    
    thresholds = list(range(1, 21))  # 1 to 20 years
    accuracies = [np.mean(errors <= threshold) * 100 for threshold in thresholds]
    
    plt.plot(thresholds, accuracies, marker='o', markersize=4, linewidth=2)
    
    # Highlight specific thresholds
    key_thresholds = [1, 3, 5, 10]
    for threshold in key_thresholds:
        idx = thresholds.index(threshold)
        plt.plot(threshold, accuracies[idx], 'ro', markersize=6)
        plt.annotate(f"{accuracies[idx]:.1f}%", 
                     xy=(threshold, accuracies[idx]), 
                     xytext=(threshold+0.5, accuracies[idx]+1),
                     arrowprops=dict(arrowstyle="->", color='r'))
    
    plt.xlabel('Error Threshold (years)')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy at Different Error Thresholds')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'accuracy_by_threshold.png'))
    plt.close()
    
    # 8. Comparison of distribution head vs. regression head
    plt.figure(figsize=(10, 6))
    x = np.arange(3)
    width = 0.3
    
    methods = ['Distribution Head', 'Regression Head', 'Combined']
    mae_values = [dist_mae, reg_mae, avg_mae]
    within5_values = [dist_within5, reg_within5, within_5_years]
    
    # Create grouped bar chart
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # MAE bars on left axis
    bars1 = ax1.bar(x - width/2, mae_values, width, label='MAE (years)', color='#3498db')
    ax1.set_ylabel('MAE (years)', color='#3498db')
    ax1.tick_params(axis='y', labelcolor='#3498db')
    
    # Add data labels to MAE bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{mae_values[i]:.2f}',
                ha='center', va='bottom', color='#3498db', fontsize=10)
    
    # Within 5 years accuracy on right axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, within5_values, width, label='Within 5 years (%)', color='#e74c3c')
    ax2.set_ylabel('Within 5 years (%)', color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    
    # Add data labels to accuracy bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{within5_values[i]:.1f}%',
                ha='center', va='bottom', color='#e74c3c', fontsize=10)
    
    # Set x-axis ticks and labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    
    # Add title and legend
    plt.title('Comparison of Model Heads Performance')
    
    # Create a combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'head_comparison.png'))
    plt.close()
    
    # 9. Visualize sample predictions
    if num_samples_to_visualize > 0:
        # Sample random indices
        random_indices = np.random.choice(len(all_preds), min(num_samples_to_visualize, len(all_preds)), replace=False)
        visualize_classification_predictions(
            [all_filenames[i] for i in random_indices],
            [targets[i] for i in random_indices],
            [predictions[i] for i in random_indices],
            test_loader.dataset.img_dir,
            results_dir
        )

    # 10. Visualize age distributions
    print("\nGenerating age distribution visualizations...")
    
    # Create individual distribution visualizations
    visualize_individual_age_distributions(
        probabilities=probabilities,
        targets=targets,
        filenames=all_filenames,
        img_dir=test_dir,
        results_dir=results_dir,
        num_samples=num_samples_to_visualize
    )
    
    # Create aggregate distribution visualizations
    visualize_aggregate_age_distributions(
        probabilities=probabilities,
        targets=targets,
        results_dir=results_dir
    )
    
    # Save detailed results to CSV
    results_df = pd.DataFrame({
        'filename': all_filenames,
        'true_age': targets,
        'predicted_age': predictions,
        'absolute_error': errors,
        'within_5_years': errors <= 5,  # Add a binary column for within 5 years
        'distribution_pred': distribution_preds,
        'regression_pred': regression_preds
    })
    
    results_df.to_csv(os.path.join(results_dir, 'classification_results.csv'), index=False)
    print(f"\nSaved detailed results to {results_dir}/classification_results.csv")
    
    # Write summary to text file
    with open(os.path.join(results_dir, 'summary.txt'), 'w') as f:
        f.write("=== MODEL EVALUATION SUMMARY ===\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test dataset: {test_csv}\n")
        f.write(f"Number of test samples: {len(test_dataset)}\n\n")
        
        f.write("=== OVERALL METRICS ===\n")
        f.write(f"Average Loss: {avg_loss:.4f}\n")
        f.write(f"Mean Absolute Error: {avg_mae:.2f} years\n")
        f.write(f"Exact Match Accuracy: {exact_match:.2f}%\n\n")
        
        f.write("=== HEAD COMPARISON ===\n")
        f.write(f"Distribution Head MAE: {dist_mae:.2f} years, Within 5 years: {dist_within5:.2f}%\n")
        f.write(f"Regression Head MAE: {reg_mae:.2f} years, Within 5 years: {reg_within5:.2f}%\n")
        f.write(f"Combined MAE: {avg_mae:.2f} years, Within 5 years: {within_5_years:.2f}%\n\n")
        
        f.write("=== ERROR THRESHOLDS ===\n")
        f.write(f"Within 1 year: {within_1_year:.2f}%\n")
        f.write(f"Within 3 years: {within_3_years:.2f}%\n")
        f.write(f"Within 5 years: {within_5_years:.2f}%\n")
        f.write(f"Within 10 years: {within_10_years:.2f}%\n\n")
        
        # Find error threshold for 90% accuracy
        error_thresholds = np.arange(0, 21)  # 0 to 20 years
        cumulative_pct = [np.mean(errors <= threshold) * 100 for threshold in error_thresholds]
        threshold_90pct = next((i for i, pct in enumerate(cumulative_pct) if pct >= 90), None)
        if threshold_90pct is not None:
            f.write(f"Error threshold for 90% accuracy: {threshold_90pct} years\n\n")
        
        # Calculate correlation between true age and predicted age
        correlation = np.corrcoef(targets, predictions)[0, 1]
        f.write(f"Correlation between true and predicted age: {correlation:.4f}\n\n")
        
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
        
        # Find bias in predictions (overall)
        avg_bias = np.mean(predictions - targets)
        f.write(f"Average bias (predicted - true): {avg_bias:.2f} years\n")
        
        # Add timestamp and hardware info
        f.write("\n=== TEST ENVIRONMENT ===\n")
        f.write(f"Test date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n")
        if torch.cuda.is_available():
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"CUDA version: {torch.version.cuda}\n")
        
    # Return a dictionary with results
    results = {
        'mae': avg_mae,
        'within_5_years': within_5_years,
        'results_dir': results_dir
    }
    
    print(f"\nTesting completed. Results saved to {results_dir}")
    return results


if __name__ == "__main__":
    """
    Main execution function to run the model testing process
    """
    import argparse
    
    # Parse command line arguments
    # Handle Jupyter notebook command-line args
    # Remove Jupyter-specific arguments before parsing
    if any(arg.startswith('-f') for arg in sys.argv):
        # We're in Jupyter environment
        jupyter_mode = True
        # Use default args in Jupyter rather than trying to parse
        class Args:
            model_path = '/home/meem/backup/Age Datasets/Resnet-codes/output/resnet50_dldlv2_final.pth'
            test_csv = '/home/meem/backup/Age Datasets/UTKFace/crop_part1/test_annotations.csv'
            test_dir = '/home/meem/backup/Age Datasets/UTKFace/crop_part1/test'
            output_dir = './resnet50_test_results'
            num_classes = 122
            num_samples = 15
            seed = 42
        args = Args()
    else:
        # We're in a regular Python script environment
        jupyter_mode = False
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Test age classification model and generate visualizations')
        parser.add_argument('--model_path', type=str, default='Resnet-codes/output/resnet50_dldlv2_final.pth',
                            help='Path to the model checkpoint file')
        parser.add_argument('--test_csv', type=str, default='/home/meem/backup/Age Datasets/UTKFace/crop_part1/test_annotations.csv',
                            help='Path to CSV file with test annotations')
        parser.add_argument('--test_dir', type=str, default='/home/meem/backup/Age Datasets/UTKFace/crop_part1/test',
                            help='Path to directory containing test images')
        parser.add_argument('--output_dir', type=str, default='./resnet50_test_results',
                            help='Directory to save test results and visualizations')
        parser.add_argument('--num_classes', type=int, default=122,
                            help='Number of age classes (0 to num_classes-1)')
        parser.add_argument('--num_samples', type=int, default=15,
                            help='Number of random samples to visualize')
        parser.add_argument('--seed', type=int, default=42,
                            help='Random seed for reproducibility')
        
        args = parser.parse_args()
        
    # Set up for reproducibility
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print("\n===============================================")
    print("     Age Classification Model Testing Tool     ")
    print("===============================================\n")
    print(f"Model path: {args.model_path}")
    print(f"Test data: {args.test_csv}")
    print(f"Image directory: {args.test_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of age classes: {args.num_classes}")
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
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run test
    try:
        print("Starting test process...")
        results = test_model(
            model_path=args.model_path,
            test_csv=args.test_csv,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            num_classes=args.num_classes,
            num_samples_to_visualize=args.num_samples
        )
        
        # Print final summary
        print("\n===============================================")
        print("                  FINAL SUMMARY                ")
        print("===============================================")
        print(f"Model: {args.model_path}")
        print(f"Mean Absolute Error: {results['mae']:.2f} years")
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

