#!/usr/bin/env python
# coding: utf-8

# In[1]:
"""
This code works for all volo models except volo trained with DLDLv2 loss function
"""

# Inside your test_classification_model function, add this code where you create the scatter plot
# (or create a new function to generate this specific visualization)

def create_complete_age_prediction_scatter(predictions, targets, results_dir):
    """
    Create a scatter plot showing ALL data points of predicted vs true ages
    """
    plt.figure(figsize=(6, 5))
    
    # Create the scatter plot with smaller, more transparent points to avoid overplotting
    plt.scatter(targets, predictions, alpha=0.3, s=15, color='#3498db', edgecolor='none')
    
    # Add the reference lines
    max_val = max(np.max(targets), np.max(predictions))
    min_val = min(np.min(targets), np.min(predictions))
    
    # Perfect prediction line
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    # Linear regression line
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    X = targets.reshape(-1, 1)
    y = predictions
    model.fit(X, y)
    slope = model.coef_[0]
    plt.plot([min_val, max_val], 
             [model.predict([[min_val]])[0], model.predict([[max_val]])[0]], 
             'g-', linewidth=2, 
             label=f'Actual fit (slope={slope:.2f})')
    
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
    plt.annotate(txt, xy=(0.98, 0.10), xycoords='axes fraction', 
                va='top', ha='left', fontsize=6, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add count information to the plot
    plt.text(0.98, 0.15, f"Total samples: {len(predictions)}", fontsize=6, transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Style the plot
    plt.grid(True, alpha=0.3)
    plt.xlabel('True Age')
    plt.ylabel('Predicted Age')
    plt.title('Age Prediction Results (All Data Points)')
    plt.legend(loc='upper left')
    
    # Make sure we show the full data range
    buffer = 5
    plt.xlim(min_val - buffer, max_val + buffer)
    plt.ylim(min_val - buffer, max_val + buffer)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'complete_age_prediction_scatter.png'), dpi=300)
    plt.close()
    
   


# In[4]:


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Generate timestamp for file naming
timestamp = datetime.now().strftime("-%Y-%m-%d_%H-%M-%S")

# Dataset class for classification
class AgeClassificationDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to CSV with annotations (image_name, age)
            img_dir (string): Path to the images
            transform (callable, optional): Optional transform to be applied
        """
        self.age_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.age_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.age_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        age = self.age_frame.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        
        # For classification, return age as an integer (class label)
        age = int(age)  # Ensure age is an integer for classification
        
        return image, age, self.age_frame.iloc[idx, 0]  # Return filename as well for visualization

# Setup VOLO model with classification head
def setup_volo_model(checkpoint_path, num_classes=122):
    import sys
    import os
    
    # First make sure the VOLO directory is in the Python path
    volo_dir = os.path.join(os.getcwd(), 'volo')
    if os.path.exists(volo_dir) and volo_dir not in sys.path:
        sys.path.append(volo_dir)
    
    # Now try to import from the volo directory
    try:
        from volo.models import volo_d1
        from volo.utils import load_pretrained_weights
    except ImportError:
        # If that fails, try direct import (assuming we're in the volo directory)
        try:
            from models import volo_d1
            from utils import load_pretrained_weights
        except ImportError:
            raise ImportError(
                "Could not import VOLO modules. Please make sure you're either:\n"
                "1. Running from the VOLO directory, or\n"
                "2. Have the VOLO directory in your current working directory\n"
                "Current directory: " + os.getcwd()
            )
    
    # Load the base model with ImageNet weights
    model = volo_d1(img_size=224)
    
    # Load pretrained weights
    load_pretrained_weights(
        model=model,
        checkpoint_path=checkpoint_path,
        use_ema=False,
        strict=False,
        num_classes=num_classes
    )
    
    # Replace the classification head for age classification
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    if hasattr(model, 'aux_head'):
        model.aux_head = nn.Linear(model.aux_head.in_features, num_classes)
    
    return model

# Load model from trained checkpoint
def load_model(model_path, device='cpu', num_classes=122):
    """
    Load the trained model weights
    
    Args:
        model_path: Path to the trained model checkpoint
        device: Device to load the model on ('cuda' or 'cpu')
        num_classes: Number of age classes
        
    Returns:
        Loaded model ready for inference
    """
    # Create base model
    base_model = setup_volo_model(
        checkpoint_path='/home/meem/backup/d1_224_84.2.pth.tar',
        num_classes=num_classes
    )
    
    # Load the trained weights
    print(f"Loading trained model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        base_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If the checkpoint is just the state dict
        base_model.load_state_dict(checkpoint)
    
    base_model = base_model.to(device)
    base_model.eval()
    
    return base_model

# Run inference and evaluation for classification with detailed metrics
def test_classification_model(model_path, model, test_loader, device='cpu', visualize=True, num_samples=10, max_age=100):
    """
    Test the classification model on a dataset and compute detailed metrics
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        device: Device to test on ('cuda' or 'cpu')
        visualize: Whether to visualize predictions
        num_samples: Number of random samples to visualize
        max_age: Maximum age to consider for visualizations
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_filenames = []
    all_errors = []  # MAE for each sample
    all_probabilities = []  # Store softmax outputs
    
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    
    with torch.no_grad():
        for inputs, ages, filenames in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            target_ages = ages.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Handle different return formats of VOLO
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Main classification output
            
            # Calculate probabilities
            probabilities = F.softmax(outputs, dim=1)
            
            # Calculate loss
            loss = criterion(outputs, target_ages)
            running_loss += loss.item() * inputs.size(0)
            
            # Get predictions
            _, predicted_classes = torch.max(outputs, 1)
            correct += (predicted_classes == target_ages).sum().item()
            
            # Calculate MAE (treating predictions as continuous values)
            mae = torch.abs(predicted_classes.float() - target_ages.float())
            
            # Save data for detailed analysis
            all_preds.extend(predicted_classes.cpu().numpy())
            all_targets.extend(target_ages.cpu().numpy())
            all_filenames.extend(filenames)
            all_errors.extend(mae.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays for easier processing
    predictions = np.array(all_preds)
    targets = np.array(all_targets)
    errors = np.array(all_errors)
    probabilities = np.array(all_probabilities)
    
    # Calculate overall metrics
    avg_loss = running_loss / len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    avg_mae = np.mean(errors)
    
    # Calculate accuracy within different thresholds
    within_1_year = np.mean(errors <= 1) * 100
    within_3_years = np.mean(errors <= 3) * 100
    within_5_years = np.mean(errors <= 5) * 100
    within_10_years = np.mean(errors <= 10) * 100
    
    # Print overall metrics
    print(f"Test Results:")
    print(f"Avg Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Avg MAE: {avg_mae:.2f} years")
    print(f"Predictions within 1 year: {within_1_year:.2f}%")
    print(f"Predictions within 3 years: {within_3_years:.2f}%")
    print(f"Predictions within 5 years: {within_5_years:.2f}%")
    print(f"Predictions within 10 years: {within_10_years:.2f}%")
    
    # Calculate metrics by age group
    age_groups = [
        (0, 12, "Children (0-12)"),
        (13, 19, "Teenagers (13-19)"),
        (20, 35, "Young Adults (20-35)"),
        (36, 50, "Middle-aged (36-50)"),
        (51, 70, "Seniors (51-70)"),
        (71, 100, "Elderly (71+)")
    ]
    
    print("\nMetrics by Age Group:")
    for start, end, name in age_groups:
        mask = (targets >= start) & (targets <= end)
        if np.sum(mask) > 0:
            group_accuracy = np.mean(targets[mask] == predictions[mask])
            group_mae = np.mean(errors[mask])
            group_within_5 = np.mean(errors[mask] <= 5) * 100
            count = np.sum(mask)
            print(f"{name}: Accuracy: {group_accuracy:.2%}, MAE: {group_mae:.2f} years, Within 5 years: {group_within_5:.2f}% (n={count})")
    
    # Find worst predictions
    worst_indices = np.argsort(errors)[-10:][::-1]  # Top 10 worst predictions
    
    print("\nWorst Predictions:")
    for idx in worst_indices:
        print(f"File: {all_filenames[idx]}, True Age: {targets[idx]}, Predicted: {predictions[idx]}, Error: {errors[idx]} years")
    
    # Visualize predictions if requested
    if visualize:
        # Create a directory for results
        results_dir = f"age_classification_results{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Set common figure parameters for better viewing
        plt.rcParams.update({
            'font.size': 10,  # Smaller font size
            'figure.dpi': 100,  # Lower DPI for smaller file sizes
            'savefig.dpi': 150  # Higher DPI for saving but not too high
        })
        
        # 1. Plot a full confusion matrix (no bins) - REDUCED SIZE
        # Limit to ages up to max_age for better visualization
        cm_limit = min(max_age + 1, max(np.max(targets), np.max(predictions)) + 1)
        
        plt.figure(figsize=(10, 8))  # Reduced from (20, 16)
        cm = confusion_matrix(
            targets[targets < cm_limit], 
            predictions[targets < cm_limit], 
            labels=range(cm_limit)
        )
        
        # For better visibility, apply log scale to the confusion matrix
        # Use log(x+1) to handle zeros
        cm_log = np.log1p(cm)
        
        # Create a heatmap
        ax = sns.heatmap(cm_log, cmap='viridis', annot=False, 
                         xticklabels=10, yticklabels=10)
        
        # Set labels for every 10th tick
        tick_positions = np.arange(0, cm_limit, 10)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_positions)
        ax.set_yticklabels(tick_positions)
        
        plt.xlabel('Predicted Age')
        plt.ylabel('True Age')
        plt.title('Age Confusion Matrix (Log Scale)')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'confusion_matrix_full_log.png'))
        plt.close()
        
        # 2. Create a heatmap of prediction error by age - REDUCED SIZE
        # First, calculate average error for each true age
        max_target_age = min(max_age, int(np.max(targets)))
        avg_error_by_age = np.zeros(max_target_age + 1)
        count_by_age = np.zeros(max_target_age + 1)
        
        for true_age, error in zip(targets, errors):
            if true_age <= max_target_age:
                avg_error_by_age[int(true_age)] += error
                count_by_age[int(true_age)] += 1
        
        # Avoid division by zero
        mask = count_by_age > 0
        avg_error_by_age[mask] = avg_error_by_age[mask] / count_by_age[mask]
        
        plt.figure(figsize=(8, 4))  # Reduced from (15, 6)
        plt.bar(range(max_target_age + 1), avg_error_by_age, alpha=0.7)
        plt.xlabel('True Age')
        plt.ylabel('Average Error (years)')
        plt.title('Average Prediction Error by Age')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'avg_error_by_age.png'))
        plt.close()
        
        # 3. NEW: Add cumulative error distribution chart
        plt.figure(figsize=(8, 5))
        error_thresholds = np.arange(0, 21)  # 0 to 20 years
        cumulative_pct = [np.mean(errors <= threshold) * 100 for threshold in error_thresholds]
        
        plt.plot(error_thresholds, cumulative_pct, marker='o', markersize=4)
        plt.axhline(y=90, color='r', linestyle='--', alpha=0.7, label='90% threshold')
        plt.axvline(x=5, color='g', linestyle='--', alpha=0.7, label='5-year threshold')
        
        # Mark the percentage of predictions within 5 years
        five_year_pct = np.mean(errors <= 5) * 100
        plt.plot(5, five_year_pct, 'ro', markersize=8)
        plt.annotate(f"{five_year_pct:.1f}%", 
                     xy=(5, five_year_pct), 
                     xytext=(7, five_year_pct - 5),
                     arrowprops=dict(arrowstyle="->", color='r'))
        
        plt.xlabel('Error Threshold (years)')
        plt.ylabel('Percentage of Predictions (%)')
        plt.title('Cumulative Error Distribution')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'cumulative_error_distribution.png'))
        plt.close()
        
        # 4. NEW: Add pie chart for error thresholds
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
        plt.savefig(os.path.join(results_dir, f'error_distribution_pie.png'))
        plt.close()
        
        # 5. NEW: Add bar chart showing percentage of predictions within 5 years by age group
        plt.figure(figsize=(8, 5))
        
        # Calculate percentage within 5 years for each age group
        age_group_labels = []
        within_5_years_pct = []
        
        for start, end, name in age_groups:
            mask = (targets >= start) & (targets <= end)
            if np.sum(mask) > 0:
                pct = np.mean(errors[mask] <= 5) * 100
                within_5_years_pct.append(pct)
                age_group_labels.append(name)
        
        # Add overall percentage
        age_group_labels.append("Overall")
        within_5_years_pct.append(within_5_years)
        
        # Create bar chart
        bars = plt.bar(range(len(age_group_labels)), within_5_years_pct, color='#3498db')
        
        # Add data labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{within_5_years_pct[i]:.1f}%',
                    ha='center', va='bottom', rotation=0)
        
        plt.axhline(y=within_5_years, color='r', linestyle='--', alpha=0.7, label='Overall Average')
        plt.xticks(range(len(age_group_labels)), age_group_labels, rotation=45, ha='right')
        plt.ylabel('Percentage within 5 years (%)')
        plt.title('Predictions Within 5 Years by Age Group')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'within_5_years_by_age_group.png'))
        plt.close()
        
        # 6. NEW: Add line chart showing accuracy by exact error threshold
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
        plt.savefig(os.path.join(results_dir, f'accuracy_by_threshold.png'))
        plt.close()
        
        # 7. Create a distance heatmap - REDUCED SIZE
        # Calculate prediction distances
        distances = predictions - targets
        max_distance = min(40, max(abs(np.min(distances)), np.max(distances)))
        
        # Create a 2D histogram of true ages vs prediction distance
        hist_data = np.zeros((max_target_age + 1, 2 * max_distance + 1))
        
        for true_age, distance in zip(targets, distances):
            if true_age <= max_target_age and abs(distance) <= max_distance:
                hist_data[int(true_age), int(distance) + max_distance] += 1
        
        # Normalize by the number of samples at each age
        for age in range(max_target_age + 1):
            if np.sum(hist_data[age, :]) > 0:
                hist_data[age, :] = hist_data[age, :] / np.sum(hist_data[age, :])
        
        plt.figure(figsize=(10, 8))  # Reduced from (20, 16)
        sns.heatmap(hist_data, cmap='viridis', 
                   xticklabels=10, yticklabels=10)
        
        # Set custom x-axis ticks showing the prediction distance
        tick_positions = np.linspace(0, 2 * max_distance, 9)
        tick_labels = np.linspace(-max_distance, max_distance, 9).astype(int)
        plt.xticks(tick_positions, tick_labels)
        
        # Add vertical lines at +/- 5 years
        center_idx = max_distance
        plt.axvline(x=center_idx + 5, color='r', linestyle='--', alpha=0.7)
        plt.axvline(x=center_idx - 5, color='r', linestyle='--', alpha=0.7)
        
        # Set y-axis ticks for true age
        y_tick_positions = np.arange(0, max_target_age + 1, 10)
        plt.yticks(y_tick_positions, y_tick_positions)
        
        plt.xlabel('Prediction Distance (Predicted - True)')
        plt.ylabel('True Age')
        plt.title('Age Prediction Distance Heatmap (Normalized)')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'age_distance_heatmap.png'))
        plt.close()
        
        # 8. Plot a random subset of predictions - REDUCED SIZE
        if num_samples > 0:
            sample_indices = np.random.choice(len(all_preds), min(num_samples, len(all_preds)), replace=False)
            visualize_classification_predictions(
                [all_filenames[i] for i in sample_indices],
                [targets[i] for i in sample_indices],
                [predictions[i] for i in sample_indices],
                test_loader.dataset.img_dir,
                results_dir
            )
        
        # 9. Plot absolute error vs. true age as a scatter plot with density - REDUCED SIZE
        plt.figure(figsize=(8, 6))  # Reduced from (12, 8)
        
        # Create hexbin plot for density visualization
        hb = plt.hexbin(targets, errors, gridsize=25, cmap='Blues', mincnt=1)
        plt.colorbar(hb, label='Count')
        
        # Add a horizontal line at error = 5
        plt.axhline(y=5, color='r', linestyle='--', alpha=0.7, label='5-year threshold')
        
        plt.xlabel('True Age')
        plt.ylabel('Absolute Error (years)')
        plt.title('Prediction Error vs True Age')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'error_vs_age_density.png'))
        plt.close()
        
        # 10. Plot the probability distribution for selected ages - REDUCED SIZE
        # Choose a few interesting ages to visualize
        ages_to_visualize = [5, 20, 40, 60, 80]
        plt.figure(figsize=(8, 6))  # Reduced from (15, 10)
        
        for age in ages_to_visualize:
            # Find examples with the true age
            indices = np.where(targets == age)[0]
            if len(indices) > 0:
                # Average the probability distributions
                avg_probs = np.mean(probabilities[indices], axis=0)
                
                # Plot probabilities for a range around the true age
                window = 40
                start_idx = max(0, age - window)
                end_idx = min(len(avg_probs), age + window + 1)
                x_range = np.arange(start_idx, end_idx)
                
                plt.plot(x_range, avg_probs[start_idx:end_idx], label=f'Age {age}')
                plt.axvline(x=age, linestyle='--', alpha=0.5, color='gray')
        
        plt.xlabel('Age Class')
        plt.ylabel('Average Probability')
        plt.title('Average Probability Distribution for Selected Ages')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'probability_distributions.png'))
        plt.close()
    
    # Save detailed results to CSV
    results_df = pd.DataFrame({
        'filename': all_filenames,
        'true_age': targets,
        'predicted_age': predictions,
        'absolute_error': errors,
        'within_5_years': errors <= 5  # Add a binary column for within 5 years
    })
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    
    results_df.to_csv(os.path.join(results_dir, 'classification_results.csv'), index=False)
    print(f"Saved detailed results to {results_dir}/classification_results.csv")
    
    # Generate a summary statistics file
    with open(os.path.join(results_dir, 'summary_stats.txt'), 'w') as f:
        f.write("SUMMARY STATISTICS\n")
        f.write("=================\n\n")
        f.write(f"Model Name: {model_path}\n")
        f.write("=================\n\n")
        f.write(f"Test dataset size: {len(test_loader.dataset)} images\n")
        f.write(f"Mean Absolute Error: {avg_mae:.2f} years\n")
        f.write(f"Accuracy (exact match): {accuracy:.2%}\n\n")
        
        f.write("PREDICTIONS WITHIN ERROR THRESHOLDS\n")
        f.write("==================================\n")
        f.write(f"Within 1 year: {within_1_year:.2f}%\n")
        f.write(f"Within 3 years: {within_3_years:.2f}%\n")
        f.write(f"Within 5 years: {within_5_years:.2f}%\n")
        f.write(f"Within 10 years: {within_10_years:.2f}%\n\n")
        
        f.write("METRICS BY AGE GROUP\n")
        f.write("===================\n")
        for start, end, name in age_groups:
            mask = (targets >= start) & (targets <= end)
            if np.sum(mask) > 0:
                group_accuracy = np.mean(targets[mask] == predictions[mask])
                group_mae = np.mean(errors[mask])
                group_within_5 = np.mean(errors[mask] <= 5) * 100
                count = np.sum(mask)
                f.write(f"{name}:\n")
                f.write(f"  - Sample size: {count}\n")
                f.write(f"  - Accuracy: {group_accuracy:.2%}\n")
                f.write(f"  - MAE: {group_mae:.2f} years\n")
                f.write(f"  - Within 5 years: {group_within_5:.2f}%\n\n")
    
    print(f"Saved summary statistics to {results_dir}/summary_stats.txt")
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'mae': avg_mae,
        'within_5_years': within_5_years,
        'predictions': predictions,
        'targets': targets,
        'filenames': all_filenames,
        'errors': errors,
        'probabilities': probabilities,
        'results_dir': results_dir
    }

# Helper function to visualize classification predictions - REDUCED SIZE
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
    
    plt.figure(figsize=(n_cols * 2, n_rows * 2))  # Reduced from (n_cols * 3, n_rows * 4)
    
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
        plt.title(f"True: {true_age}\nPred: {pred_age}\n{within_5}", color=color, fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_predictions.png'))
    plt.close()

# Main function for testing
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up for reproducibility
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# Set up for reproducibility
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Define paths
    test_csv = '/home/meem/backup/Age Datasets/test_annotations.csv'
    test_dir = '/home/meem/backup/Age Datasets/test'
    
    # Use your trained model file
    model_path = 'output/volo_d1_gradual_final.pth'  # Change to your model path

    # model_path = 'Resnet-codes/output/resnet50_crossentropy_final.pth'  # Change to your model path
    
    
    # Test transforms (same as validation)
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset and dataloader
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
    
    print(f"Test dataset size: {len(test_dataset)} images")
    
    # Load model
    model = load_model(model_path, device, num_classes=122)
    
    # Test model
    results = test_classification_model(
        model_path = model_path,
        model=model, 
        test_loader=test_loader, 
        device=device,
        visualize=True,
        num_samples=15,
        max_age=100  # Limit visualization to ages 0-100
    )

    # Call this function after your test_classification_model function
    create_complete_age_prediction_scatter(results['predictions'], results['targets'], results['results_dir'])

    
    # Print key results
    print("\nKEY RESULTS SUMMARY:")
    print(f"Mean Absolute Error: {results['mae']:.2f} years")
    print(f"Percentage within 5 years: {results['within_5_years']:.2f}%")
    print(f"All results saved in directory: {results['results_dir']}")

if __name__ == "__main__":
    main()

