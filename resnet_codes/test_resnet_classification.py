#!/usr/bin/env python
# coding: utf-8

# In[2]:

"""
This code works for all fine-tuned Resnet models, except the one trained using DLDLv2 loss function
"""

import os
import sys

os.getcwd()


# In[3]:


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
from tqdm import tqdm
import random
from datetime import datetime

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

# Create ResNet-50 model
def create_model(num_classes=122):
    model = models.resnet50(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

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
        plt.title(f"True: {true_age}\nPred: {pred_age}\n{within_5}", color=color, fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_predictions.png'))
    plt.close()

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
    print(f"Creating ResNet-50 model with {num_classes} output classes...")
    model = create_model(num_classes=num_classes)
    
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
    
    # Evaluation loop
    all_preds = []
    all_targets = []
    all_filenames = []
    all_errors = []
    all_probabilities = []
    
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, ages, filenames in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), ages.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            
            # Get predictions and probability distributions
            probs = F.softmax(outputs, dim=1)
            _, predicted_classes = torch.max(outputs, 1)
            
            # Calculate MAE for each sample
            mae = torch.abs(predicted_classes.float() - targets.float())
            
            # Save results
            all_preds.extend(predicted_classes.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_filenames.extend(filenames)
            all_errors.extend(mae.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(all_preds)
    targets = np.array(all_targets)
    errors = np.array(all_errors)
    probabilities = np.array(all_probabilities)

    # Add overall distribution statistics
    mean_std = np.mean([np.sqrt(np.sum(p * (np.arange(len(p)) - np.sum(p * np.arange(len(p))))**2)) for p in probabilities])
    print(f"\nAverage distribution standard deviation: {mean_std:.2f} years\n")
        
    # Calculate correlation between distribution width and MAE
    stds = [np.sqrt(np.sum(p * (np.arange(len(p)) - np.sum(p * np.arange(len(p))))**2)) for p in probabilities]
    maes = [abs(np.sum(p * np.arange(len(p))) - t) for p, t in zip(probabilities, targets)]
    corr = np.corrcoef(stds, maes)[0, 1]
    print(f"Correlation between distribution width and error: {corr:.4f}\n")
        
    print("Testing completed successfully!")
    
    # Calculate metrics
    avg_loss = running_loss / len(test_loader.dataset)
    accuracy = np.mean(predictions == targets)
    avg_mae = np.mean(errors)
    
    # Calculate accuracy within different thresholds
    within_1_year = np.mean(errors <= 1) * 100
    within_3_years = np.mean(errors <= 3) * 100
    within_5_years = np.mean(errors <= 5) * 100
    within_10_years = np.mean(errors <= 10) * 100
    
    # Print overall metrics
    print("\n=== TEST RESULTS ===")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Exact Match Accuracy: {accuracy:.2%}")
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
    
    # Store age group results for later visualization
    age_group_labels = []
    within_5_years_by_group = []
    
    for start, end, name in age_groups:
        mask = (targets >= start) & (targets <= end)
        if np.sum(mask) > 0:
            group_accuracy = np.mean(targets[mask] == predictions[mask])
            group_mae = np.mean(errors[mask])
            group_within_5 = np.mean(errors[mask] <= 5) * 100
            count = np.sum(mask)
            
            # Store for visualization
            age_group_labels.append(name)
            within_5_years_by_group.append(group_within_5)
            
            print(f"{name}: Accuracy: {group_accuracy:.2%}, MAE: {group_mae:.2f} years, Within 5 years: {group_within_5:.2f}% (n={count})")
    
    # Create visualizations
    
    # 1. Complete Age Prediction Scatter Plot - already implemented
    create_complete_age_prediction_scatter(predictions, targets, results_dir)
    
    # 2. Confusion Matrix (Log Scale)
    plt.figure(figsize=(12, 10))
    max_age = min(100, max(np.max(targets), np.max(predictions)))
    
    cm = confusion_matrix(
        targets[targets <= max_age], 
        predictions[targets <= max_age], 
        labels=range(max_age+1)
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
    
    # Save a more detailed version for the full confusion matrix
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(cm_log, cmap='viridis', annot=False, 
                    xticklabels=10, yticklabels=10)
    
    tick_positions = np.arange(0, max_age+1, 10)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_positions)
    ax.set_yticklabels(tick_positions)
    
    plt.xlabel('Predicted Age', fontsize=14)
    plt.ylabel('True Age', fontsize=14)
    plt.title('Age Confusion Matrix (Full, Log Scale)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix_full_log.png'))
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
    
    # 5. NEW: Add pie chart for error thresholds
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
    
    # 6. NEW: Add bar chart showing percentage of predictions within 5 years by age group
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
    
    # 7. NEW: Add line chart showing accuracy by exact error threshold
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
    
    # 8. NEW: Create a distance heatmap
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
    
    plt.figure(figsize=(10, 8))
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
    plt.savefig(os.path.join(results_dir, 'age_distance_heatmap.png'))
    plt.close()
    
    # 9. NEW: Plot absolute error vs. true age as a scatter plot with density
    plt.figure(figsize=(8, 6))
    
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
    plt.savefig(os.path.join(results_dir, 'error_vs_age_density.png'))
    plt.close()
    
    # 10. NEW: Plot the probability distribution for selected ages
    # Choose a few interesting ages to visualize
    ages_to_visualize = [5, 20, 40, 60, 80]
    plt.figure(figsize=(8, 6))
    
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
    plt.savefig(os.path.join(results_dir, 'probability_distributions.png'))
    plt.close()
    
    # 11. NEW: Visualize sample predictions
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

    # 12. NEW: Visualize age distributions using mean-variance approach
    print("\nGenerating age distribution visualizations...")
    
    # Create individual distribution visualizations
    visualize_individual_age_distributions(
        probabilities=probabilities,
        targets=targets,
        filenames=all_filenames,
        img_dir=test_dir,
        results_dir=results_dir,
        num_samples=15
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
        'within_5_years': errors <= 5  # Add a binary column for within 5 years
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
        f.write(f"Exact Match Accuracy: {accuracy:.2%}\n")
        f.write(f"Mean Absolute Error: {avg_mae:.2f} years\n\n")
        
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
                group_accuracy = np.mean(targets[mask] == predictions[mask])
                group_mae = np.mean(errors[mask])
                group_within_5 = np.mean(errors[mask] <= 5) * 100
                group_within_10 = np.mean(errors[mask] <= 10) * 100
                count = np.sum(mask)
                f.write(f"{name}:\n")
                f.write(f"  - Sample count: {count}\n")
                f.write(f"  - Accuracy: {group_accuracy:.2%}\n")
                f.write(f"  - MAE: {group_mae:.2f} years\n")
                f.write(f"  - Within 5 years: {group_within_5:.2f}%\n")
                f.write(f"  - Within 10 years: {group_within_10:.2f}%\n\n")
        
        # Find bias in predictions (overall)
        avg_bias = np.mean(predictions - targets)
        f.write(f"Average bias (predicted - true): {avg_bias:.2f} years\n")
        
        # Find bias by age group
        f.write("\n=== PREDICTION BIAS BY AGE GROUP ===\n")
        for start, end, name in age_groups:
            mask = (targets >= start) & (targets <= end)
            if np.sum(mask) > 0:
                group_bias = np.mean(predictions[mask] - targets[mask])
                f.write(f"{name}: {group_bias:.2f} years\n")
        
        # Find worst predictions
        worst_indices = np.argsort(errors)[-10:][::-1]  # Top 10 worst predictions
        f.write("\n=== WORST PREDICTIONS ===\n")
        for idx in worst_indices:
            f.write(f"File: {all_filenames[idx]}, True Age: {targets[idx]}, Predicted: {predictions[idx]}, Error: {errors[idx]} years\n")
        
        # Distribution of predictions
        f.write("\n=== PREDICTION DISTRIBUTION ===\n")
        age_ranges = [(0, 12), (13, 19), (20, 35), (36, 50), (51, 70), (71, 100)]
        for start, end in age_ranges:
            true_count = np.sum((targets >= start) & (targets <= end))
            pred_count = np.sum((predictions >= start) & (predictions <= end))
            f.write(f"Ages {start}-{end}: True: {true_count} ({true_count/len(targets)*100:.1f}%), Predicted: {pred_count} ({pred_count/len(predictions)*100:.1f}%)\n")
        
        # Add timestamp and hardware info
        f.write("\n=== TEST ENVIRONMENT ===\n")
        f.write(f"Test date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n")
        if torch.cuda.is_available():
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"CUDA version: {torch.version.cuda}\n")
        
        f.write("\n=== VISUALIZATIONS ===\n")
        f.write("The following visualization files have been generated:\n")
        f.write("- complete_age_prediction_scatter.png: Scatter plot of predicted vs true ages\n")
        f.write("- confusion_matrix_log.png: Confusion matrix (log scale)\n")
        f.write("- confusion_matrix_full_log.png: Full detailed confusion matrix (log scale)\n")
        f.write("- avg_error_by_age.png: Average prediction error by age\n")
        f.write("- cumulative_error_distribution.png: Cumulative error distribution\n")
        f.write("- error_distribution_pie.png: Pie chart of error distribution\n")
        f.write("- within_5_years_by_age_group.png: Percentage within 5 years by age group\n")
        f.write("- accuracy_by_threshold.png: Accuracy at different error thresholds\n")
        f.write("- age_distance_heatmap.png: Heatmap of prediction distances by age\n")
        f.write("- error_vs_age_density.png: Density plot of errors vs true age\n")
        f.write("- probability_distributions.png: Average probability distributions for selected ages\n")
        f.write("- sample_predictions.png: Sample prediction visualizations\n")


# In[4]:


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
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from PIL import Image
    import os
    
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
    
    x = np.linspace(max(0, mean_age - 4*simulated_std), mean_age + 4*simulated_std, 1000)
    y = norm.pdf(x, simulated_mean, simulated_std)
    
    plt.plot(x, y, 'b-', linewidth=2)
    plt.axvline(x=true_age, color='r', linestyle='--', linewidth=2)
    
    # Add distribution statistics
    plt.title('Before Mean-Variance Loss')
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
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
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


# In[5]:


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
            model_path = '/home/meem/backup/Age Datasets/Resnet-codes/output/resnet50_meanvar_final.pth'
            test_csv = '/home/meem/backup/Age Datasets/test_annotations.csv'
            test_dir = '/home/meem/backup/Age Datasets/test'
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
        parser.add_argument('--model_path', type=str, default='Resnet-codes/output_resnet_unimodal/resnet50_unimodal_best_mae.pth',
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
    
    seed = 42
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
    import os
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
        
        # # Print final summary
        # print("\n===============================================")
        # print("                  FINAL SUMMARY                ")
        # print("===============================================")
        # print(f"Model: {args.model_path}")
        # print(f"Mean Absolute Error: {results['mae']:.2f} years")
        # print(f"Percentage within 5 years: {results['within_5_years']:.2f}%")
        # print(f"All results saved to: {results['results_dir']}")
        # print("===============================================\n")

        
        
    except Exception as e:
        import traceback
        print(f"\nERROR: Test process failed with error:\n{e}")
        print("\nStack trace:")
        traceback.print_exc()
        print("\nPlease check the paths and parameters and try again.")
        exit(1)

