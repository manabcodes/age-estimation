#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import warnings
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
import scipy


# In[40]:


def visualize_raw_vs_scalar(df, metrics, output_dir):
    """Visualize the relationship between raw metrics and their scalar versions"""
    # Check which metrics have scalar versions
    scalar_metrics = [f"{metric}.scalar" for metric in metrics 
                     if f"{metric}.scalar" in df.columns]
    
    if not scalar_metrics:
        print("No scalar versions found for metrics, skipping raw vs scalar visualization")
        return
    
    # Create a grid of scatter plots
    n_metrics = len(scalar_metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # Reduce figsize for MacBook Air
    plt.figure(figsize=(16, n_rows * 5))
    
    for i, scalar_metric in enumerate(scalar_metrics, 1):
        metric = scalar_metric.replace('.scalar', '')
        
        # Check if we have at least 2 distinct values (required for trend line)
        if df[metric].nunique() <= 1 or df[scalar_metric].nunique() <= 1:
            continue
            
        plt.subplot(n_rows, n_cols, i)
        
        # Use smaller points for large datasets
        sample_size = min(3000, len(df))  # Limit number of points for large datasets
        if len(df) > sample_size:
            sample_df = df.sample(sample_size, random_state=42)
        else:
            sample_df = df
            
        plt.scatter(sample_df[metric], sample_df[scalar_metric], alpha=0.4, s=5, c='teal')
        
        try:
            # Safely try to add trend line, but continue if it fails
            if df[metric].nunique() > 1:
                # Ensure no NaN or Inf values that could cause issues
                valid_mask = np.isfinite(df[metric]) & np.isfinite(df[scalar_metric])
                if valid_mask.sum() > 2:  # Need at least 3 valid points
                    x_valid = df[metric][valid_mask]
                    y_valid = df[scalar_metric][valid_mask]
                    
                    try:
                        z = np.polyfit(x_valid, y_valid, 1)
                        p = np.poly1d(z)
                        plt.plot(np.linspace(x_valid.min(), x_valid.max(), 100), 
                                p(np.linspace(x_valid.min(), x_valid.max(), 100)), 
                                "r--", alpha=0.7)
                    except:
                        # If polyfit still fails, just continue without the trend line
                        pass
        except Exception:
            # Just continue without the trend line if there's an error
            pass
        
        plt.title(f'{metric} vs {scalar_metric}', fontsize=9)
        plt.xlabel(metric, fontsize=8)
        plt.ylabel(scalar_metric, fontsize=8)
        plt.tick_params(labelsize=7)
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'raw_vs_scalar.png'), dpi=200)
    plt.close()


# In[2]:


#!/usr/bin/env python
# coding: utf-8


# Suppress warnings
# warnings.filterwarnings('ignore')

# Define the columns we're interested in analyzing
QUALITY_METRICS = [
    'UnifiedQualityScore', 'BackgroundUniformity', 'IlluminationUniformity',
    'LuminanceMean', 'LuminanceVariance', 'UnderExposurePrevention',
    'OverExposurePrevention', 'DynamicRange', 'Sharpness',
    'CompressionArtifacts', 'NaturalColour', 'SingleFacePresent', 'EyesOpen',
    'MouthClosed,EyesVisible','MouthOcclusionPrevention','FaceOcclusionPrevention',
    'InterEyeDistance','HeadSize','LeftwardCropOfTheFaceImage',
    'RightwardCropOfTheFaceImage',
    'MarginAboveOfTheFaceImage',
    'MarginBelowOfTheFaceImage',
    'HeadPoseYaw','HeadPosePitch','HeadPoseRoll',
    'ExpressionNeutrality','NoHeadCoverings'
]

def get_scalar_metrics(metrics):
    """Convert regular metrics to scalar metrics"""
    return [f"{metric}.scalar" for metric in metrics]

def get_image_properties(image_path):
    """Extract image size and resolution from a given path"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            filesize = os.path.getsize(image_path) / 1024  # Size in KB
            return {
                'width': width,
                'height': height,
                'resolution': width * height,
                'filesize': filesize,
                'aspect_ratio': width / height
            }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return {
            'width': np.nan,
            'height': np.nan,
            'resolution': np.nan,
            'filesize': np.nan,
            'aspect_ratio': np.nan
        }

def prepare_data(csv_file):
    """Load data and prepare it for visualization"""
    # Load the data - using semicolon as separator
    df = pd.read_csv(csv_file, sep=';')
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Check if image properties calculation is needed and possible
    if 'Filename' in df.columns:
        print("Calculating image properties from filenames...")
        # Sample a few images to see if paths are valid
        sample_size = min(5, len(df))
        valid_paths = 0
        
        for idx, filename in enumerate(df['Filename'].head(sample_size)):
            try:
                with Image.open(filename) as img:
                    valid_paths += 1
            except Exception as e:
                print(f"Warning: Could not open image {idx+1}/{sample_size}: {filename}")
                print(f"Error: {e}")
        
        # Only calculate properties if at least one image path is valid
        if valid_paths > 0:
            print(f"Found {valid_paths}/{sample_size} valid image paths. Calculating properties...")
            # Create a dict to store image properties
            props = []
            
            # Process in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(df), batch_size):
                batch = df['Filename'].iloc[i:i+batch_size]
                batch_props = [get_image_properties(path) for path in batch]
                props.extend(batch_props)
                print(f"Processed {min(i+batch_size, len(df))}/{len(df)} images")
            
            # Add properties to dataframe
            for prop in ['width', 'height', 'resolution', 'filesize', 'aspect_ratio']:
                df[prop] = [p[prop] for p in props]
            
            print("Image properties calculated and added to dataframe")
        else:
            print("No valid image paths found. Skipping image properties calculation.")
    
    # Check which metrics are actually in the dataframe
    available_raw_metrics = [col for col in QUALITY_METRICS if col in df.columns]
    available_scalar_metrics = [f"{col}.scalar" for col in available_raw_metrics if f"{col}.scalar" in df.columns]
    
    print(f"Available raw metrics: {', '.join(available_raw_metrics)}")
    print(f"Available scalar metrics: {', '.join(available_scalar_metrics)}")
    
    return df, available_raw_metrics, available_scalar_metrics

def visualize_metric_distributions(df, metrics, output_dir, suffix=""):
    """
    Create distribution plots for each metric, handling large numbers of metrics by creating multiple figures.
    This version avoids memory issues by using memory-efficient histogram calculation.
    """
    # Determine how many metrics we're plotting
    n_metrics = len(metrics)
    print(f"Preparing to visualize distributions for {n_metrics} metrics")
    
    # Set up parameters for plotting
    metrics_per_figure = 12  # Maximum number of metrics to show in one figure
    n_cols = 4  # Number of columns in each figure grid
    
    # Calculate how many figures we need
    n_figures = (n_metrics + metrics_per_figure - 1) // metrics_per_figure
    
    # Process metrics in chunks of 12
    for fig_num in range(n_figures):
        # Determine which metrics go in this figure
        start_idx = fig_num * metrics_per_figure
        end_idx = min(start_idx + metrics_per_figure, n_metrics)
        current_metrics = metrics[start_idx:end_idx]
        current_count = len(current_metrics)
        
        # Calculate grid layout
        n_cols = min(4, current_count)
        n_rows = (current_count + n_cols - 1) // n_cols
        
        print(f"Creating figure {fig_num+1}/{n_figures} with {current_count} metrics (indices {start_idx} to {end_idx-1})")
        
        try:
            # Create the figure
            plt.figure(figsize=(16, n_rows * 3))
            
            for i, metric in enumerate(current_metrics):
                ax = plt.subplot(n_rows, n_cols, i+1)
                
                # Get data for the current metric
                data = df[metric].dropna().values
                
                # Calculate statistics
                mean_val = np.mean(data)
                median_val = np.median(data)
                std_val = np.std(data)
                
                try:
                    # Memory-efficient histogram calculation
                    max_bins = min(100, max(10, int(np.sqrt(len(data)))))
                    
                    # Create the histogram
                    hist, bin_edges = np.histogram(data, bins=max_bins, density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    bin_width = bin_edges[1] - bin_edges[0]
                    
                    # Plot histogram bars
                    ax.bar(bin_centers, hist, width=bin_width, alpha=0.6, color='steelblue')
                    
                    # Create KDE manually if we have enough data points
                    if len(data) > 5 and len(np.unique(data)) > 3:
                        try:
                            from scipy import stats
                            
                            # For very large datasets, sample to prevent memory issues
                            if len(data) > 10000:
                                sample_size = 10000
                                sample_indices = np.random.choice(len(data), sample_size, replace=False)
                                kde_data = data[sample_indices]
                            else:
                                kde_data = data
                                
                            kde = stats.gaussian_kde(kde_data)
                            x_range = np.linspace(np.min(data), np.max(data), 500)
                            kde_values = kde(x_range)
                            ax.plot(x_range, kde_values, color='navy', linewidth=2)
                        except Exception as e:
                            print(f"  KDE calculation failed for {metric}: {e}")
                except Exception as e:
                    print(f"  Histogram calculation failed for {metric}: {e}")
                    # Provide a fallback message
                    ax.text(0.5, 0.5, f"Could not create histogram for {metric}",
                           ha='center', va='center', transform=ax.transAxes)
                
                # Add vertical lines for mean and median
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle='-.', alpha=0.7, label=f'Median: {median_val:.2f}')
                
                # Add legend on the first plot only
                if i == 0:
                    ax.legend(fontsize='small')
                
                # Add statistics text box
                stats_text = (f"Mean: {mean_val:.2f}\n"
                             f"Median: {median_val:.2f}\n"
                             f"Std Dev: {std_val:.2f}")
                
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                # Format metric name for display
                display_name = metric.replace('.scalar', ' (scalar)') if '.scalar' in metric else metric
                ax.set_title(f'Distribution of {display_name}', fontsize=9)
                ax.tick_params(labelsize=8)
            
            plt.tight_layout()
            
            # Add figure number to filename if multiple figures
            figure_suffix = f"_{fig_num+1}" if n_figures > 1 else ""
            filename = f'metric_distributions{suffix}{figure_suffix}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=200)
            plt.close()
            print(f"✓ {filename} created")
            
        except Exception as e:
            print(f"Error creating figure {fig_num+1}: {e}")
            plt.close()
    
    print(f"Completed distribution visualizations for all {n_metrics} metrics")

def visualize_correlation_heatmap(df, metrics, output_dir, suffix=""):
    """Create a correlation heatmap for quality metrics"""
    # Calculate correlation matrix
    corr_matrix = df[metrics].corr()
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with custom diverging colormap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Adjust label display for better readability
    labels = [m.replace('.scalar', '\n(scalar)') if '.scalar' in m else m for m in metrics]
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, fmt=".2f", 
                annot_kws={"size": 7}, xticklabels=labels, yticklabels=labels)
    
    plt.title(f'Correlation Between Quality Metrics{" (Scalar)" if ".scalar" in metrics[0] else ""}', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    filename = f'correlation_heatmap{suffix}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close()
    print(f"✓ {filename} created")
    
    # Also identify and visualize the most strongly correlated metrics
    corr_unstack = corr_matrix.unstack()
    corr_unstack = corr_unstack[corr_unstack < 1.0]  # Remove self-correlations
    
    # Handle case where there might be too few correlations
    n_correlations = min(5, len(corr_unstack))
    if n_correlations == 0:
        print("Not enough correlations to visualize strongest pairs")
        return
        
    strongest_corrs = corr_unstack.abs().nlargest(n_correlations)
    
    plt.figure(figsize=(12, 3.5))
    for i, (pair, corr_value) in enumerate(strongest_corrs.items()):
        metric1, metric2 = pair
        plt.subplot(1, n_correlations, i+1)
        plt.scatter(df[metric1], df[metric2], alpha=0.5, s=5)
        
        # Format names for display
        display_name1 = metric1.replace('.scalar', ' (scalar)') if '.scalar' in metric1 else metric1
        display_name2 = metric2.replace('.scalar', ' (scalar)') if '.scalar' in metric2 else metric2
        
        plt.title(f'Corr: {corr_value:.2f}', fontsize=9)
        plt.xlabel(display_name1, fontsize=7)
        plt.ylabel(display_name2, fontsize=7)
        plt.tick_params(labelsize=6)
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    filename = f'strongest_correlations{suffix}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close()
    print(f"✓ {filename} created")

def visualize_metric_boxplots(df, metrics, output_dir, suffix=""):
    """
    Create boxplots for each quality metric to show distribution and outliers,
    along with individual frequency histograms for each metric showing
    mean, median, and standard deviation.
    """
    # Part 1: Create the boxplot
    plt.figure(figsize=(12, 7))
    
    # Melt the dataframe to get data in format for boxplot
    melted_df = pd.melt(df[metrics], var_name='Metric', value_name='Value')
    
    # Format metric names for display
    melted_df['Metric'] = melted_df['Metric'].apply(
        lambda x: x.replace('.scalar', ' (scalar)') if '.scalar' in x else x
    )
    
    # Create boxplot
    sns.boxplot(x='Metric', y='Value', data=melted_df)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.title(f'Distribution of Quality Metrics{" (Scalar)" if ".scalar" in metrics[0] else ""}', fontsize=11)
    plt.tight_layout()
    
    filename = f'metric_boxplots{suffix}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close()
    print(f"✓ {filename} created")


# def visualize_metric_distributions(df, metrics, output_dir, suffix=""):
#     """
#     Create distribution plots for each metric, with memory-efficient implementation
#     to handle large datasets and prevent memory errors.
#     """
#     # Determine how many metrics we're plotting
#     n_metrics = len(metrics)
#     print(f"Preparing to visualize distributions for {n_metrics} metrics")
    
#     # Set up parameters for plotting
#     metrics_per_figure = 12  # Maximum number of metrics to show in one figure
#     n_cols = 4  # Number of columns in each figure grid
#     n_rows_per_figure = (metrics_per_figure + n_cols - 1) // n_cols  # Rows needed per figure
    
#     # Calculate how many figures we need
#     n_figures = (n_metrics + metrics_per_figure - 1) // metrics_per_figure
    
#     # Process metrics in chunks
#     for fig_num in range(n_figures):
#         # Determine which metrics go in this figure
#         start_idx = fig_num * metrics_per_figure
#         end_idx = min(start_idx + metrics_per_figure, n_metrics)
#         current_metrics = metrics[start_idx:end_idx]
#         current_count = len(current_metrics)
        
#         # Calculate rows needed for this figure
#         n_rows = (current_count + n_cols - 1) // n_cols
        
#         print(f"Creating figure {fig_num+1}/{n_figures} with {current_count} metrics (indices {start_idx} to {end_idx-1})")
        
#         try:
#             # Create the figure
#             plt.figure(figsize=(16, n_rows * 3))
            
#             for i, metric in enumerate(current_metrics):
#                 ax = plt.subplot(n_rows, n_cols, i+1)
                
#                 # Get data for this metric only
#                 data = df[metric].dropna()
                
#                 # Calculate statistics directly (more memory efficient)
#                 mean_val = data.mean()
#                 median_val = data.median()
#                 std_val = data.std()
#                 min_val = data.min()
#                 max_val = data.max()
                
#                 # Memory-efficient histogram creation
#                 try:
#                     # Calculate optimal bin count - Freedman-Diaconis rule with safety caps
#                     iqr = np.percentile(data, 75) - np.percentile(data, 25)
#                     if iqr == 0:
#                         iqr = std_val * 1.35  # Approximate IQR from std dev if IQR is zero
#                     if iqr == 0:
#                         bin_width = 1.0  # Fallback if we still have zero
#                     else:
#                         bin_width = 2 * iqr / (len(data) ** (1/3))
                    
#                     # Safety guard on bin width to avoid memory issues
#                     data_range = max_val - min_val
#                     if bin_width == 0 or data_range / bin_width > 200:
#                         bin_width = data_range / 100  # Limit to 100 bins max
                    
#                     # Further safety check
#                     if bin_width == 0:
#                         bins = 10  # Fallback to 10 bins if everything else fails
#                     else:
#                         bins = int(np.ceil((max_val - min_val) / bin_width)) + 1
#                         bins = min(100, max(10, bins))  # Keep bins between 10-100
                    
#                     # Create histogram with controlled bin count
#                     hist, bin_edges = np.histogram(data, bins=bins, density=True)
#                     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
#                     # Plot histogram bars
#                     ax.bar(bin_centers, hist, width=bin_edges[1]-bin_edges[0], 
#                            alpha=0.4, color='skyblue', align='center')
                    
#                     # Create KDE manually if there are enough unique values
#                     if len(data.unique()) > 5:
#                         # Use Gaussian KDE with limited sample size
#                         if len(data) > 10000:
#                             # Sample data for KDE to avoid memory issues
#                             sample_size = 10000
#                             kde_sample = data.sample(sample_size, random_state=42)
#                         else:
#                             kde_sample = data
                            
#                         kde = scipy.stats.gaussian_kde(kde_sample)
#                         x_range = np.linspace(min_val, max_val, 500)
#                         kde_vals = kde(x_range)
#                         ax.plot(x_range, kde_vals, color='navy', linewidth=2)
                    
#                 except Exception as e:
#                     print(f"Warning: Could not create histogram for {metric}: {e}")
#                     # Fallback to simple plot without histogram
#                     ax.text(0.5, 0.5, f"Could not create histogram:\n{e}", 
#                            transform=ax.transAxes, horizontalalignment='center',
#                            verticalalignment='center', fontsize=9)
                
#                 # Add vertical lines for mean and median
#                 ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
#                 ax.axvline(median_val, color='green', linestyle='-.', alpha=0.7, label=f'Median: {median_val:.2f}')
                
#                 # Add stats text
#                 stats_text = (f"Mean: {mean_val:.2f}\n"
#                              f"Median: {median_val:.2f}\n"
#                              f"Std Dev: {std_val:.2f}\n"
#                              f"Min: {min_val:.2f}\n"
#                              f"Max: {max_val:.2f}")
                
#                 ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
#                        verticalalignment='top', horizontalalignment='right',
#                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
#                 # Only add legend on the first plot of each figure
#                 if i == 0:
#                     ax.legend(fontsize='small')
                
#                 # Format metric name for display
#                 display_name = metric.replace('.scalar', ' (scalar)') if '.scalar' in metric else metric
#                 ax.set_title(f'Distribution of {display_name}', fontsize=9)
#                 ax.tick_params(labelsize=8)
            
#             plt.tight_layout()
            
#             # Add figure number to filename if we have multiple figures
#             figure_suffix = f"_{fig_num+1}" if n_figures > 1 else ""
#             filename = f'metric_distributions{suffix}{figure_suffix}.png'
#             plt.savefig(os.path.join(output_dir, filename), dpi=200)
#             plt.close()
#             print(f"✓ {filename} created")
            
#         except Exception as e:
#             print(f"Error creating figure {fig_num+1}: {e}")
#             plt.close()  # Ensure any open figure is closed
    
#     print(f"Completed distribution visualizations for all {n_metrics} metrics")

def visualize_quality_clusters(df, metrics, output_dir, suffix=""):
    """Create a dimensionality reduction plot to see how images cluster by quality"""
    # Use metrics other than the overall score for PCA
    score_col = 'UnifiedQualityScore' if 'UnifiedQualityScore' in metrics else 'UnifiedQualityScore.scalar'
    feature_cols = [m for m in metrics if m != score_col]
    
    if len(feature_cols) < 2:
        print("Not enough metrics for clustering visualization")
        return
    
    # Scale the data
    X = df[feature_cols].fillna(df[feature_cols].mean())  # Replace NaN with mean for scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create scatter plot
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1], 
        c=df[score_col], 
        cmap='viridis', 
        alpha=0.6,
        s=30
    )
    
    # Add colorbar
    plt.colorbar(scatter, label=score_col.replace('.scalar', ' (scalar)') if '.scalar' in score_col else score_col)
    
    # Add labels
    is_scalar = '.scalar' in metrics[0]
    plt.title(f'Image Clustering by Quality Metrics{" (Scalar)" if is_scalar else ""} (PCA)', fontsize=11)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=9)
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=9)
    plt.tight_layout()
    
    filename = f'quality_clusters{suffix}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close()
    print(f"✓ {filename} created")

def visualize_metric_ridges(df, metrics, output_dir, suffix=""):
    """
    Create a ridge plot for comparing metric distributions.
    Only includes metrics up to NaturalColour regardless of how many are provided.
    """
    # Filter metrics to only include up to NaturalColour
    cutoff_metric_options = ['NaturalColour', 'NaturalColour.scalar']

    # Find the first matching metric in the list
    cutoff_metric = next((m for m in cutoff_metric_options if m in metrics), None)
    
    if cutoff_metric:
        cutoff_index = metrics.index(cutoff_metric)
        cutoff_index = metrics.index(cutoff_metric)
        filtered_metrics = metrics[:cutoff_index + 1]  # Include NaturalColour
        print(f"Ridge plot: Limiting to {len(filtered_metrics)} metrics (up to {cutoff_metric})")
    else:
        # If NaturalColour isn't in the list, use all metrics but log a warning
        filtered_metrics = metrics
        print(f"Warning: '{cutoff_metric}' not found in metrics list. Using all provided metrics for ridge plot.")
    
    # Normalize each metric for fair comparison (z-score)
    df_norm = df.copy()
    for metric in filtered_metrics:
        mean = df[metric].mean()
        std = df[metric].std()
        if std > 0:  # Avoid division by zero
            df_norm[metric] = (df[metric] - mean) / std
        else:
            df_norm[metric] = 0  # Set to constant if std is 0
    
    # Melt the dataframe
    melted_df = pd.melt(df_norm[filtered_metrics], var_name='Metric', value_name='Value')
    
    # Format metric names for display
    melted_df['Metric'] = melted_df['Metric'].apply(
        lambda x: x.replace('.scalar', ' (scalar)') if '.scalar' in x else x
    )
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Define the colors for each metric (palette with distinct colors)
    palette = sns.color_palette("husl", len(filtered_metrics))
    
    # Create a ridge plot with Seaborn's kdeplot
    for i, (metric, color) in enumerate(zip(filtered_metrics, palette)):
        metric_display = metric.replace('.scalar', ' (scalar)') if '.scalar' in metric else metric
        data = df_norm[metric].dropna()
        
        # Plot the KDE with filled area
        sns.kdeplot(
            x=data, 
            fill=True,
            alpha=0.5,
            linewidth=1.5,
            color=color,
            label=metric_display
        )
        
    is_scalar = '.scalar' in filtered_metrics[0] if filtered_metrics else False
    plt.title(f'Normalized Distribution of Quality Metrics{" (Scalar)" if is_scalar else ""}', fontsize=11)
    plt.xlabel('Standardized Value (Z-score)', fontsize=9)
    plt.legend(fontsize=7)
    plt.tight_layout()
    
    filename = f'metric_ridges{suffix}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close()
    print(f"✓ {filename} created")

def visualize_quality_categories(df, output_dir, suffix=""):
    """Categorize images by quality and visualize the count"""
    # Create quality categories
    quality_col = 'UnifiedQualityScore' if 'UnifiedQualityScore' in df.columns else 'UnifiedQualityScore.scalar'
    if quality_col not in df.columns:
        print(f"Column {quality_col} not found, skipping quality categories visualization")
        return
    
    # Create quality bins
    max_quality = df[quality_col].max()
    bins = [0, max_quality*0.2, max_quality*0.4, max_quality*0.6, max_quality*0.8, max_quality]
    labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    
    # Add a new column with the quality category
    df['QualityCategory'] = pd.cut(df[quality_col], bins=bins, labels=labels, right=False)
    
    # Count images in each category
    category_counts = df['QualityCategory'].value_counts().sort_index()
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(category_counts.index, category_counts.values, color='skyblue')
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height}', ha='center', va='bottom', fontsize=9)
    
    is_scalar = '.scalar' in quality_col
    plt.title(f'Image Count by Quality Category{" (Based on Scalar Values)" if is_scalar else ""}', fontsize=11)
    plt.xlabel('Quality Category', fontsize=9)
    plt.ylabel('Number of Images', fontsize=9)
    plt.tick_params(labelsize=8)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filename = f'quality_categories{suffix}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close()
    print(f"✓ {filename} created")

def visualize_quality_comparison(df, metrics, output_dir, suffix=""):
    """Compare quality metrics between high and low quality images"""
    # Identify quality score column
    score_col = 'UnifiedQualityScore' if 'UnifiedQualityScore' in metrics else 'UnifiedQualityScore.scalar'
    
    # Split data into high and low quality groups
    median_quality = df[score_col].median()
    high_quality = df[df[score_col] >= median_quality]
    low_quality = df[df[score_col] < median_quality]
    
    # Prepare data for plotting - exclude quality score itself
    comparison_metrics = [m for m in metrics if m != score_col]
    
    if not comparison_metrics:
        print("No metrics available for comparison besides quality score")
        return
    
    # Calculate means for each group
    high_means = [high_quality[metric].mean() for metric in comparison_metrics]
    low_means = [low_quality[metric].mean() for metric in comparison_metrics]
    
    # Format metric names for display
    display_metrics = [m.replace('.scalar', '\n(scalar)') if '.scalar' in m else m for m in comparison_metrics]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(comparison_metrics))
    width = 0.35
    
    # Create bars
    plt.bar(x - width/2, high_means, width, label='High Quality Images', color='green', alpha=0.7)
    plt.bar(x + width/2, low_means, width, label='Low Quality Images', color='red', alpha=0.7)
    
    # Add labels and legend
    plt.xlabel('Quality Metrics', fontsize=9)
    plt.ylabel('Average Value', fontsize=9)
    is_scalar = '.scalar' in metrics[0]
    plt.title(f'Comparison of Quality Metrics{" (Scalar)" if is_scalar else ""}: High vs Low Quality Images', fontsize=11)
    plt.xticks(x, display_metrics, rotation=45, ha='right', fontsize=8)
    plt.legend(fontsize=8)
    
    plt.tight_layout()
    filename = f'quality_comparison{suffix}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close()
    print(f"✓ {filename} created")

def visualize_image_tileboard(df, scalar_metrics, output_dir):
    """Create an improved tileboard of images with detailed metrics"""
    # Check if the filename column exists
    if 'Filename' not in df.columns:
        print("Filename column not found, skipping image tileboard")
        return
        
    # Determine if we can actually access the image files
    can_access_images = False
    sample_path = df['Filename'].iloc[0]
    try:
        # Try to load the first image to see if paths are accessible
        img = Image.open(sample_path)
        can_access_images = True
        img.close()
    except:
        print(f"Warning: Could not open image at {sample_path}")
        print("Will create tileboard without actual images")
    
    # Sort dataframe by quality score (use scalar version if available)
    quality_col = ('UnifiedQualityScore.scalar' if 'UnifiedQualityScore.scalar' in df.columns 
                  else 'UnifiedQualityScore')
    if quality_col not in df.columns:
        print(f"Quality column {quality_col} not found, using first column for sorting")
        quality_col = df.columns[0]
    
    df_sorted = df.sort_values(by=quality_col, ascending=False)
    
    # Show fewer images with more details (3x3 grid = 9 images)
    n_images = min(9, len(df_sorted))
    n_cols = 3
    n_rows = 3
    
    plt.figure(figsize=(15, 15))
    for i in range(n_images):
        plt.subplot(n_rows, n_cols, i+1)
        
        # Get filename and metrics for current image
        img_path = df_sorted['Filename'].iloc[i]
        img_metrics = {m: df_sorted[m].iloc[i] for m in scalar_metrics}
        
        if can_access_images:
            try:
                img = mpimg.imread(img_path)
                plt.imshow(img)
            except:
                # If we can't load the image, show a placeholder
                plt.text(0.5, 0.5, "Image not available", 
                         horizontalalignment='center', verticalalignment='center', fontsize=10)
                plt.axis('off')
        else:
            # Create colored rectangle based on quality
            quality = df_sorted[quality_col].iloc[i]
            quality_max = df[quality_col].max()
            normalized_quality = min(1.0, max(0.0, quality / quality_max))
            
            # Use a color gradient: red (low quality) to green (high quality)
            color = (1 - normalized_quality, normalized_quality, 0.2)
            
            # Create a colored rectangle
            rect = plt.Rectangle((0, 0), 1, 1, color=color)
            plt.gca().add_patch(rect)
            plt.gca().set_aspect('equal')
            plt.axis('off')
        
        # Add image details as a text block below the image
        filename = Path(img_path).name
        if len(filename) > 30:  # Truncate long filenames
            filename = filename[:27] + '...'
            
        # Format metrics text (include scalar metrics)
        metrics_text = f"Filename: {filename}\n"
        metrics_text += f"Quality Score: {img_metrics.get('UnifiedQualityScore.scalar', 'N/A'):.2f}\n"
        
        # Add other scalar metrics (skip UnifiedQualityScore which we already included)
        other_metrics = [m for m in scalar_metrics if m != 'UnifiedQualityScore.scalar']
        for m in other_metrics[:5]:  # Limit to 5 additional metrics to avoid clutter
            display_name = m.replace('.scalar', '')
            value = img_metrics.get(m, 'N/A')
            if isinstance(value, (int, float)):
                metrics_text += f"{display_name}: {value:.2f}\n"
            else:
                metrics_text += f"{display_name}: {value}\n"
        
        plt.title(metrics_text, fontsize=8, loc='left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'image_tileboard.png'), dpi=200)
    plt.close()
    print("✓ image_tileboard.png created")

def visualize_size_resolution_correlation(df, metrics, output_dir):
    """Create plots showing correlation between image properties and quality metrics"""
    # Check if we have image property columns
    property_cols = ['width', 'height', 'resolution', 'filesize', 'aspect_ratio']
    available_props = [col for col in property_cols if col in df.columns]
    
    if not available_props:
        print("No image property columns found, skipping size/resolution correlation")
        return
    
    # Choose most relevant metrics (limit to avoid too many plots)
    key_metrics = ['UnifiedQualityScore', 'Sharpness', 'CompressionArtifacts'] if 'UnifiedQualityScore' in metrics else []
    key_metrics += [m for m in metrics if m not in key_metrics][:5]  # Add a few more if needed
    
    # For each property, create correlation plots with key metrics
    for prop in available_props:
        plt.figure(figsize=(12, 5))
        
        n_metrics = len(key_metrics)
        for i, metric in enumerate(key_metrics, 1):
            plt.subplot(1, n_metrics, i)
            
            # Create scatter plot
            plt.scatter(df[prop], df[metric], alpha=0.4, s=5)
            
            # Add trend line
            if df[prop].notna().sum() > 2 and df[metric].notna().sum() > 2:
                try:
                    mask = np.isfinite(df[prop]) & np.isfinite(df[metric])
                    if mask.sum() > 2:
                        z = np.polyfit(df[prop][mask], df[metric][mask], 1)
                        p = np.poly1d(z)
                        x_range = np.linspace(df[prop][mask].min(), df[prop][mask].max(), 100)
                        plt.plot(x_range, p(x_range), "r--", alpha=0.7)
                        
                        # Calculate correlation coefficient
                        corr = df[[prop, metric]].corr().iloc[0, 1]
                        plt.title(f"{metric}\nCorr: {corr:.2f}", fontsize=9)
                except:
                    plt.title(metric, fontsize=9)
            else:
                plt.title(metric, fontsize=9)
                
            plt.xlabel(prop, fontsize=8)
            plt.ylabel(metric if i == 1 else "", fontsize=8)
            plt.tick_params(labelsize=7)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prop}_correlation.png'), dpi=200)
        plt.close()
        print(f"✓ {prop}_correlation.png created")
    
    # Create a heatmap showing correlation between all properties and metrics
    corr_cols = available_props + key_metrics
    corr_matrix = df[corr_cols].corr()
    
    plt.figure(figsize=(10, 8))
    
    # Extract just the correlations between properties and metrics
    prop_metric_corr = corr_matrix.loc[available_props, key_metrics]
    
    # Create heatmap
    sns.heatmap(prop_metric_corr, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", linewidths=.5, annot_kws={"size": 8})
    
    plt.title('Correlation: Image Properties vs. Quality Metrics', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'property_metric_correlation.png'), dpi=200)
    plt.close()
    print("✓ property_metric_correlation.png created")

def create_quality_visualizations(csv_file, output_dir='quality_plots'):
    """Create comprehensive visualizations for image quality metrics"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    df, raw_metrics, scalar_metrics = prepare_data(csv_file)
    
    if not raw_metrics:
        print("Error: No quality metrics found in the CSV file")
        return
    
    print("\n=== Creating visualizations for raw metrics ===")
    
    # Raw metrics visualizations
    visualize_metric_distributions(df, raw_metrics, output_dir)
    
    if len(raw_metrics) > 1:
        visualize_correlation_heatmap(df, raw_metrics, output_dir)
        visualize_metric_boxplots(df, raw_metrics, output_dir)
        visualize_metric_ridges(df, raw_metrics, output_dir)
        visualize_raw_vs_scalar(df, raw_metrics, output_dir)
        # visualize_metric_distributions(df, raw_metrics, output_dir)
        
        if 'UnifiedQualityScore' in raw_metrics:
            visualize_quality_categories(df, output_dir)
            visualize_quality_clusters(df, raw_metrics, output_dir)
            visualize_quality_comparison(df, raw_metrics, output_dir)
    
    # Scalar metrics visualizations (if available)
    if scalar_metrics:
        print("\n=== Creating visualizations for scalar metrics ===")
        
        visualize_metric_distributions(df, scalar_metrics, output_dir, "_scalar")
        
        if len(scalar_metrics) > 1:
            visualize_correlation_heatmap(df, scalar_metrics, output_dir, "_scalar")
            visualize_metric_boxplots(df, scalar_metrics, output_dir, "_scalar")
            visualize_metric_ridges(df, scalar_metrics, output_dir, "_scalar")
            
            if 'UnifiedQualityScore.scalar' in scalar_metrics:
                visualize_quality_categories(df, output_dir, "_scalar")
                visualize_quality_clusters(df, scalar_metrics, output_dir, "_scalar")
                visualize_quality_comparison(df, scalar_metrics, output_dir, "_scalar")
    
    # Create improved image tileboard
    available_metrics = scalar_metrics if scalar_metrics else raw_metrics
    visualize_image_tileboard(df, available_metrics, output_dir)
    
    # Create size/resolution correlation plots
    visualize_size_resolution_correlation(df, available_metrics, output_dir)
    
    print(f"\nAll visualizations complete. Check {output_dir} directory for results.")


# In[46]:


get_ipython().system("find '/home/meem/backup/Age Datasets/utkface-cropped/all_results.csv'")


# In[47]:


def main():
    """Main function to run the visualization code"""
    # #1
    # # Specify your CSV file path here
    # csv_file = '/home/meem/backup/Age Datasets/fairface/all_results.csv'  # Update with your file path
    # output_dir = 'vis/fairface'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    # create_quality_visualizations(csv_file, output_dir)

    # #2
    # csv_file = '/home/meem/backup/Age Datasets/afad/ofiq.csv'  # Update with your file path
    # output_dir = 'vis/afad'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    # create_quality_visualizations(csv_file, output_dir)

    # #3    
    # csv_file = '/home/meem/backup/Age Datasets/AdienceGender Dataset/ofiq.csv'  # Update with your file path
    # output_dir = 'vis/adience'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    # create_quality_visualizations(csv_file, output_dir)

    # #4    
    # csv_file = '/home/meem/backup/Age Datasets/agedb/ofiq.csv'  # Update with your file path
    # output_dir = 'vis/agedb'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    # create_quality_visualizations(csv_file, output_dir)

    # #5    
    # csv_file = '/home/meem/backup/Age Datasets/appa-real-release/fixed_ofiq.csv'  # Update with your file path
    # output_dir = 'vis/appa-real'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    # create_quality_visualizations(csv_file, output_dir)

    # #6    
    # csv_file = '/home/meem/backup/Age Datasets/FGNET Dataset/ofiq.csv'  # Update with your file path
    # output_dir = 'vis/fgnet'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    # create_quality_visualizations(csv_file, output_dir)

    # #7    
    # csv_file = '/home/meem/backup/Age Datasets/Groups-of-People Dataset/ofiq.csv'  # Update with your file path
    # output_dir = 'vis/groups'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    # create_quality_visualizations(csv_file, output_dir)

    # #8    
    # csv_file = '/home/meem/backup/Age Datasets/IMDB - WIKI/ofiq.csv'  # Update with your file path
    # output_dir = 'vis/imdb-wiki'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    # create_quality_visualizations(csv_file, output_dir)

    # #9     
    # csv_file = '/home/meem/backup/Age Datasets/Juvenile_Dataset/Juvenile_Dataset/ofiq.csv'  # Update with your file path
    # output_dir = 'vis/juvenil'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    # create_quality_visualizations(csv_file, output_dir)  # Update with your file path


    # #10     
    # csv_file = '/home/meem/backup/Age Datasets/lagenda/lagenda_seg_results.csv'  # Update with your file path
    # output_dir = 'vis/lagenda'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    # create_quality_visualizations(csv_file, output_dir)  # Update with your file path

    # #11     
    # csv_file = '/home/meem/backup/Age Datasets/Morph2 Dataset/ofiq.csv'  # Update with your file path
    # output_dir = 'vis/morph2'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    # create_quality_visualizations(csv_file, output_dir)  # Update with your file path

    # #12     
    # csv_file = '/home/meem/backup/Age Datasets/utkface-cropped/all_results.csv'  # Update with your file path
    # output_dir = 'vis/utkface'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    # create_quality_visualizations(csv_file, output_dir)  # Update with your file path

    


if __name__ == "__main__":
    main()


# In[ ]:


#!find "/home/meem/backup/Age Datasets/appa-real-release/" -type f -name '005245.jpg_face.jpg'
#!find "/home/meem/backup/Age Datasets/appa-real-release/" -type f -name '003830.jpg_face.jpg'


# In[ ]:


#!sed -i 's|.*\(/home/meem.*\)|\1|' "/home/meem/backup/Age Datasets/fairface/all_results.csv"
#!head -5 "/home/meem/backup/Age Datasets/fairface/all_results.csv"


# In[ ]:





# In[ ]:


""" UnifiedQualityScore >> bad vs good example

BackgroundUniformity >> bad vs good example

IlluminationUniformity >> bad vs good example 

LuminanceMean >> bad vs good example 

LuminanceVariance >> ?

UnderExposurePrevention >> bad vs good example 

OverExposurePrevention >> bad vs good example 

DynamicRange >> ??

Sharpness >> bad vs good example 

CompressionArtifacts >> bad vs good example 

NaturalColour >> bad vs good example 

------------------------

SingleFacePresent >> bad vs good example 

-------

EyesOpen >> distribution, relation w ground truth, sample

MouthClosed >> distribution, relation w ground truth, sample

EyesVisible >> distribution, relation w ground truth, sample

MouthOcclusionPrevention >> distribution, relation w ground truth, sample

FaceOcclusionPrevention >> distribution, relation w ground truth, sample

InterEyeDistance >> distribution, relation w ground truth, sample

HeadSize >> box, distribution, relation w ground truth, sample

LeftwardCropOfTheFaceImage, RightwardCropOfTheFaceImage, MarginAboveOfTheFaceImage, 
MarginBelowOfTheFaceImage >> box, distribution, relation w ground truth, sample

HeadPoseYaw, HeadPosePitch, HeadPoseRoll, >> distribution, relation w ground truth, samples

ExpressionNeutrality >> relation w ground truth, samples

NoHeadCoverings >> relation w ground truth, samples"""


# In[ ]:





# In[ ]:




