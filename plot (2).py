#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import traceback
from pathlib import Path

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
plt.rcParams['figure.figsize'] = (12, 8)

def load_groundtruth(groundtruth_file):
    """Load and validate groundtruth data"""
    try:
        # First attempt: standard CSV parsing
        gt = pd.read_csv(groundtruth_file)
        
        # Check if required columns exist
        if 'path' in gt.columns and 'age' in gt.columns and 'gender' in gt.columns:
            print("Groundtruth data loaded successfully")
            return gt
            
        # If we have the right number of columns but wrong names, rename them
        if gt.shape[1] == 3:
            gt.columns = ['path', 'age', 'gender']
            gt['age'] = pd.to_numeric(gt['age'], errors='coerce')
            print("Renamed groundtruth columns")
            return gt
            
        # Try alternative parsing approaches
        print("Standard parsing failed. Trying alternative approaches...")
        
        # Approach 2: No header, comma delimiter
        gt = pd.read_csv(groundtruth_file, header=None)
        
        # If single column, try splitting
        if gt.shape[1] == 1:
            gt = gt[0].str.split(',', expand=True)
            if gt.shape[1] >= 3:
                gt = gt.iloc[:, 0:3]  # Take first 3 columns
                gt.columns = ['path', 'age', 'gender']
                gt['age'] = pd.to_numeric(gt['age'], errors='coerce')
                print("Parsed groundtruth by splitting single column")
                return gt
        
        # Approach 3: Try different delimiters
        for delimiter in ['\t', ';', '|']:
            try:
                gt = pd.read_csv(groundtruth_file, sep=delimiter)
                if gt.shape[1] >= 3:
                    gt = gt.iloc[:, 0:3]
                    gt.columns = ['path', 'age', 'gender']
                    gt['age'] = pd.to_numeric(gt['age'], errors='coerce')
                    print(f"Parsed groundtruth using '{delimiter}' delimiter")
                    return gt
            except:
                continue
                
        # If all approaches fail, try manual parsing
        with open(groundtruth_file, 'r') as f:
            lines = f.readlines()
            
        data = []
        for line in lines:
            # Try various splitting strategies
            parts = re.split(r',|\t|;|\|', line.strip())
            if len(parts) >= 3:
                path = parts[0]
                try:
                    age = int(parts[1])
                except:
                    age = np.nan
                gender = parts[2]
                data.append([path, age, gender])
                
        if data:
            gt = pd.DataFrame(data, columns=['path', 'age', 'gender'])
            print("Parsed groundtruth manually")
            return gt
            
        raise ValueError(f"Could not parse groundtruth file: {groundtruth_file}")
        
    except Exception as e:
        print(f"Error loading groundtruth: {e}")
        traceback.print_exc()
        return None

def load_ofiq(ofiq_file):
    """Load and validate OFIQ data"""
    try:
        # First attempt: standard CSV parsing
        ofiq = pd.read_csv(ofiq_file, sep = ";")
        
        # Verify it has the expected columns
        if 'Filename' in ofiq.columns or any('UnifiedQualityScore' in col for col in ofiq.columns):
            print("OFIQ data loaded successfully")
            return ofiq
            
        # Try alternative parsing approaches
        print("Standard parsing failed. Trying alternative approaches...")
        
        # Try different delimiters
        for delimiter in [',', '\t', ';', '|']:
            try:
                ofiq = pd.read_csv(ofiq_file, sep=delimiter)
                if any('UnifiedQualityScore' in col for col in ofiq.columns) or any('Quality' in col for col in ofiq.columns):
                    print(f"Parsed OFIQ using '{delimiter}' delimiter")
                    return ofiq
            except:
                continue
                
        # Last resort: try to load the first line to get column names
        with open(ofiq_file, 'r') as f:
            header = f.readline().strip()
            
        if 'UnifiedQualityScore' in header or 'Quality' in header:
            # Find the most likely delimiter
            delimiters = [',', '\t', ';', '|']
            counts = [header.count(d) for d in delimiters]
            best_delimiter = delimiters[np.argmax(counts)]
            
            ofiq = pd.read_csv(ofiq_file, sep=best_delimiter)
            print(f"Parsed OFIQ using detected delimiter '{best_delimiter}'")
            return ofiq
            
        raise ValueError(f"Could not parse OFIQ file: {ofiq_file}")
        
    except Exception as e:
        print(f"Error loading OFIQ: {e}")
        traceback.print_exc()
        return None


# Modify the plot_age_distribution function to handle unknown gender values
def plot_age_distribution(groundtruth, output_dir):
    """Create age distribution plots"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get unique gender values
        unique_genders = groundtruth['gender'].unique()
        has_males = 'm' in unique_genders
        has_females = 'f' in unique_genders
        has_unknown = any(g for g in unique_genders if g not in ['m', 'f'])
        
        # 1. Box plots (three side by side: all, male, female)
        plt.figure(figsize=(18, 6))
        
        # Plot 1: Overall boxplot
        plt.subplot(1, 3, 1)
        ax1 = sns.boxplot(y=groundtruth['age'], color='purple')
        plt.title('Overall Age Distribution', fontsize=14)
        plt.ylabel('Age', fontsize=12)
        
        # Add mean and median markers
        mean_overall = groundtruth['age'].mean()
        median_overall = groundtruth['age'].median()
        plt.axhline(mean_overall, color='red', linestyle='--', alpha=0.7)
        plt.axhline(median_overall, color='green', linestyle='-', alpha=0.7)
        plt.text(0.05, mean_overall + 1, f'Mean: {mean_overall:.1f}', color='red', fontweight='bold')
        plt.text(0.05, median_overall - 2, f'Median: {median_overall:.1f}', color='green', fontweight='bold')
        
        # Plots for different genders
        if has_males:
            # Plot 2: Male boxplot
            plt.subplot(1, 3, 2)
            males = groundtruth[groundtruth['gender'] == 'm']
            ax2 = sns.boxplot(y=males['age'], color='blue')
            plt.title('Male Age Distribution', fontsize=14)
            plt.ylabel('Age', fontsize=12)
            
            # Add mean and median markers
            mean_male = males['age'].mean()
            median_male = males['age'].median()
            plt.axhline(mean_male, color='red', linestyle='--', alpha=0.7)
            plt.axhline(median_male, color='green', linestyle='-', alpha=0.7)
            plt.text(0.05, mean_male + 1, f'Mean: {mean_male:.1f}', color='red', fontweight='bold')
            plt.text(0.05, median_male - 2, f'Median: {median_male:.1f}', color='green', fontweight='bold')
        
        if has_females:
            # Plot 3: Female boxplot
            subplot_position = 3 if has_males else 2
            plt.subplot(1, 3, subplot_position)
            females = groundtruth[groundtruth['gender'] == 'f']
            ax3 = sns.boxplot(y=females['age'], color='pink')
            plt.title('Female Age Distribution', fontsize=14)
            plt.ylabel('Age', fontsize=12)
            
            # Add mean and median markers
            mean_female = females['age'].mean()
            median_female = females['age'].median()
            plt.axhline(mean_female, color='red', linestyle='--', alpha=0.7)
            plt.axhline(median_female, color='green', linestyle='-', alpha=0.7)
            plt.text(0.05, mean_female + 1, f'Mean: {mean_female:.1f}', color='red', fontweight='bold')
            plt.text(0.05, median_female - 2, f'Median: {median_female:.1f}', color='green', fontweight='bold')
        
        if has_unknown:
            pass
            # plt.figure(figsize=(6, 6))
        
        # Add a legend for mean and median (on the original figure)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, linestyle='--', label='Mean'),
            Line2D([0], [0], color='green', lw=2, linestyle='-', label='Median')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'age_boxplots.png'), bbox_inches='tight')
        plt.close()
        
        # Similar modifications for histogram plots...

                
        # 2. KDE + Histogram plots - handling all gender types
        plt.figure(figsize=(18, 6))
        
        # First subplot: Overall histogram + KDE
        plt.subplot(1, 3, 1)
        sns.histplot(groundtruth['age'], kde=True, color='purple', alpha=0.5)
        plt.title('Overall Age Distribution', fontsize=14)
        plt.xlabel('Age', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        # Handle different gender values
        subplot_index = 2
        if has_males:
            # Male histogram + KDE
            plt.subplot(1, 3, subplot_index)
            males = groundtruth[groundtruth['gender'] == 'm']
            sns.histplot(males['age'], kde=True, color='blue', alpha=0.5)
            plt.title('Male Age Distribution', fontsize=14)
            plt.xlabel('Age', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            subplot_index += 1
        
        if has_females:
            # Female histogram + KDE
            if subplot_index <= 3:  # Only if we have space in the current figure
                plt.subplot(1, 3, subplot_index)
                females = groundtruth[groundtruth['gender'] == 'f']
                sns.histplot(females['age'], kde=True, color='pink', alpha=0.5)
                plt.title('Female Age Distribution', fontsize=14)
                plt.xlabel('Age', fontsize=12)
                plt.ylabel('Count', fontsize=12)
                subplot_index += 1
        
        # If we have unknown gender or didn't have space for all plots, create additional figure
        if has_unknown:
            pass
            # plt.figure(figsize=(6, 6))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'age_histogram.png'), bbox_inches='tight')
        plt.close()
        
        # 3. Population Pyramid
        # Group by age and gender
        males = groundtruth[groundtruth['gender'] == 'm'].groupby('age').size()
        females = groundtruth[groundtruth['gender'] == 'f'].groupby('age').size()
        
        # Ensure both have the same age range
        all_ages = np.arange(groundtruth['age'].min(), groundtruth['age'].max() + 1)
        males = males.reindex(all_ages, fill_value=0)
        females = females.reindex(all_ages, fill_value=0)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Males on left (negative values)
        ax.barh(males.index, -males.values, color='skyblue', alpha=0.8, label='Male')
        # Females on right (positive values)
        ax.barh(females.index, females.values, color='pink', alpha=0.8, label='Female')
        
        # Formatting
        ax.set_xlabel('Count', fontsize=14)
        ax.set_ylabel('Age', fontsize=14)
        ax.set_title('Population Pyramid', fontsize=16)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Fix x-axis to show absolute values
        max_count = max(males.max(), females.max()) if not (males.empty or females.empty) else 1
        ax.set_xlim(-max_count*1.1, max_count*1.1)
        
        # Custom x-tick labels to show absolute values
        ticks = np.linspace(0, max_count, 5).astype(int)
        ax.set_xticks(np.concatenate([-ticks[1:], ticks]))
        ax.set_xticklabels([str(abs(x)) for x in np.concatenate([-ticks[1:], ticks])])
        
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'population_pyramid.png'), bbox_inches='tight')
        plt.close()
        
        print("Age distribution plots created successfully")
        return True
        
    except Exception as e:
        print(f"Error creating age distribution plots: {e}")
        traceback.print_exc()
        return False

def match_datasets(groundtruth, ofiq):
    """Match records between groundtruth and OFIQ datasets"""
    try:
        # First, identify the path/filename column in OFIQ
        ofiq_path_col = None
        
        # Look for the Filename column
        if 'Filename' in ofiq.columns:
            ofiq_path_col = 'Filename'
        else:
            # Try to find a column containing file paths
            for col in ofiq.columns:
                if ofiq[col].dtype == object and isinstance(ofiq[col].iloc[0], str):
                    if any(ofiq[col].str.contains('/', na=False)):
                        ofiq_path_col = col
                        print(f"Using '{col}' as the path column in OFIQ")
                        break
        
        # If still not found, use the first column
        if ofiq_path_col is None:
            ofiq_path_col = ofiq.columns[0]
            print(f"Using first column '{ofiq_path_col}' as path in OFIQ")
        
        # Extract filenames from paths
        groundtruth['filename'] = groundtruth['path'].apply(lambda x: os.path.basename(str(x)))
        ofiq['filename'] = ofiq[ofiq_path_col].apply(lambda x: os.path.basename(str(x)))
        
        # Try to match based on filename
        merged = pd.merge(groundtruth, ofiq, on='filename', how='inner')
        print(f"Matched {merged.shape[0]} records using filenames")
        
        # If few matches, try extracting IDs from filenames
        if merged.shape[0] < min(len(groundtruth), len(ofiq)) * 0.1:
            print("Few matches found. Trying ID extraction...")
            
            # Try to extract IDs from filenames using various patterns
            def extract_id(filename):
                # Try hex ID pattern (common in your example)
                hex_match = re.search(r'([0-9a-f]{16})', str(filename))
                if hex_match:
                    return hex_match.group(1)
                
                # Try numeric ID pattern
                num_match = re.search(r'(\d{6})', str(filename))
                if num_match:
                    return num_match.group(1)
                
                return None
            
            groundtruth['file_id'] = groundtruth['filename'].apply(extract_id)
            ofiq['file_id'] = ofiq['filename'].apply(extract_id)
            
            # Drop rows where ID couldn't be extracted
            gt_with_id = groundtruth.dropna(subset=['file_id'])
            ofiq_with_id = ofiq.dropna(subset=['file_id'])
            
            # Merge on extracted ID
            id_merged = pd.merge(gt_with_id, ofiq_with_id, on='file_id', how='inner')
            print(f"Matched {id_merged.shape[0]} records using extracted IDs")
            
            if id_merged.shape[0] > merged.shape[0]:
                merged = id_merged
        
        return merged
        
    except Exception as e:
        print(f"Error matching datasets: {e}")
        traceback.print_exc()
        return pd.DataFrame()  # Return empty DataFrame on error

def analyze_correlations(merged_data, output_dir):
    """Analyze correlations between OFIQ metrics and age/gender"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert gender to numeric (1 for male, 0 for female)
        merged_data['gender_numeric'] = merged_data['gender'].map({'m': 1, 'f': 0})
        
        # Get all potential OFIQ metrics (exclude groundtruth columns and matching columns)
        exclude_cols = ['path', 'age', 'gender', 'filename', 'file_id', 'gender_numeric']
        potential_metrics = [col for col in merged_data.columns if col not in exclude_cols]
        
        # Find numeric columns only
        numeric_metrics = []
        for col in potential_metrics:
            try:
                pd.to_numeric(merged_data[col])
                numeric_metrics.append(col)
            except:
                continue
        
        print(f"Found {len(numeric_metrics)} numeric OFIQ metrics")
        
        # Calculate correlations with age and gender
        correlations = []
        for metric in numeric_metrics:
            try:
                age_corr = merged_data[['age', metric]].corr().iloc[0, 1]
                gender_corr = merged_data[['gender_numeric', metric]].corr().iloc[0, 1]
                
                correlations.append({
                    'Metric': metric,
                    'Correlation_with_Age': age_corr,
                    'Correlation_with_Gender': gender_corr
                })
            except:
                # Skip metrics that cause problems
                continue
        
        # Create correlation dataframe
        corr_df = pd.DataFrame(correlations)
        
        # Sort by absolute correlation values
        age_corr_df = corr_df.sort_values('Correlation_with_Age', key=abs, ascending=False)
        gender_corr_df = corr_df.sort_values('Correlation_with_Gender', key=abs, ascending=False)
        
        # Save results to CSV
        age_corr_df.to_csv(os.path.join(output_dir, 'age_correlations.csv'), index=False)
        gender_corr_df.to_csv(os.path.join(output_dir, 'gender_correlations.csv'), index=False)
        
        # Plot top correlations with age
        plt.figure(figsize=(12, 8))
        top_n = min(15, len(age_corr_df))
        
        if top_n > 0:
            top_age = age_corr_df.head(top_n)
            sns.barplot(x='Correlation_with_Age', y='Metric', data=top_age)
            plt.title(f'Top {top_n} OFIQ Metrics Correlated with Age', fontsize=16)
            plt.xlabel('Correlation Coefficient', fontsize=14)
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'age_correlation.png'), bbox_inches='tight')
        plt.close()
        
        # Plot top correlations with gender
        plt.figure(figsize=(12, 8))
        top_n = min(15, len(gender_corr_df))
        
        if top_n > 0:
            top_gender = gender_corr_df.head(top_n)
            sns.barplot(x='Correlation_with_Gender', y='Metric', data=top_gender)
            plt.title(f'Top {top_n} OFIQ Metrics Correlated with Gender', fontsize=16)
            plt.xlabel('Correlation Coefficient (1=Male, 0=Female)', fontsize=14)
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'gender_correlation.png'), bbox_inches='tight')
        plt.close()
        
        # Create correlation heatmap for top metrics
        if len(age_corr_df) > 0 and len(gender_corr_df) > 0:
            # Get top metrics from both age and gender correlations
            top_age_metrics = age_corr_df.head(5)['Metric'].tolist() if len(age_corr_df) >= 5 else age_corr_df['Metric'].tolist()
            top_gender_metrics = gender_corr_df.head(5)['Metric'].tolist() if len(gender_corr_df) >= 5 else gender_corr_df['Metric'].tolist()
            
            # Combine unique metrics
            top_metrics = list(set(top_age_metrics + top_gender_metrics))
            
            if top_metrics:
                # Create correlation matrix with age, gender, and top metrics
                cols_for_heatmap = ['age', 'gender_numeric'] + top_metrics
                
                # Ensure all columns exist in the dataframe
                cols_for_heatmap = [col for col in cols_for_heatmap if col in merged_data.columns]
                
                if len(cols_for_heatmap) > 2:  # Need more than just age and gender
                    corr_matrix = merged_data[cols_for_heatmap].corr()
                    
                    plt.figure(figsize=(14, 12))
                    mask = np.triu(np.ones_like(corr_matrix))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                              mask=mask, fmt='.2f', linewidths=0.5)
                    plt.title('Correlation Matrix: Age, Gender and Top OFIQ Metrics', fontsize=16)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), bbox_inches='tight')
                    plt.close()
        
        print("Correlation analysis completed successfully")
        return age_corr_df, gender_corr_df
        
    except Exception as e:
        print(f"Error in correlation analysis: {e}")
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()

def create_scatter_plots(merged_data, top_corr_metrics, target='age', output_dir=None):
    """Create scatter plots for top correlated metrics"""
    try:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Select top metrics
        top_metrics = top_corr_metrics.head(6)['Metric'].tolist() if len(top_corr_metrics) >= 6 else top_corr_metrics['Metric'].tolist()
        
        if not top_metrics:
            print("No metrics available for scatter plots")
            return False
            
        # Create grid of scatter plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        target_label = 'Age' if target == 'age' else 'Gender'
        target_col = target if target == 'age' else 'gender_numeric'
        
        for i, metric in enumerate(top_metrics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            if target == 'age':
                # Scatter plot with gender as hue
                sns.scatterplot(x=target_col, y=metric, hue='gender', 
                              data=merged_data, alpha=0.5, ax=ax)
            else:
                # For gender, use jitter to see distribution better
                sns.stripplot(x='gender', y=metric, data=merged_data, 
                            alpha=0.5, jitter=True, ax=ax)
            
            # Add correlation coefficient
            corr = merged_data[[target_col, metric]].corr().iloc[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
                  fontsize=12, verticalalignment='top')
            
            ax.set_title(f'{metric} vs {target_label}', fontsize=12)
        
        # Hide unused subplots
        for i in range(len(top_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'{target}_scatter_plots.png'), bbox_inches='tight')
        
        plt.close()
        return True
        
    except Exception as e:
        print(f"Error creating scatter plots: {e}")
        traceback.print_exc()
        return False

def analyze_age_gender_ofiq(groundtruth_file, ofiq_file, output_dir):
    """
    Main function to analyze age, gender and OFIQ data
    
    Parameters:
    -----------
    groundtruth_file : str
        Path to the groundtruth CSV file containing path, age, gender columns
    ofiq_file : str
        Path to the OFIQ CSV file containing quality metrics
    output_dir : str
        Directory to save output visualizations and CSV files
    
    Returns:
    --------
    dict
        Dictionary with analysis results and status
    """
    results = {
        'success': False,
        'dataset_name': Path(groundtruth_file).stem,
        'metrics_found': 0,
        'matches_found': 0,
        'top_age_correlations': None,
        'top_gender_correlations': None
    }
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Load datasets
        print(f"\nAnalyzing dataset: {results['dataset_name']}")
        print(f"Loading groundtruth from: {groundtruth_file}")
        groundtruth = load_groundtruth(groundtruth_file)
        
        if groundtruth is None or groundtruth.empty:
            print(f"Failed to load groundtruth data from {groundtruth_file}")
            return results
            
        print(f"Loaded groundtruth data: {groundtruth.shape[0]} records")
        
        print(f"Loading OFIQ data from: {ofiq_file}")
        ofiq = load_ofiq(ofiq_file)
        
        if ofiq is None or ofiq.empty:
            print(f"Failed to load OFIQ data from {ofiq_file}")
            return results
            
        print(f"Loaded OFIQ data: {ofiq.shape[0]} records")
        results['metrics_found'] = ofiq.shape[1]
        
        # Step 2: EDA on groundtruth data
        print("\nCreating age distribution visualizations...")
        plot_age_distribution(groundtruth, output_dir)
        
        # Step 3: Match datasets
        print("\nMatching groundtruth and OFIQ data...")
        merged_data = match_datasets(groundtruth, ofiq)
        
        if merged_data.empty:
            print("Failed to match datasets.")
            return results
            
        results['matches_found'] = merged_data.shape[0]
        print(f"Successfully matched {merged_data.shape[0]} records")
        
        # Step 4: Correlation analysis
        print("\nPerforming correlation analysis...")
        age_corr, gender_corr = analyze_correlations(merged_data, output_dir)
        
        # Step 5: Create scatter plots
        if not age_corr.empty:
            print("\nCreating scatter plots for age correlations...")
            create_scatter_plots(merged_data, age_corr, target='age', output_dir=output_dir)
            
            # Store top correlations in results
            results['top_age_correlations'] = age_corr.head(10)[['Metric', 'Correlation_with_Age']].to_dict('records')
            
        if not gender_corr.empty:
            print("\nCreating scatter plots for gender correlations...")
            create_scatter_plots(merged_data, gender_corr, target='gender', output_dir=output_dir)
            
            # Store top correlations in results
            results['top_gender_correlations'] = gender_corr.head(10)[['Metric', 'Correlation_with_Gender']].to_dict('records')
        
        # Step 6: Display results
        if not age_corr.empty and not gender_corr.empty:
            print("\nTop OFIQ metrics correlated with Age:")
            print(age_corr.head(10)[['Metric', 'Correlation_with_Age']].to_string(index=False))
            
            print("\nTop OFIQ metrics correlated with Gender:")
            print(gender_corr.head(10)[['Metric', 'Correlation_with_Gender']].to_string(index=False))
            
            # Create a summary file
            with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
                f.write(f"Dataset: {results['dataset_name']}\n")
                f.write(f"Groundtruth records: {groundtruth.shape[0]}\n")
                f.write(f"OFIQ records: {ofiq.shape[0]}\n")
                f.write(f"Matched records: {merged_data.shape[0]}\n\n")
                
                f.write("Top OFIQ metrics correlated with Age:\n")
                f.write(age_corr.head(10)[['Metric', 'Correlation_with_Age']].to_string(index=False))
                f.write("\n\nTop OFIQ metrics correlated with Gender:\n")
                f.write(gender_corr.head(10)[['Metric', 'Correlation_with_Gender']].to_string(index=False))
            
            print(f"\nAnalysis completed successfully! Results saved to {output_dir}")
            results['success'] = True
        else:
            print("\nCorrelation analysis failed or produced no results.")
            
        return results
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        traceback.print_exc()
        return results

def main():
    """Sample main function to run the analysis on multiple datasets"""
    # Define dataset paths and output directories
    datasets = [
        # Example format:
        # (groundtruth_path, ofiq_path, output_dir)
        # ('/home/meem/backup/Age Datasets/afad/groundtruth.csv', '/home/meem/backup/Age Datasets/afad/ofiq.csv', 'plots/afad'),
        #  ('/home/meem/backup/Age Datasets/agedb/groundtruth.csv', '/home/meem/backup/Age Datasets/agedb/ofiq.csv', 'plots/agedb'),
        #  ('/home/meem/backup/Age Datasets/appa-real-release/groundtruth.csv', '/home/meem/backup/Age Datasets/appa-real-release/ofiq.csv', 'plots/appa-real'),
        #  ('/home/meem/backup/Age Datasets/FGNET Dataset/groundtruth.csv', '/home/meem/backup/Age Datasets/FGNET Dataset/ofiq.csv', 'plots/fgnet'),
        #  ('/home/meem/backup/Age Datasets/IMDB - WIKI/groundtruth.csv', '/home/meem/backup/Age Datasets/IMDB - WIKI/ofiq.csv', 'plots/imdb-wiki'),
        #  ('/home/meem/backup/Age Datasets/Juvenile_Dataset/Juvenile_Dataset/groundtruth.csv', '/home/meem/backup/Age Datasets/Juvenile_Dataset/Juvenile_Dataset/ofiq.csv', 'plots/juvenil'),
        #  ('/home/meem/backup/Age Datasets/Morph2 Dataset/groundtruth.csv', '/home/meem/backup/Age Datasets/Morph2 Dataset/ofiq.csv', 'plots/morph2'),
         ('/home/meem/backup/Age Datasets/utkface-cropped/groundtruth.csv', '/home/meem/backup/Age Datasets/utkface-cropped/ofiq.csv', 'plots/utkface'),
        # Add more datasets here following the same pattern
        # ('/path/to/groundtruth2.csv', '/path/to/ofiq2.csv', 'results/dataset2'),
    ]
    
    # Run analysis for each dataset
    results_summary = []
    
    for i, (groundtruth_file, ofiq_file, output_dir) in enumerate(datasets):
        print(f"\n{'='*50}")
        print(f"Processing dataset {i+1}/{len(datasets)}")
        print(f"{'='*50}")
        
        results = analyze_age_gender_ofiq(groundtruth_file, ofiq_file, output_dir)
        results_summary.append(results)
    
    # Print overall summary
    print("\n\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    
    for result in results_summary:
        status = "SUCCESS" if result['success'] else "FAILED"
        print(f"{result['dataset_name']}: {status} - {result['matches_found']} matches found")
    
    print("="*50)

if __name__ == "__main__":
    main()


# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import traceback
from pathlib import Path

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
plt.rcParams['figure.figsize'] = (12, 8)

def load_groundtruth_ranges(groundtruth_file):
    """Load and validate groundtruth data with age ranges"""
    try:
        # First attempt: standard CSV parsing
        gt = pd.read_csv(groundtruth_file)
        
        # Check if required columns exist
        required_cols = ['path', 'lower_age', 'upper_age', 'mean_age', 'gender']
        if all(col in gt.columns for col in required_cols):
            print("Groundtruth data with age ranges loaded successfully")
            return gt
        
        # If columns don't match, try to detect appropriate columns
        print("Searching for age range columns...")
        
        # Map columns to expected names
        col_mapping = {}
        
        # Look for path column
        path_candidates = ['path', 'filepath', 'file_path', 'filename', 'image_path']
        for candidate in path_candidates:
            if candidate in gt.columns:
                col_mapping['path'] = candidate
                break
        else:
            # If no path column found, use the first column if it looks like paths
            first_col = gt.columns[0]
            first_val = str(gt.iloc[0, 0])
            if '/' in first_val or '\\' in first_val or '.jpg' in first_val.lower():
                col_mapping['path'] = first_col
        
        # Look for age range columns
        age_patterns = {
            'lower_age': ['lower_age', 'min_age', 'age_min', 'age_lower', 'start_age'],
            'upper_age': ['upper_age', 'max_age', 'age_max', 'age_upper', 'end_age'],
            'mean_age': ['mean_age', 'avg_age', 'age_mean', 'age_avg', 'center_age']
        }
        
        for target_col, candidates in age_patterns.items():
            for candidate in candidates:
                if candidate in gt.columns:
                    col_mapping[target_col] = candidate
                    break
        
        # Look for gender column
        gender_candidates = ['gender', 'sex', 'gender_label']
        for candidate in gender_candidates:
            if candidate in gt.columns:
                col_mapping['gender'] = candidate
                break
        
        # If we found mappings for all required columns, rename
        if len(col_mapping) >= 4:  # Need at least path, lower_age, upper_age, and gender
            print(f"Found column mappings: {col_mapping}")
            # Rename columns
            gt = gt.rename(columns={v: k for k, v in col_mapping.items()})
            
            # If mean_age not found but we have lower_age and upper_age, calculate it
            if 'mean_age' not in gt.columns and 'lower_age' in gt.columns and 'upper_age' in gt.columns:
                gt['mean_age'] = (gt['lower_age'] + gt['upper_age']) / 2
                print("Calculated mean_age from lower_age and upper_age")
            
            # Convert age columns to numeric
            for age_col in ['lower_age', 'upper_age', 'mean_age']:
                if age_col in gt.columns:
                    gt[age_col] = pd.to_numeric(gt[age_col], errors='coerce')
            
            return gt
        
        # If still not found, try to infer from data
        print("Could not find all required columns. Trying to infer from data...")
        
        # Look for numeric columns that might be age values
        numeric_cols = []
        for col in gt.columns:
            try:
                vals = pd.to_numeric(gt[col], errors='coerce')
                if not vals.isna().all() and vals.mean() > 0 and vals.mean() < 100:
                    numeric_cols.append(col)
            except:
                continue
        
        # If we found 2-3 numeric columns, they might be age range values
        if len(numeric_cols) >= 2:
            print(f"Found potential age columns: {numeric_cols}")
            
            # Sort by mean value to guess lower_age, mean_age, upper_age
            col_means = {col: pd.to_numeric(gt[col], errors='coerce').mean() for col in numeric_cols}
            sorted_cols = sorted(col_means.items(), key=lambda x: x[1])
            
            # Create a new dataframe with appropriate columns
            new_gt = pd.DataFrame()
            
            # Set path column
            if 'path' in col_mapping:
                new_gt['path'] = gt[col_mapping['path']]
            else:
                # Use first column as path if it contains strings
                for col in gt.columns:
                    if gt[col].dtype == 'object':
                        new_gt['path'] = gt[col]
                        break
            
            # Set age columns
            if len(sorted_cols) >= 3:
                new_gt['lower_age'] = pd.to_numeric(gt[sorted_cols[0][0]], errors='coerce')
                new_gt['mean_age'] = pd.to_numeric(gt[sorted_cols[1][0]], errors='coerce')
                new_gt['upper_age'] = pd.to_numeric(gt[sorted_cols[2][0]], errors='coerce')
            elif len(sorted_cols) == 2:
                new_gt['lower_age'] = pd.to_numeric(gt[sorted_cols[0][0]], errors='coerce')
                new_gt['upper_age'] = pd.to_numeric(gt[sorted_cols[1][0]], errors='coerce')
                new_gt['mean_age'] = (new_gt['lower_age'] + new_gt['upper_age']) / 2
            
            # Set gender column
            if 'gender' in col_mapping:
                new_gt['gender'] = gt[col_mapping['gender']]
            else:
                # Look for a column with values like 'm', 'f', 'male', 'female'
                for col in gt.columns:
                    if gt[col].dtype == 'object':
                        sample_vals = gt[col].dropna().astype(str).str.lower().unique()
                        if any(val in ['m', 'f', 'male', 'female'] for val in sample_vals):
                            new_gt['gender'] = gt[col]
                            break
                
                # If still not found, create a placeholder gender column
                if 'gender' not in new_gt.columns:
                    new_gt['gender'] = 'x'  # Unknown gender
            
            return new_gt
            
        # If we still can't identify the columns, raise an error
        raise ValueError(f"Could not identify required columns in {groundtruth_file}")
        
    except Exception as e:
        print(f"Error loading groundtruth with age ranges: {e}")
        traceback.print_exc()
        return None

def plot_age_range_distribution(groundtruth, output_dir):
    """Create age distribution plots for datasets with age ranges"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get unique gender values
        unique_genders = groundtruth['gender'].unique()
        has_males = 'm' in unique_genders
        has_females = 'f' in unique_genders
        has_unknown = any(g for g in unique_genders if g not in ['m', 'f'])
        
        # 1. Box plots of mean_age (three side by side: all, male, female)
        plt.figure(figsize=(18, 6))
        
        # Plot 1: Overall boxplot of mean_age
        plt.subplot(1, 3, 1)
        ax1 = sns.boxplot(y=groundtruth['mean_age'], color='purple')
        plt.title('Overall Mean Age Distribution', fontsize=14)
        plt.ylabel('Mean Age', fontsize=12)
        
        # Add mean and median markers
        mean_overall = groundtruth['mean_age'].mean()
        median_overall = groundtruth['mean_age'].median()
        plt.axhline(mean_overall, color='red', linestyle='--', alpha=0.7)
        plt.axhline(median_overall, color='green', linestyle='-', alpha=0.7)
        plt.text(0.05, mean_overall + 1, f'Mean: {mean_overall:.1f}', color='red', fontweight='bold')
        plt.text(0.05, median_overall - 2, f'Median: {median_overall:.1f}', color='green', fontweight='bold')
        
        # Plots for different genders
        if has_males:
            # Plot 2: Male boxplot
            plt.subplot(1, 3, 2)
            males = groundtruth[groundtruth['gender'] == 'm']
            ax2 = sns.boxplot(y=males['mean_age'], color='blue')
            plt.title('Male Mean Age Distribution', fontsize=14)
            plt.ylabel('Mean Age', fontsize=12)
            
            # Add mean and median markers
            mean_male = males['mean_age'].mean()
            median_male = males['mean_age'].median()
            plt.axhline(mean_male, color='red', linestyle='--', alpha=0.7)
            plt.axhline(median_male, color='green', linestyle='-', alpha=0.7)
            plt.text(0.05, mean_male + 1, f'Mean: {mean_male:.1f}', color='red', fontweight='bold')
            plt.text(0.05, median_male - 2, f'Median: {median_male:.1f}', color='green', fontweight='bold')
        
        if has_females:
            # Plot 3: Female boxplot
            subplot_position = 3 if has_males else 2
            plt.subplot(1, 3, subplot_position)
            females = groundtruth[groundtruth['gender'] == 'f']
            ax3 = sns.boxplot(y=females['mean_age'], color='pink')
            plt.title('Female Mean Age Distribution', fontsize=14)
            plt.ylabel('Mean Age', fontsize=12)
            
            # Add mean and median markers
            mean_female = females['mean_age'].mean()
            median_female = females['mean_age'].median()
            plt.axhline(mean_female, color='red', linestyle='--', alpha=0.7)
            plt.axhline(median_female, color='green', linestyle='-', alpha=0.7)
            plt.text(0.05, mean_female + 1, f'Mean: {mean_female:.1f}', color='red', fontweight='bold')
            plt.text(0.05, median_female - 2, f'Median: {median_female:.1f}', color='green', fontweight='bold')
        
        # Add a legend for mean and median
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, linestyle='--', label='Mean'),
            Line2D([0], [0], color='green', lw=2, linestyle='-', label='Median')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mean_age_boxplots.png'), bbox_inches='tight')
        plt.close()
        
        # 2. Bar charts of age ranges
        # Create age range categories for better visualization
        age_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 100]
        age_labels = [f"{a}-{b-1}" for a, b in zip(age_bins[:-1], age_bins[1:])]
        
        # For lower age
        groundtruth['lower_age_bin'] = pd.cut(groundtruth['lower_age'], bins=age_bins, labels=age_labels)
        # For upper age
        groundtruth['upper_age_bin'] = pd.cut(groundtruth['upper_age'], bins=age_bins, labels=age_labels)
        # For mean age
        groundtruth['mean_age_bin'] = pd.cut(groundtruth['mean_age'], bins=age_bins, labels=age_labels)
        
        # Plot age range distribution
        plt.figure(figsize=(18, 6))
        
        # Overall age range distribution
        plt.subplot(1, 3, 1)
        lower_counts = groundtruth['lower_age_bin'].value_counts().sort_index()
        upper_counts = groundtruth['upper_age_bin'].value_counts().sort_index()
        mean_counts = groundtruth['mean_age_bin'].value_counts().sort_index()
        
        # Get all unique categories across all three variables
        all_categories = sorted(set(lower_counts.index) | set(upper_counts.index) | set(mean_counts.index))
        
        # Convert to DataFrame for easy plotting
        range_df = pd.DataFrame(index=all_categories)
        range_df['Lower Age'] = lower_counts
        range_df['Upper Age'] = upper_counts
        range_df['Mean Age'] = mean_counts
        range_df = range_df.fillna(0)
        
        # Bar chart
        range_df.plot(kind='bar', ax=plt.gca(), width=0.8)
        plt.title('Age Range Distribution (All)', fontsize=14)
        plt.xlabel('Age Range', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title='Age Type')
        
        # Gender-specific distributions
        if has_males:
            plt.subplot(1, 3, 2)
            males = groundtruth[groundtruth['gender'] == 'm']
            
            lower_counts_m = males['lower_age_bin'].value_counts().sort_index()
            upper_counts_m = males['upper_age_bin'].value_counts().sort_index()
            mean_counts_m = males['mean_age_bin'].value_counts().sort_index()
            
            range_df_m = pd.DataFrame(index=all_categories)
            range_df_m['Lower Age'] = lower_counts_m
            range_df_m['Upper Age'] = upper_counts_m
            range_df_m['Mean Age'] = mean_counts_m
            range_df_m = range_df_m.fillna(0)
            
            range_df_m.plot(kind='bar', ax=plt.gca(), width=0.8)
            plt.title('Age Range Distribution (Male)', fontsize=14)
            plt.xlabel('Age Range', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.xticks(rotation=45)
            plt.legend(title='Age Type')
        
        if has_females:
            subplot_idx = 3 if has_males else 2
            plt.subplot(1, 3, subplot_idx)
            females = groundtruth[groundtruth['gender'] == 'f']
            
            lower_counts_f = females['lower_age_bin'].value_counts().sort_index()
            upper_counts_f = females['upper_age_bin'].value_counts().sort_index()
            mean_counts_f = females['mean_age_bin'].value_counts().sort_index()
            
            range_df_f = pd.DataFrame(index=all_categories)
            range_df_f['Lower Age'] = lower_counts_f
            range_df_f['Upper Age'] = upper_counts_f
            range_df_f['Mean Age'] = mean_counts_f
            range_df_f = range_df_f.fillna(0)
            
            range_df_f.plot(kind='bar', ax=plt.gca(), width=0.8)
            plt.title('Age Range Distribution (Female)', fontsize=14)
            plt.xlabel('Age Range', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.xticks(rotation=45)
            plt.legend(title='Age Type')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'age_range_distribution.png'), bbox_inches='tight')
        plt.close()
        
        # 3. Age range width distribution (upper_age - lower_age)
        groundtruth['age_range_width'] = groundtruth['upper_age'] - groundtruth['lower_age']
        
        plt.figure(figsize=(18, 6))
        
        # Overall age range width distribution
        plt.subplot(1, 3, 1)
        sns.histplot(groundtruth['age_range_width'], kde=True, color='purple')
        plt.title('Age Range Width Distribution (All)', fontsize=14)
        plt.xlabel('Age Range Width (years)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        # Add mean and median lines
        mean_width = groundtruth['age_range_width'].mean()
        median_width = groundtruth['age_range_width'].median()
        plt.axvline(mean_width, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_width:.1f}')
        plt.axvline(median_width, color='green', linestyle='-', alpha=0.7, label=f'Median: {median_width:.1f}')
        plt.legend()
        
        # Gender-specific distributions
        if has_males:
            plt.subplot(1, 3, 2)
            males = groundtruth[groundtruth['gender'] == 'm']
            
            sns.histplot(males['age_range_width'], kde=True, color='blue')
            plt.title('Age Range Width Distribution (Male)', fontsize=14)
            plt.xlabel('Age Range Width (years)', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            
            # Add mean and median lines
            mean_width_m = males['age_range_width'].mean()
            median_width_m = males['age_range_width'].median()
            plt.axvline(mean_width_m, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_width_m:.1f}')
            plt.axvline(median_width_m, color='green', linestyle='-', alpha=0.7, label=f'Median: {median_width_m:.1f}')
            plt.legend()
        
        if has_females:
            subplot_idx = 3 if has_males else 2
            plt.subplot(1, 3, subplot_idx)
            females = groundtruth[groundtruth['gender'] == 'f']
            
            sns.histplot(females['age_range_width'], kde=True, color='pink')
            plt.title('Age Range Width Distribution (Female)', fontsize=14)
            plt.xlabel('Age Range Width (years)', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            
            # Add mean and median lines
            mean_width_f = females['age_range_width'].mean()
            median_width_f = females['age_range_width'].median()
            plt.axvline(mean_width_f, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_width_f:.1f}')
            plt.axvline(median_width_f, color='green', linestyle='-', alpha=0.7, label=f'Median: {median_width_f:.1f}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'age_range_width.png'), bbox_inches='tight')
        plt.close()

        # 4. Population Pyramid for mean age (with observed=True to fix the warning)
        # Group by mean age and gender
        males = groundtruth[groundtruth['gender'] == 'm'].groupby('mean_age_bin', observed=True).size()
        females = groundtruth[groundtruth['gender'] == 'f'].groupby('mean_age_bin', observed=True).size()
        
        # Ensure both dataframes have the same index (all age bins)
        all_bins = sorted(set(males.index) | set(females.index))
        males = males.reindex(all_bins, fill_value=0)
        females = females.reindex(all_bins, fill_value=0)
        
        # Create the population pyramid plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Males on left (negative values)
        y_pos = np.arange(len(males.index))
        ax.barh(y_pos, -males.values, height=0.8, color='skyblue', alpha=0.8, label='Male')
        
        # Females on right (positive values)
        ax.barh(y_pos, females.values, height=0.8, color='pink', alpha=0.8, label='Female')
        
        # Set y-tick labels to age ranges
        ax.set_yticks(y_pos)
        ax.set_yticklabels(males.index)
        
        # Formatting
        ax.set_xlabel('Count', fontsize=14)
        ax.set_ylabel('Age Range', fontsize=14)
        ax.set_title('Population Pyramid by Mean Age', fontsize=16)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Fix x-axis to show absolute numbers
        max_count = max(males.max(), females.max()) if not (males.empty or females.empty) else 1
        ax.set_xlim(-max_count*1.1, max_count*1.1)
        
        # Custom x-tick labels to show absolute values
        ticks = np.linspace(0, max_count, 5).astype(int)
        ax.set_xticks(np.concatenate([-ticks[1:], ticks]))
        ax.set_xticklabels([str(abs(x)) for x in np.concatenate([-ticks[1:], ticks])])
        
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'population_pyramid.png'), bbox_inches='tight')
        plt.close()
        
        # 5. Simplified version of the age range distribution by gender to fix the error
        if 'lower_age_bin' in groundtruth.columns and 'upper_age_bin' in groundtruth.columns:
            # Create a simpler visualization that shows age distributions by gender
            plt.figure(figsize=(14, 10))
            
            # Calculate mean age distributions by gender
            if has_males:
                males_df = groundtruth[groundtruth['gender'] == 'm'].copy()
                male_means = males_df['mean_age_bin'].value_counts(normalize=True).sort_index() * 100
                plt.barh(range(len(age_labels)), 
                        -male_means.reindex(age_labels, fill_value=0).values,
                        height=0.8, color='skyblue', alpha=0.6, label='Male')
            
            if has_females:
                females_df = groundtruth[groundtruth['gender'] == 'f'].copy()
                female_means = females_df['mean_age_bin'].value_counts(normalize=True).sort_index() * 100
                plt.barh(range(len(age_labels)), 
                        female_means.reindex(age_labels, fill_value=0).values,
                        height=0.8, color='pink', alpha=0.6, label='Female')
            
            # Formatting
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            plt.yticks(range(len(age_labels)), age_labels)
            plt.xlabel('Percentage (%)', fontsize=14)
            plt.ylabel('Age Range', fontsize=14)
            plt.title('Age Range Distribution by Gender (Percentage)', fontsize=16)
            
            # Fix x-axis to show absolute numbers
            max_pct = 50  # Max percentage to show on either side
            plt.xlim(-max_pct, max_pct)
            
            # Custom x-tick labels to show absolute values
            ticks = np.linspace(0, max_pct, 6)
            plt.xticks(np.concatenate([-ticks[1:], ticks]))
            plt.gca().set_xticklabels([f"{abs(x)}%" for x in np.concatenate([-ticks[1:], ticks])])
            
            plt.legend()
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'age_range_gender_distribution.png'), bbox_inches='tight')
            plt.close()
        
        print("Age range distribution plots created successfully")
        return True
        
    except Exception as e:
        print(f"Error creating age range distribution plots: {e}")
        traceback.print_exc()
        return False

def correlate_with_age_ranges(merged_data, output_dir):
    """Analyze correlations between OFIQ metrics and age ranges (lower, upper, mean)"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert gender to numeric (1 for male, 0 for female)
        if 'gender' in merged_data.columns:
            merged_data['gender_numeric'] = merged_data['gender'].map({'m': 1, 'f': 0})
        
        # Get all potential OFIQ metrics (exclude groundtruth columns and matching columns)
        exclude_cols = ['path', 'lower_age', 'upper_age', 'mean_age', 'gender', 
                         'filename', 'file_id', 'gender_numeric', 
                         'lower_age_bin', 'upper_age_bin', 'mean_age_bin', 'age_range_width']
        potential_metrics = [col for col in merged_data.columns if col not in exclude_cols]
        
        # Find numeric columns only
        numeric_metrics = []
        for col in potential_metrics:
            try:
                pd.to_numeric(merged_data[col])
                numeric_metrics.append(col)
            except:
                continue
        
        print(f"Found {len(numeric_metrics)} numeric OFIQ metrics")
        
        # Calculate correlations with all age measures and gender
        correlations = []
        for metric in numeric_metrics:
            try:
                corr_data = {
                    'Metric': metric,
                }
                
                # Calculate correlations with different age measures
                if 'lower_age' in merged_data.columns:
                    corr_data['Correlation_with_Lower_Age'] = merged_data[['lower_age', metric]].corr().iloc[0, 1]
                
                if 'upper_age' in merged_data.columns:
                    corr_data['Correlation_with_Upper_Age'] = merged_data[['upper_age', metric]].corr().iloc[0, 1]
                
                if 'mean_age' in merged_data.columns:
                    corr_data['Correlation_with_Mean_Age'] = merged_data[['mean_age', metric]].corr().iloc[0, 1]
                
                if 'age_range_width' in merged_data.columns:
                    corr_data['Correlation_with_Age_Range_Width'] = merged_data[['age_range_width', metric]].corr().iloc[0, 1]
                
                if 'gender_numeric' in merged_data.columns:
                    corr_data['Correlation_with_Gender'] = merged_data[['gender_numeric', metric]].corr().iloc[0, 1]
                
                correlations.append(corr_data)
            except:
                # Skip metrics that cause problems
                continue
        
        # Create correlation dataframe
        corr_df = pd.DataFrame(correlations)
        
        # Save all correlations to CSV
        corr_df.to_csv(os.path.join(output_dir, 'all_correlations.csv'), index=False)
        
        # Create separate sorted dataframes for each age measure
        if 'Correlation_with_Lower_Age' in corr_df.columns:
            lower_age_corr_df = corr_df.sort_values('Correlation_with_Lower_Age', key=abs, ascending=False)
            lower_age_corr_df.to_csv(os.path.join(output_dir, 'lower_age_correlations.csv'), index=False)
            
            # Plot top correlations with lower age
            plt.figure(figsize=(12, 8))
            top_n = min(15, len(lower_age_corr_df))
            
            if top_n > 0:
                top_lower_age = lower_age_corr_df.head(top_n)
                sns.barplot(x='Correlation_with_Lower_Age', y='Metric', data=top_lower_age)
                plt.title(f'Top {top_n} OFIQ Metrics Correlated with Lower Age', fontsize=16)
                plt.xlabel('Correlation Coefficient', fontsize=14)
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'lower_age_correlation.png'), bbox_inches='tight')
            plt.close()
        
        if 'Correlation_with_Upper_Age' in corr_df.columns:
            upper_age_corr_df = corr_df.sort_values('Correlation_with_Upper_Age', key=abs, ascending=False)
            upper_age_corr_df.to_csv(os.path.join(output_dir, 'upper_age_correlations.csv'), index=False)
            
            # Plot top correlations with upper age
            plt.figure(figsize=(12, 8))
            top_n = min(15, len(upper_age_corr_df))
            
            if top_n > 0:
                top_upper_age = upper_age_corr_df.head(top_n)
                sns.barplot(x='Correlation_with_Upper_Age', y='Metric', data=top_upper_age)
                plt.title(f'Top {top_n} OFIQ Metrics Correlated with Upper Age', fontsize=16)
                plt.xlabel('Correlation Coefficient', fontsize=14)
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'upper_age_correlation.png'), bbox_inches='tight')
            plt.close()
        
        if 'Correlation_with_Mean_Age' in corr_df.columns:
            mean_age_corr_df = corr_df.sort_values('Correlation_with_Mean_Age', key=abs, ascending=False)
            mean_age_corr_df.to_csv(os.path.join(output_dir, 'mean_age_correlations.csv'), index=False)
            
            # Plot top correlations with mean age
            plt.figure(figsize=(12, 8))
            top_n = min(15, len(mean_age_corr_df))
            
            if top_n > 0:
                top_mean_age = mean_age_corr_df.head(top_n)
                sns.barplot(x='Correlation_with_Mean_Age', y='Metric', data=top_mean_age)
                plt.title(f'Top {top_n} OFIQ Metrics Correlated with Mean Age', fontsize=16)
                plt.xlabel('Correlation Coefficient', fontsize=14)
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'mean_age_correlation.png'), bbox_inches='tight')
            plt.close()
        
        if 'Correlation_with_Age_Range_Width' in corr_df.columns:
            width_corr_df = corr_df.sort_values('Correlation_with_Age_Range_Width', key=abs, ascending=False)
            width_corr_df.to_csv(os.path.join(output_dir, 'age_range_width_correlations.csv'), index=False)
            
            # Plot top correlations with age range width
            plt.figure(figsize=(12, 8))
            top_n = min(15, len(width_corr_df))
            
            if top_n > 0:
                top_width = width_corr_df.head(top_n)
                sns.barplot(x='Correlation_with_Age_Range_Width', y='Metric', data=top_width)
                plt.title(f'Top {top_n} OFIQ Metrics Correlated with Age Range Width', fontsize=16)
                plt.xlabel('Correlation Coefficient', fontsize=14)
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'age_range_width_correlation.png'), bbox_inches='tight')
            plt.close()
        
        if 'Correlation_with_Gender' in corr_df.columns:
            gender_corr_df = corr_df.sort_values('Correlation_with_Gender', key=abs, ascending=False)
            gender_corr_df.to_csv(os.path.join(output_dir, 'gender_correlations.csv'), index=False)
            
            # Plot top correlations with gender
            plt.figure(figsize=(12, 8))
            top_n = min(15, len(gender_corr_df))
            
            if top_n > 0:
                top_gender = gender_corr_df.head(top_n)
                sns.barplot(x='Correlation_with_Gender', y='Metric', data=top_gender)
                plt.title(f'Top {top_n} OFIQ Metrics Correlated with Gender', fontsize=16)
                plt.xlabel('Correlation Coefficient (1=Male, 0=Female)', fontsize=14)
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'gender_correlation.png'), bbox_inches='tight')
            plt.close()
        
        # Create heatmap with correlations to all age measures
        # Select columns for heatmap
        heatmap_columns = []
        if 'Correlation_with_Lower_Age' in corr_df.columns:
            heatmap_columns.append('Correlation_with_Lower_Age')
        if 'Correlation_with_Upper_Age' in corr_df.columns:
            heatmap_columns.append('Correlation_with_Upper_Age')
        if 'Correlation_with_Mean_Age' in corr_df.columns:
            heatmap_columns.append('Correlation_with_Mean_Age')
        if 'Correlation_with_Age_Range_Width' in corr_df.columns:
            heatmap_columns.append('Correlation_with_Age_Range_Width')
        if 'Correlation_with_Gender' in corr_df.columns:
            heatmap_columns.append('Correlation_with_Gender')
        
        if heatmap_columns:
            # Get top metrics across all age measures
            top_metrics = set()
            for col in heatmap_columns:
                sorted_df = corr_df.sort_values(col, key=abs, ascending=False)
                top_metrics.update(sorted_df.head(5)['Metric'].tolist())
            
            top_metrics = list(top_metrics)
            
            # Create correlation matrix
            if top_metrics:
                heatmap_data = []
                for metric in top_metrics:
                    row = {'Metric': metric}
                    for col in heatmap_columns:
                        metric_corr = corr_df[corr_df['Metric'] == metric][col].values[0]
                        row[col.replace('Correlation_with_', '')] = metric_corr
                    heatmap_data.append(row)
                
                heatmap_df = pd.DataFrame(heatmap_data)
                heatmap_df = heatmap_df.set_index('Metric')
                
                # Plot heatmap
                plt.figure(figsize=(12, len(top_metrics) * 0.5 + 3))
                sns.heatmap(heatmap_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
                plt.title('Correlation Between OFIQ Metrics and Age Measures', fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'age_correlation_heatmap.png'), bbox_inches='tight')
                plt.close()
        
        print("Correlation analysis completed successfully")
        return corr_df
        
    except Exception as e:
        print(f"Error in correlation analysis: {e}")
        traceback.print_exc()
        return pd.DataFrame()

def analyze_age_range_ofiq(groundtruth_file, ofiq_file, output_dir):
    """
    Main function to analyze age ranges and OFIQ data
    
    Parameters:
    -----------
    groundtruth_file : str
        Path to the groundtruth CSV file containing path, lower_age, upper_age, mean_age, gender columns
    ofiq_file : str
        Path to the OFIQ CSV file containing quality metrics
    output_dir : str
        Directory to save output visualizations and CSV files
    
    Returns:
    --------
    dict
        Dictionary with analysis results and status
    """
    results = {
        'success': False,
        'dataset_name': Path(groundtruth_file).stem,
        'metrics_found': 0,
        'matches_found': 0,
        'has_age_ranges': True  # Flag that this uses age ranges instead of single age value
    }
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Load datasets
        print(f"\nAnalyzing dataset with age ranges: {results['dataset_name']}")
        print(f"Loading groundtruth from: {groundtruth_file}")
        groundtruth = load_groundtruth_ranges(groundtruth_file)
        
        if groundtruth is None or groundtruth.empty:
            print(f"Failed to load groundtruth data from {groundtruth_file}")
            return results
        
        # Ensure all required columns exist
        required_cols = ['path', 'lower_age', 'upper_age', 'mean_age', 'gender']
        missing_cols = [col for col in required_cols if col not in groundtruth.columns]
        
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            # Try to fix missing columns if possible
            if 'lower_age' in groundtruth.columns and 'upper_age' in groundtruth.columns and 'mean_age' not in groundtruth.columns:
                groundtruth['mean_age'] = (groundtruth['lower_age'] + groundtruth['upper_age']) / 2
                print("Calculated mean_age from lower_age and upper_age")
            elif 'mean_age' in groundtruth.columns and 'lower_age' not in groundtruth.columns and 'upper_age' not in groundtruth.columns:
                # If we only have mean_age, create dummy lower and upper bounds
                groundtruth['lower_age'] = groundtruth['mean_age'] - 2.5  # Assume ±2.5 years range
                groundtruth['upper_age'] = groundtruth['mean_age'] + 2.5
                print("Created dummy lower_age and upper_age from mean_age")
            
            # Check again after fixes
            missing_cols = [col for col in required_cols if col not in groundtruth.columns]
            if missing_cols:
                print(f"Still missing required columns after fixes: {missing_cols}")
                return results
        
        print(f"Loaded groundtruth data: {groundtruth.shape[0]} records")
        
        print(f"Loading OFIQ data from: {ofiq_file}")
        ofiq = load_ofiq(ofiq_file)
        
        if ofiq is None or ofiq.empty:
            print(f"Failed to load OFIQ data from {ofiq_file}")
            return results
            
        print(f"Loaded OFIQ data: {ofiq.shape[0]} records")
        results['metrics_found'] = ofiq.shape[1]
        
        # Step 2: EDA on groundtruth data
        print("\nCreating age range distribution visualizations...")
        plot_age_range_distribution(groundtruth, output_dir)
        
        # Step 3: Match datasets
        print("\nMatching groundtruth and OFIQ data...")
        merged_data = match_datasets(groundtruth, ofiq)
        
        if merged_data.empty:
            print("Failed to match datasets.")
            return results
            
        results['matches_found'] = merged_data.shape[0]
        print(f"Successfully matched {merged_data.shape[0]} records")
        
        # Add age range width
        merged_data['age_range_width'] = merged_data['upper_age'] - merged_data['lower_age']
        
        # Step 4: Correlation analysis
        print("\nPerforming correlation analysis...")
        corr_df = correlate_with_age_ranges(merged_data, output_dir)
        
        # Step 5: Display results
        if not corr_df.empty:
            # Show top correlations for each age measure
            print("\nTop OFIQ metrics correlated with Age Measures:")
            
            for col in corr_df.columns:
                if col.startswith('Correlation_with_'):
                    measure = col.replace('Correlation_with_', '')
                    print(f"\nTop 5 correlations with {measure}:")
                    sorted_df = corr_df.sort_values(col, key=abs, ascending=False)
                    print(sorted_df.head(5)[['Metric', col]].to_string(index=False))
            
            # Create a summary file
            with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
                f.write(f"Dataset with Age Ranges: {results['dataset_name']}\n")
                f.write(f"Groundtruth records: {groundtruth.shape[0]}\n")
                f.write(f"OFIQ records: {ofiq.shape[0]}\n")
                f.write(f"Matched records: {merged_data.shape[0]}\n\n")
                
                for col in corr_df.columns:
                    if col.startswith('Correlation_with_'):
                        measure = col.replace('Correlation_with_', '')
                        f.write(f"Top 10 correlations with {measure}:\n")
                        sorted_df = corr_df.sort_values(col, key=abs, ascending=False)
                        f.write(sorted_df.head(10)[['Metric', col]].to_string(index=False))
                        f.write("\n\n")
            
            print(f"\nAnalysis completed successfully! Results saved to {output_dir}")
            results['success'] = True
        else:
            print("\nCorrelation analysis failed or produced no results.")
            
        return results
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        traceback.print_exc()
        return results

def main():
    """Sample main function to run the analysis on datasets with age ranges"""
    # Define dataset paths and output directories
    datasets = [
        # Example format:
        # (groundtruth_path, ofiq_path, output_dir)
        ('/home/meem/backup/Age Datasets/AdienceGender Dataset/groundtruth.csv', '/home/meem/backup/Age Datasets/AdienceGender Dataset/ofiq.csv', 'plots/adience'),
        ('/home/meem/backup/Age Datasets/fairface/groundtruth.csv', '/home/meem/backup/Age Datasets/fairface/ofiq.csv', 'plots/fairface'),
        ('/home/meem/backup/Age Datasets/Groups-of-People Dataset/groundtruth.csv', '/home/meem/backup/Age Datasets/Groups-of-People Dataset/ofiq.csv', 'plots/groups'),
        #  
        # Add more datasets here following the same pattern
    ]
    
    # Run analysis for each dataset
    results_summary = []
    
    for i, (groundtruth_file, ofiq_file, output_dir) in enumerate(datasets):
        print(f"\n{'='*50}")
        print(f"Processing dataset with age ranges {i+1}/{len(datasets)}")
        print(f"{'='*50}")
        
        results = analyze_age_range_ofiq(groundtruth_file, ofiq_file, output_dir)
        results_summary.append(results)
    
    # Print overall summary
    print("\n\n" + "="*50)
    print("ANALYSIS SUMMARY FOR DATASETS WITH AGE RANGES")
    print("="*50)
    
    for result in results_summary:
        status = "SUCCESS" if result['success'] else "FAILED"
        print(f"{result['dataset_name']}: {status} - {result['matches_found']} matches found")
    
    print("="*50)

if __name__ == "__main__":
    main()


# In[22]:


print("hello")

