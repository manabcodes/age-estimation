#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.gridspec as gridspec


# In[20]:


def create_comprehensive_metric_tileboards(csv_file, output_dir='metric_tileboards'):
    """
    Create tileboards comparing outliers vs acceptable examples for each quality metric.
    Each tileboard uses a 2×6 grid (12 images total: 6 outliers, 6 acceptable).
    Ensures no image is reused across different tileboards or within the same tileboard.
    Prioritizes scalar versions of metrics when available.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data - using semicolon as separator
    df = pd.read_csv(csv_file, sep=';')
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Check if Filename column exists
    if 'Filename' not in df.columns:
        print("Error: Filename column not found in the CSV file")
        return
    
    # Define quality metrics with their descriptions and thresholds
    METRICS_INFO = {
        'UnifiedQualityScore': {
            'description': 'Overall quality (0–100). 100=optimal.',
            'threshold': '≥60–80 depending on application',
            'lower_is_better': False,
            'ideal': 100
        },
        'BackgroundUniformity': {
            'description': 'Uniform, light background (no clutter).',
            'threshold': '≥80',
            'lower_is_better': False,
            'ideal': 100
        },
        'IlluminationUniformity': {
            'description': 'Even lighting (no strong shadows).',
            'threshold': '≥80; <1 EV difference across face',
            'lower_is_better': True,
            'ideal': 100
        },
        'LuminanceMean': {
            'description': 'Brightness (normalized).',
            'threshold': '~50; reject extremes (<30 or >70)',
            'lower_is_better': False,
            'ideal': 50
        },
        'LuminanceVariance': {
            'description': 'Contrast/lighting variance.',
            'threshold': 'Moderate-high',
            'lower_is_better': False,
            'ideal': 50  # Assuming mid-range is ideal
        },
        'UnderExposurePrevention': {
            'description': 'No clipped dark areas.',
            'threshold': '≥90',
            'lower_is_better': False,
            'ideal': 100
        },
        'OverExposurePrevention': {
            'description': 'No clipped bright areas.',
            'threshold': '≥90',
            'lower_is_better': False,
            'ideal': 100
        },
        'DynamicRange': {
            'description': 'Captured contrast range.',
            'threshold': '≥30',
            'lower_is_better': False,
            'ideal': 100
        },
        'Sharpness': {
            'description': 'Focus measure (Laplacian).',
            'threshold': '≥60–70',
            'lower_is_better': False,
            'ideal': 100
        },
        'CompressionArtifacts': {
            'description': 'JPEG artifact detection (0–100).',
            'threshold': '≥80–90',
            'lower_is_better': False,  # Lower artifacts is better
            'ideal': 0
        },
        'NaturalColour': {
            'description': 'Color fidelity (skin tone neutral).',
            'threshold': '≥80',
            'lower_is_better': False,
            'ideal': 100
        },
        'SingleFacePresent': {
            'description': 'Exactly one face in frame.',
            'threshold': 'Must be 100 (one face)',
            'lower_is_better': False,
            'ideal': 100
        },
        'EyesOpen': {
            'description': 'Eyes open (yes/no).',
            'threshold': 'Must be 100 (eyes open)',
            'lower_is_better': False,
            'ideal': 100
        },
        'MouthClosed': {
            'description': 'Mouth closed (neutral).',
            'threshold': 'Must be 100 (mouth closed)',
            'lower_is_better': False,
            'ideal': 100
        },
        'EyesVisible': {
            'description': 'Eyes unobstructed (no glare/hair).',
            'threshold': '≥90',
            'lower_is_better': False,
            'ideal': 100
        },
        'MouthOcclusionPrevention': {
            'description': 'No objects covering mouth.',
            'threshold': '≈100',
            'lower_is_better': False,
            'ideal': 100
        },
        'FaceOcclusionPrevention': {
            'description': 'No face occlusion (hair, hands, etc.).',
            'threshold': '≈100',
            'lower_is_better': False,
            'ideal': 100
        },
        'InterEyeDistance': {
            'description': 'Eye distance (pixels or % of width).',
            'threshold': '>60 px at low res; or ≥0.15 image height',
            'lower_is_better': False,
            'ideal': 70  # Estimate based on threshold
        },
        'HeadSize': {
            'description': 'Face height relative to image.',
            'threshold': 'Face ~70–80% image height',
            'lower_is_better': False,
            'ideal': 75  # Midpoint of recommended range
        },
        'LeftwardCropOfFaceImage': {
            'description': 'Portion of image left of face bounding box.',
            'threshold': 'Small (face centered)',
            'lower_is_better': True,  # Smaller crop is better (centered)
            'ideal': 0
        },
        'RightwardCropOfFaceImage': {
            'description': 'Portion of image right of face box.',
            'threshold': 'Small (face centered)',
            'lower_is_better': True,  # Smaller crop is better (centered)
            'ideal': 0
        },
        'MarginAboveOfFaceImage': {
            'description': 'Top margin above head (pixels/%).',
            'threshold': '~5–15% image height',
            'lower_is_better': False,  # Within range is better
            'ideal': 10  # Midpoint of recommended range
        },
        'MarginBelowOfFaceImage': {
            'description': 'Bottom margin (pixels/%).',
            'threshold': '~10–20%',
            'lower_is_better': False,  # Within range is better
            'ideal': 15  # Midpoint of recommended range
        },
        'HeadPoseYaw': {
            'description': 'Rotation around vertical axis (± deg).',
            'threshold': 'Ideally <5°; tolerate up to ~15–20° (score≥70)',
            'lower_is_better': False,  # In your data, 100 is best (no rotation)
            'ideal': 100
        },
        'HeadPosePitch': {
            'description': 'Rotation around horizontal axis (± deg).',
            'threshold': 'Ideally <5°; tolerate up to ~15–20°',
            'lower_is_better': False,  # In your data, 100 is best (no rotation)
            'ideal': 100
        },
        'HeadPoseRoll': {
            'description': 'In-plane tilt (± deg).',
            'threshold': 'Ideally <5°; tolerate up to ~10–15°',
            'lower_is_better': False,  # In your data, 100 is best (no rotation)
            'ideal': 100
        },
        'ExpressionNeutrality': {
            'description': 'Neutral (no smile, eyes open).',
            'threshold': 'Must be high (≈100)',
            'lower_is_better': False,
            'ideal': 100
        },
        'NoHeadCoverings': {
            'description': 'No hats/scarves (except religious without obscuring).',
            'threshold': 'Must be 100 (no covering)',
            'lower_is_better': False,
            'ideal': 100
        }
    }
    
    # Check for available metrics and prefer scalar versions
    available_metrics = []
    
    for base_metric in METRICS_INFO.keys():
        scalar_version = f"{base_metric}.scalar"
        
        # Check if scalar version exists, otherwise use base version
        if scalar_version in df.columns:
            print(f"Using scalar version for {base_metric}")
            available_metrics.append({
                'name': scalar_version,
                'display_name': f"{base_metric} (scalar)",
                'base_metric': base_metric,
                'is_scalar': True
            })
        elif base_metric in df.columns:
            print(f"Using base version for {base_metric}")
            available_metrics.append({
                'name': base_metric,
                'display_name': base_metric,
                'base_metric': base_metric,
                'is_scalar': False
            })
    
    if not available_metrics:
        print("Error: None of the expected quality metrics were found in the CSV file")
        return
    
    print(f"Will create tileboards for {len(available_metrics)} metrics")
    
    # Check if we can access the image files
    can_access_images = False
    sample_path = df['Filename'].iloc[0]
    try:
        # Try to load the first image to see if paths are accessible
        img = Image.open(sample_path)
        can_access_images = True
        img.close()
    except Exception as e:
        print(f"Warning: Could not open image at {sample_path}")
        print(f"Error: {e}")
        print("Will create tileboards with placeholders instead of actual images")
    
    # Create a set to track used images across tileboards
    used_images = set()
    
    # Create a tileboard for each metric
    for metric_info in available_metrics:
        metric_name = metric_info['name']
        base_metric = metric_info['base_metric']
        display_name = metric_info['display_name']
        
        print(f"Creating tileboard for {display_name}...")
        
        # Get relevant threshold information for this metric
        base_metric_info = METRICS_INFO[base_metric]
        lower_is_better = base_metric_info['lower_is_better']
        ideal_val = base_metric_info['ideal']
        
        # Create a copy of the dataframe excluding already used images
        metric_df = df.copy()
        if used_images:
            metric_df = metric_df[~metric_df['Filename'].isin(used_images)]
            print(f"  Excluded {len(used_images)} already used images")
        
        # Ensure values are clipped at 0 (not -1)
        metric_df[metric_name] = metric_df[metric_name].clip(lower=0)
        
        # Check if we have enough images to create the tileboard
        if len(metric_df) < 12:  # We need at least 12 images (6 outliers, 6 acceptable)
            print(f"  Warning: Not enough unique images left for {display_name}. Skipping...")
            continue
        
        # For pose metrics, try to find examples where the other pose metrics are good
        if base_metric in ['HeadPoseYaw', 'HeadPosePitch', 'HeadPoseRoll']:
            # Get the names of the other pose metrics
            pose_metrics = ['HeadPoseYaw', 'HeadPosePitch', 'HeadPoseRoll']
            other_pose_metrics = [m for m in pose_metrics if m != base_metric]
            
            # Find the equivalent column names in the dataframe
            other_pose_columns = []
            for m in other_pose_metrics:
                if m in metric_df.columns:
                    other_pose_columns.append(m)
                elif f"{m}.scalar" in metric_df.columns:
                    other_pose_columns.append(f"{m}.scalar")
            
            # If we found the other pose metrics, filter to images where they're good
            if other_pose_columns:
                # Define a threshold for "good" (close to 100)
                good_pose_threshold = 80
                
                # Create a mask for images with good values for the other pose metrics
                good_pose_mask = pd.Series(True, index=metric_df.index)
                for col in other_pose_columns:
                    good_pose_mask = good_pose_mask & (metric_df[col] >= good_pose_threshold)
                
                # Check if we have enough images after filtering
                if good_pose_mask.sum() >= 12:
                    print(f"  Filtering to images with good values for other pose metrics")
                    metric_df = metric_df[good_pose_mask]
                else:
                    print(f"  Not enough images with good values for other pose metrics. Using all available.")
        
        # Sort by the metric
        if lower_is_better:
            # For lower-is-better metrics, higher values are outliers
            sorted_df = metric_df.sort_values(by=metric_name, ascending=False)
        else:
            # For higher-is-better metrics, lower values are outliers
            sorted_df = metric_df.sort_values(by=metric_name, ascending=True)
        
        # Get outlier examples (first 6 after sorting, ensuring no duplicates)
        outlier_filenames = []
        outlier_examples = []
        
        for _, row in sorted_df.iterrows():
            if len(outlier_examples) >= 6:
                break
            
            filename = row['Filename']
            if filename not in outlier_filenames:
                outlier_filenames.append(filename)
                outlier_examples.append(row)
        
        # Convert to DataFrame
        outlier_examples = pd.DataFrame(outlier_examples)
        
        # Check if we have enough outlier examples
        if len(outlier_examples) < 6:
            print(f"  Warning: Not enough unique outlier examples for {display_name}. Got {len(outlier_examples)}.")
            if len(outlier_examples) < 3:  # Skip if too few
                print(f"  Skipping {display_name} due to insufficient outlier examples.")
                continue
        
        # Get acceptable examples (from the other end of sorted dataframe, ensuring no duplicates)
        acceptable_filenames = []
        acceptable_examples = []
        
        for _, row in sorted_df.iloc[::-1].iterrows():
            if len(acceptable_examples) >= 6:
                break
            
            filename = row['Filename']
            if filename not in outlier_filenames and filename not in acceptable_filenames:
                acceptable_filenames.append(filename)
                acceptable_examples.append(row)
        
        # Convert to DataFrame
        acceptable_examples = pd.DataFrame(acceptable_examples)
        
        # Check if we have enough acceptable examples
        if len(acceptable_examples) < 6:
            print(f"  Warning: Not enough unique acceptable examples for {display_name}. Got {len(acceptable_examples)}.")
            if len(acceptable_examples) < 3:  # Skip if too few
                print(f"  Skipping {display_name} due to insufficient acceptable examples.")
                continue
        
        # Track the images we're using in this tileboard
        tileboard_images = set(outlier_filenames + acceptable_filenames)
        
        # Add these to our global set of used images
        used_images.update(tileboard_images)
        
        # Create the figure
        plt.figure(figsize=(15, 8))
        
        # Define a grid for better layout control
        gs = gridspec.GridSpec(2, 6, figure=plt.gcf())
        
        # Calculate the color ranges for visual reference
        if lower_is_better:
            outlier_color = 'red'
            acceptable_color = 'green'
            color_map = plt.cm.RdYlGn_r  # Reversed (red is high, green is low)
        else:
            outlier_color = 'red'
            acceptable_color = 'green'
            color_map = plt.cm.RdYlGn    # Normal (red is low, green is high)
        
        # Get the overall range for normalization
        min_val = max(0, df[metric_name].min())  # Clip at 0
        max_val = df[metric_name].max()
        
        # Function to normalize metric value to color
        def get_color_for_value(val):
            if max_val - min_val <= 0:
                return color_map(0.5)  # Default to middle if no range
            
            if lower_is_better:
                # For lower-is-better, values closer to 0 are greener
                normalized = 1 - (val - min_val) / (max_val - min_val)
            else:
                normalized = (val - min_val) / (max_val - min_val)
            
            return color_map(normalized)
        
        # Plot outlier examples (top row)
        for i, (_, row) in enumerate(outlier_examples.iterrows()):
            if i >= 6:  # Limit to 6 examples
                break
                
            ax = plt.subplot(gs[0, i])
            
            img_path = row['Filename']
            metric_value = row[metric_name]
            
            # Display the image or a colored placeholder
            if can_access_images:
                try:
                    img = mpimg.imread(img_path)
                    plt.imshow(img)
                except Exception as e:
                    plt.axhspan(0, 1, color=get_color_for_value(metric_value), alpha=0.7)
                    plt.axis('off')
                    print(f"  Warning: Could not load image {img_path}: {e}")
            else:
                plt.axhspan(0, 1, color=get_color_for_value(metric_value), alpha=0.7)
                plt.axis('off')
            
            # Add image filename and metric value as title
            filename = Path(img_path).name
            if len(filename) > 20:  # Truncate long filenames
                filename = filename[:17] + '...'
            
            plt.title(f"OUTLIER\n{filename}\n{metric_value:.1f}\n(Ideal: {ideal_val})", 
                     fontsize=8, color=outlier_color)
        
        # Plot acceptable examples (bottom row)
        for i, (_, row) in enumerate(acceptable_examples.iterrows()):
            if i >= 6:  # Limit to 6 examples
                break
                
            ax = plt.subplot(gs[1, i])
            
            img_path = row['Filename']
            metric_value = row[metric_name]
            
            # Display the image or a colored placeholder
            if can_access_images:
                try:
                    img = mpimg.imread(img_path)
                    plt.imshow(img)
                except Exception as e:
                    plt.axhspan(0, 1, color=get_color_for_value(metric_value), alpha=0.7)
                    plt.axis('off')
                    print(f"  Warning: Could not load image {img_path}: {e}")
            else:
                plt.axhspan(0, 1, color=get_color_for_value(metric_value), alpha=0.7)
                plt.axis('off')
            
            # Add image filename and metric value as title
            filename = Path(img_path).name
            if len(filename) > 20:  # Truncate long filenames
                filename = filename[:17] + '...'
                
            plt.title(f"ACCEPTABLE\n{filename}\n{metric_value:.1f}\n(Ideal: {ideal_val})", 
                     fontsize=8, color=acceptable_color)
        
        # Add a main title and detailed information
        plt.suptitle(f"{display_name}\n{base_metric_info['description']}", fontsize=16)
        
        # Add a footer with threshold information
        plt.figtext(0.5, 0.01, 
                   f"Pass Threshold: {base_metric_info['threshold']} | {'Lower values are better' if lower_is_better else 'Higher values are better'}", 
                   ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        
        # Save the figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])  # Make room for the suptitle and footer
        output_file = os.path.join(output_dir, f"{metric_name.replace('.', '_')}_comparison.png")
        plt.savefig(output_file, dpi=200)
        plt.close()
        
        print(f"  ✓ Saved to {output_file}")
    
    print(f"\nAll tileboards complete. Check {output_dir} directory for results.")
    print(f"Used {len(used_images)} unique images across all tileboards.")


# In[36]:


def main():
    """Main function to run the tileboard creation"""
    # Specify your CSV file path here
    csv_file = '/home/meem/backup/Age Datasets/AdienceGender Dataset/ofiq.csv'  # Update with your file path
    output_dir = 'tile/adience'  # Output directory for tileboards
    
    # Create the comprehensive comparison tileboards
    create_comprehensive_metric_tileboards(csv_file, output_dir)

    # """Main function to run the visualization code"""
    # #1
    # # Specify your CSV file path here
    csv_file = '/home/meem/backup/Age Datasets/fairface/all_results.csv'  # Update with your file path
    output_dir = 'tile/fairface'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    create_comprehensive_metric_tileboards(csv_file, output_dir)

    # #2
    csv_file = '/home/meem/backup/Age Datasets/afad/ofiq.csv'  # Update with your file path
    output_dir = 'tile/afad'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    create_comprehensive_metric_tileboards(csv_file, output_dir)

    # #4    
    csv_file = '/home/meem/backup/Age Datasets/agedb/ofiq.csv'  # Update with your file path
    output_dir = 'tile/agedb'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    create_comprehensive_metric_tileboards(csv_file, output_dir)

    # #5    
    csv_file = '/home/meem/backup/Age Datasets/appa-real-release/fixed_ofiq.csv'  # Update with your file path
    output_dir = 'tile/appa-real'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    create_comprehensive_metric_tileboards(csv_file, output_dir)

    # #6    
    csv_file = '/home/meem/backup/Age Datasets/FGNET Dataset/ofiq.csv'  # Update with your file path
    output_dir = 'tile/fgnet'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    create_comprehensive_metric_tileboards(csv_file, output_dir)

    # # #7    
    csv_file = '/home/meem/backup/Age Datasets/Groups-of-People Dataset/ofiq.csv'  # Update with your file path
    output_dir = 'tile/groups'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    create_comprehensive_metric_tileboards(csv_file, output_dir)

    # #8    
    csv_file = '/home/meem/backup/Age Datasets/IMDB - WIKI/ofiq.csv'  # Update with your file path
    output_dir = 'tile/imdb-wiki'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    create_comprehensive_metric_tileboards(csv_file, output_dir)

    # #9     
    csv_file = '/home/meem/backup/Age Datasets/Juvenile_Dataset/Juvenile_Dataset/ofiq.csv'  # Update with your file path
    output_dir = 'tile/juvenil'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    create_comprehensive_metric_tileboards(csv_file, output_dir)  # Update with your file path


    # #10     
    csv_file = '/home/meem/backup/Age Datasets/lagenda/lagenda_seg_results.csv'  # Update with your file path
    output_dir = 'tile/lagenda'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    create_comprehensive_metric_tileboards(csv_file, output_dir)  # Update with your file path

    # #11     
    csv_file = '/home/meem/backup/Age Datasets/Morph2 Dataset/ofiq.csv'  # Update with your file path
    output_dir = 'tile/morph2'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    create_comprehensive_metric_tileboards(csv_file, output_dir)  # Update with your file path

    # #12     
    csv_file = '/home/meem/backup/Age Datasets/utkface-cropped/all_results.csv'  # Update with your file path
    output_dir = 'tile/utkface'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    create_comprehensive_metric_tileboards(csv_file, output_dir)  # Update with your file path


if __name__ == "__main__":
    main()


# In[30]:


get_ipython().system("find '/home/meem/backup/Age Datasets/utkface-cropped/all-results.csv'")


# In[35]:


get_ipython().system("find '/home/meem/backup/Age Datasets/utkface-cropped/all_results.csv'")

