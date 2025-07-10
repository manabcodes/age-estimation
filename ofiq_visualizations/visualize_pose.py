#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from matplotlib.patches import Wedge, Rectangle, Circle, Arc, Arrow
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats


# In[5]:


get_ipython().system('head -10 "/home/meem/backup/Age Datasets/lagenda/ofiq.csv"')


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

def visualize_pose_angles(csv_file, output_dir='pose_visualizations'):
    """
    Visualize head pose angles with semicircular histograms and a 3D scatter plot
    showing angles on a unit sphere.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    df = pd.read_csv(csv_file, sep=';')
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Define pose metrics - use these exact columns, no conversions
    pose_metrics = ['HeadPoseYaw', 'HeadPosePitch', 'HeadPoseRoll']
    
    # Check if metrics exist in the dataset
    available_metrics = [m for m in pose_metrics if m in df.columns]
    if not available_metrics:
        print("Error: No pose metrics found in the dataset")
        return
    
    # Define angle ranges for visualization
    angle_ranges = {
        'HeadPoseYaw': (-90, 90),   # Side-to-side rotation
        'HeadPosePitch': (-60, 60), # Up-down rotation
        'HeadPoseRoll': (-45, 45)   # In-plane tilt
    }
    
    # Create semicircular histograms
    print("Creating enhanced semicircular histograms...")
    fig = plt.figure(figsize=(18, 8))
    
    # Process each metric
    for i, metric in enumerate(available_metrics, 1):
        print(f"Processing {metric}")
        
        # Get the raw angle data directly
        angles = df[metric].values
        
        # Calculate actual statistics
        mean_angle = np.mean(angles)
        median_angle = np.median(angles)
        std_angle = np.std(angles)
        
        print(f"  {metric} statistics: Mean={mean_angle:.2f}, Median={median_angle:.2f}, Std={std_angle:.2f}")
        
        # Use appropriate range for this metric
        angle_range = angle_ranges.get(metric, (-45, 45))
        
        # Create subplot
        ax = fig.add_subplot(1, len(available_metrics), i, projection='polar')
        
        # Create histogram
        n_bins = 36
        hist, bin_edges = np.histogram(angles, bins=n_bins, range=angle_range)
        
        # Convert bin edges to radians (0° at top)
        bin_radians = np.radians(bin_edges[:-1] + 90)
        width = np.radians((angle_range[1] - angle_range[0]) / n_bins)
        
        # Get maximum frequency value
        max_freq = max(hist)
        
        # Create bars
        bars = ax.bar(bin_radians, hist, width=width, alpha=0.7)
        
        # Color bars based on deviation from 0 (ideal)
        for bar, angle_edge in zip(bars, bin_edges[:-1]):
            deviation = abs(angle_edge)
            if deviation < 5:
                bar.set_facecolor('green')
            elif deviation < 15:
                bar.set_facecolor('yellow')
            else:
                bar.set_facecolor('red')
        
        # Show only semicircle
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        
        # Add angle labels around the arc
        min_angle, max_angle = angle_range
        # Add more tick marks around the semicircle
        theta_ticks = np.radians(np.array([min_angle, -60, -30, 0, 30, 60, max_angle]) + 90)
        theta_labels = [f'{min_angle}°', '-60°', '-30°', '0°', '30°', '60°', f'{max_angle}°']
        ax.set_xticks(theta_ticks)
        ax.set_xticklabels(theta_labels)
        
        # Show circular gridlines with frequency labels
        # Calculate appropriate radial ticks (5 ticks max)
        # Calculate the maximum frequency rounded up to the nearest 500
        max_freq_rounded = int(np.ceil(max_freq / 500) * 500)
        
        # Create ticks that are multiples of 500
        r_ticks = np.arange(0, max_freq_rounded + 500, 500)
        
        # If there are too many ticks, take every nth tick to reduce clutter
        if len(r_ticks) > 6:  # Limit to approximately 5-6 ticks for readability
            step = len(r_ticks) // 5
            r_ticks = r_ticks[::step]
            
        # Always include 0 and the max_freq_rounded
        if 0 not in r_ticks:
            r_ticks = np.append([0], r_ticks)
        if max_freq_rounded not in r_ticks:
            r_ticks = np.append(r_ticks, [max_freq_rounded])
        
        # Set radial ticks and circular gridlines
        ax.set_yticks(r_ticks)
        # Format large tick numbers with K suffix (e.g., 10000 -> 10K)
        # Format tick labels with K suffix, including .5K for values like 1500, 2500, etc.
        ax.set_yticklabels([f"{x/1000:.1f}K" if x >= 1000 else str(x) for x in r_ticks])        
        ax.grid(True, alpha=0.3)
        
        # Add 0 at center of semicircle
        ax.text(np.radians(90), 0, " ", ha='center', va='center', fontsize=2, 
               bbox=dict(facecolor='white', alpha=0.1, boxstyle='circle'))
        
        # Add maximum frequency label on the outermost radial tick
        ax.text(np.radians(135), max_freq, f"Max: {max_freq}", 
               ha='center', va='center', fontsize=8, 
               bbox=dict(facecolor='white', alpha=0.7))
        
        # Set height limits with a bit of padding
        ax.set_ylim(0, max_freq * 1.2)
        
        # Add mean angle line (red)
        mean_theta = np.radians(mean_angle + 90)
        ax.plot([mean_theta, mean_theta], [0, max_freq * 1.15], 'r-', linewidth=2.5, 
                label=f'Mean: {mean_angle:.2f}°')
        
        # Add median line (green)
        median_theta = np.radians(median_angle + 90)
        ax.plot([median_theta, median_theta], [0, max_freq * 1.1], 'g-', linewidth=2, 
               label=f'Median: {median_angle:.2f}°')
        
        # Add std deviation lines
        std_plus_theta = np.radians(mean_angle + std_angle + 90)
        std_minus_theta = np.radians(mean_angle - std_angle + 90)
        ax.plot([std_plus_theta, std_plus_theta], [0, max_freq * 1.05], 'r--', linewidth=1.5)
        ax.plot([std_minus_theta, std_minus_theta], [0, max_freq * 1.05], 'r--', linewidth=1.5,
               label=f'±1σ: {std_angle:.2f}°')
        
        # Add std dev shaded area
        theta = np.linspace(std_minus_theta, std_plus_theta, 50)
        r = np.ones_like(theta) * max_freq * 1.05
        ax.fill_between(theta, 0, r, color='red', alpha=0.15)
        
        # Add acceptable range (±15°)
        threshold_minus = np.radians(-15 + 90)
        threshold_plus = np.radians(15 + 90)
        
        # Add acceptable range shading
        theta = np.linspace(threshold_minus, threshold_plus, 50)
        r = np.ones_like(theta) * max_freq * 1.2
        ax.fill_between(theta, 0, r, color='green', alpha=0.1, label='Acceptable (±15°)')
        
        # Add title with actual statistics
        display_name = metric.replace('HeadPose', '')
        ax.set_title(f"{display_name} Angle Distribution\nMean: {mean_angle:.2f}° ± {std_angle:.2f}°", fontsize=12)
        
        # Add legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9)
    
    plt.suptitle('Head Pose Angle Distributions', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join(output_dir, 'pose_angle_histograms.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"✓ Saved enhanced histograms to {output_file}")
    
    # Part 2: Create 3D scatter plot with unit sphere
    if all(metric in df.columns for metric in ['HeadPoseYaw', 'HeadPosePitch', 'HeadPoseRoll']):
        print("Creating 3D scatter plot with unit sphere...")
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get the angles
        yaw = df['HeadPoseYaw'].values
        pitch = df['HeadPosePitch'].values
        roll = df['HeadPoseRoll'].values
        
        # Calculate statistics for each angle
        yaw_mean = np.mean(yaw)
        pitch_mean = np.mean(pitch)
        roll_mean = np.mean(roll)
        
        # Sample if dataset is very large (for better visualization)
        if len(df) > 2000:
            indices = np.random.choice(len(df), 2000, replace=False)
            yaw_sample = yaw[indices]
            pitch_sample = pitch[indices]
            roll_sample = roll[indices]
        else:
            yaw_sample = yaw
            pitch_sample = pitch
            roll_sample = roll
        
        # Create the scatter plot
        scatter = ax.scatter(yaw_sample, pitch_sample, roll_sample, 
                           alpha=0.6, s=15, c='blue')
        
        # Add labels and title
        ax.set_xlabel('Yaw Angle (degrees)')
        ax.set_ylabel('Pitch Angle (degrees)')
        ax.set_zlabel('Roll Angle (degrees)')
        ax.set_title('3D Distribution of Head Pose Angles', fontsize=14)
        
        # Create a unit sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Plot the unit sphere
        ax.plot_surface(x, y, z, color='gray', alpha=0.1)
        
        # Add principal axes
        max_range = max(30, abs(yaw_sample).max(), abs(pitch_sample).max(), abs(roll_sample).max())
        
        # X-axis (yaw)
        ax.plot([-max_range, max_range], [0, 0], [0, 0], 'r-', alpha=0.7, linewidth=1)
        ax.text(max_range*1.1, 0, 0, "Yaw", color='red', fontsize=10)
        
        # Y-axis (pitch)
        ax.plot([0, 0], [-max_range, max_range], [0, 0], 'g-', alpha=0.7, linewidth=1)
        ax.text(0, max_range*1.1, 0, "Pitch", color='green', fontsize=10)
        
        # Z-axis (roll)
        ax.plot([0, 0], [0, 0], [-max_range, max_range], 'b-', alpha=0.7, linewidth=1)
        ax.text(0, 0, max_range*1.1, "Roll", color='blue', fontsize=10)
        
        # Add the mean angle as a red point
        ax.scatter([yaw_mean], [pitch_mean], [roll_mean], 
                  color='red', s=100, marker='*')
        
        # Add text label for mean point
        ax.text(yaw_mean, pitch_mean, roll_mean, 
               f"Mean: ({yaw_mean:.1f}°, {pitch_mean:.1f}°, {roll_mean:.1f}°)", 
               color='red', fontsize=10)
        
        # Add markers on the sphere surface
        # Function to convert angles to 3D point on sphere surface
        def angles_to_sphere(yaw_deg, pitch_deg, roll_deg):
            # Convert to radians
            yaw_rad = np.radians(yaw_deg)
            pitch_rad = np.radians(pitch_deg)
            roll_rad = np.radians(roll_deg)
            
            # Normalize to create a unit vector
            mag = np.sqrt(yaw_deg**2 + pitch_deg**2 + roll_deg**2)
            if mag == 0:
                return 0, 0, 0
            
            # Return normalized vector
            return yaw_deg/mag, pitch_deg/mag, roll_deg/mag
        
        # Add markers for individual axes
        # Yaw axis markers
        for angle in [-30, -15, 15, 30]:
            x, y, z = angles_to_sphere(angle, 0, 0)
            ax.scatter([x], [y], [z], color='red', s=50)
            ax.text(x*1.1, y*1.1, z*1.1, f"{angle}°", color='red', fontsize=8)
        
        # Pitch axis markers
        for angle in [-30, -15, 15, 30]:
            x, y, z = angles_to_sphere(0, angle, 0)
            ax.scatter([x], [y], [z], color='green', s=50)
            ax.text(x*1.1, y*1.1, z*1.1, f"{angle}°", color='green', fontsize=8)
        
        # Roll axis markers
        for angle in [-30, -15, 15, 30]:
            x, y, z = angles_to_sphere(0, 0, angle)
            ax.scatter([x], [y], [z], color='blue', s=50)
            ax.text(x*1.1, y*1.1, z*1.1, f"{angle}°", color='blue', fontsize=8)
        
        # Add the ideal/origin point
        ax.scatter([0], [0], [0], color='gold', s=100, marker='o')
        ax.text(0, 0, 0, "Ideal", color='gold', fontsize=10)
        
        # Set equal aspect ratio for all axes
        ax.set_box_aspect([1, 1, 1])
        
        # Set limits
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Save the figure
        scatter_output = os.path.join(output_dir, 'pose_3d_scatter_with_sphere.png')
        plt.savefig(scatter_output, dpi=300)
        plt.close()
        print(f"✓ Saved 3D scatter plot to {scatter_output}")
    else:
        print("Cannot create 3D scatter - one or more pose metrics missing")


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from pathlib import Path
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

def visualize_face_boundary_boxes(csv_file, output_dir='face_margin_visualization'):
    """
    Create visualizations showing face boundary boxes with colored margin overlays
    to indicate the quality of left, right, top, and bottom margins.
    Uses raw values instead of scalar versions.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    df = pd.read_csv(csv_file, sep=';')
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Define the raw column names we want to use
    margin_columns = {
        'LeftwardCropOfFaceImage': 'LeftwardCropOfTheFaceImage',  # Note: fixed name to match data
        'RightwardCropOfFaceImage': 'RightwardCropOfTheFaceImage',  # Note: fixed name to match data
        'MarginAboveOfFaceImage': 'MarginAboveOfTheFaceImage',
        'MarginBelowOfFaceImage': 'MarginBelowOfTheFaceImage'
    }
    
    # Check if all required columns exist
    for col_name in margin_columns.values():
        if col_name not in df.columns:
            print(f"Error: Required column {col_name} not found in dataset")
            return
    
    # Check if Filename column exists
    if 'Filename' not in df.columns:
        print("Error: Filename column not found in dataset")
        return
    
    # Define ideal values and thresholds for each margin metric
    margin_ideals = {
        'LeftwardCropOfFaceImage': {'ideal': 0, 'lower_is_better': True, 'threshold': 20},
        'RightwardCropOfFaceImage': {'ideal': 0, 'lower_is_better': True, 'threshold': 20},
        'MarginAboveOfFaceImage': {'ideal': 10, 'lower_is_better': False, 'threshold': 5},
        'MarginBelowOfFaceImage': {'ideal': 15, 'lower_is_better': False, 'threshold': 10}
    }
    
    # Create a composite score to sort images from worst to best framing
    df['CompositeMarginScore'] = 0
    for base_col, col_name in margin_columns.items():
        if margin_ideals[base_col]['lower_is_better']:
            # For metrics where lower is better, higher values give worse scores
            normalized_score = df[col_name] / df[col_name].max()
        else:
            # For metrics where a specific value is ideal, distance from ideal gives worse scores
            ideal_val = margin_ideals[base_col]['ideal']
            max_distance = max(abs(df[col_name] - ideal_val).max(), 1)  # Avoid division by zero
            normalized_score = abs(df[col_name] - ideal_val) / max_distance
        
        df['CompositeMarginScore'] += normalized_score
    
    # Sort by composite score (higher score = worse framing)
    df_sorted = df.sort_values('CompositeMarginScore', ascending=False)
    
    # Select diverse samples: 6 worst, 3 medium, 3 best
    n_samples = 12
    worst_samples = df_sorted.head(6)
    
    middle_idx = len(df_sorted) // 2
    middle_samples = df_sorted.iloc[middle_idx-1:middle_idx+2]
    
    best_samples = df_sorted.tail(3)
    
    samples = pd.concat([worst_samples, middle_samples, best_samples])
    
    # Verify we can access the image files
    can_access_images = False
    sample_path = df['Filename'].iloc[0]
    try:
        img = Image.open(sample_path)
        can_access_images = True
        img.close()
    except Exception as e:
        print(f"Warning: Could not open image at {sample_path}: {e}")
        print("Cannot create face boundary visualization without image access")
        return
    
    # Create the visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    # Function to map a metric value to a color
    def get_margin_color(value, ideal, lower_is_better, threshold):
        if lower_is_better:
            # For lower-is-better, red for high, green for low
            if value <= ideal + threshold * 0.5:
                return 'green', 0.3  # Good - low transparency
            elif value <= ideal + threshold:
                return 'yellow', 0.4  # Acceptable - medium transparency
            else:
                return 'red', 0.5  # Poor - high transparency
        else:
            # For metrics with an ideal value, green near ideal, red far from ideal
            if abs(value - ideal) <= threshold * 0.5:
                return 'green', 0.3  # Good - low transparency
            elif abs(value - ideal) <= threshold:
                return 'yellow', 0.4  # Acceptable - medium transparency
            else:
                return 'red', 0.5  # Poor - high transparency
    
    for i, (_, row) in enumerate(samples.iterrows()):
        # Skip if we've reached the end of our axes
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Load and display the image
        img_path = row['Filename']
        try:
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            # Display the image
            ax.imshow(np.array(img))
            
            # Extract margin values
            left_margin = row[margin_columns['LeftwardCropOfFaceImage']]
            right_margin = row[margin_columns['RightwardCropOfFaceImage']]
            top_margin = row[margin_columns['MarginAboveOfFaceImage']]
            bottom_margin = row[margin_columns['MarginBelowOfFaceImage']]
            
            # Calculate face bounding box (approximated from margins)
            # These calculations are approximate and would need adjusting based on how your margins are defined
            left_px = int(img_width * left_margin / 100) if left_margin <= 100 else int(left_margin)
            right_px = img_width - int(img_width * right_margin / 100) if right_margin <= 100 else (img_width - int(right_margin))
            top_px = int(img_height * top_margin / 100) if top_margin <= 100 else int(top_margin)
            bottom_px = img_height - int(img_height * bottom_margin / 100) if bottom_margin <= 100 else (img_height - int(bottom_margin))
            
            # Face width and height
            face_width = right_px - left_px
            face_height = bottom_px - top_px
            
            # Draw the face box
            face_rect = patches.Rectangle((left_px, top_px), face_width, face_height,
                                         linewidth=2, edgecolor='yellow', facecolor='none')
            ax.add_patch(face_rect)
            
            # Create colored overlay for margins
            # Left margin
            left_color, left_alpha = get_margin_color(
                left_margin,
                margin_ideals['LeftwardCropOfFaceImage']['ideal'], 
                margin_ideals['LeftwardCropOfFaceImage']['lower_is_better'],
                margin_ideals['LeftwardCropOfFaceImage']['threshold']
            )
            left_rect = patches.Rectangle((0, 0), left_px, img_height,
                                         linewidth=0, facecolor=left_color, alpha=left_alpha)
            ax.add_patch(left_rect)
            
            # Right margin
            right_color, right_alpha = get_margin_color(
                right_margin,
                margin_ideals['RightwardCropOfFaceImage']['ideal'], 
                margin_ideals['RightwardCropOfFaceImage']['lower_is_better'],
                margin_ideals['RightwardCropOfFaceImage']['threshold']
            )
            right_rect = patches.Rectangle((right_px, 0), img_width-right_px, img_height,
                                           linewidth=0, facecolor=right_color, alpha=right_alpha)
            ax.add_patch(right_rect)
            
            # Top margin
            top_color, top_alpha = get_margin_color(
                top_margin,
                margin_ideals['MarginAboveOfFaceImage']['ideal'], 
                margin_ideals['MarginAboveOfFaceImage']['lower_is_better'],
                margin_ideals['MarginAboveOfFaceImage']['threshold']
            )
            top_rect = patches.Rectangle((0, 0), img_width, top_px,
                                         linewidth=0, facecolor=top_color, alpha=top_alpha)
            ax.add_patch(top_rect)
            
            # Bottom margin
            bottom_color, bottom_alpha = get_margin_color(
                bottom_margin,
                margin_ideals['MarginBelowOfFaceImage']['ideal'], 
                margin_ideals['MarginBelowOfFaceImage']['lower_is_better'],
                margin_ideals['MarginBelowOfFaceImage']['threshold']
            )
            bottom_rect = patches.Rectangle((0, bottom_px), img_width, img_height-bottom_px,
                                           linewidth=0, facecolor=bottom_color, alpha=bottom_alpha)
            ax.add_patch(bottom_rect)
            
            # Add margin value annotations
            ax.text(left_px/2, img_height/2, f"L: {left_margin:.1f}", color='white', 
                    fontsize=8, ha='center', va='center', fontweight='bold')
            ax.text(right_px + (img_width-right_px)/2, img_height/2, f"R: {right_margin:.1f}", 
                    color='white', fontsize=8, ha='center', va='center', fontweight='bold')
            ax.text(img_width/2, top_px/2, f"T: {top_margin:.1f}", color='white', 
                    fontsize=8, ha='center', va='center', fontweight='bold')
            ax.text(img_width/2, bottom_px + (img_height-bottom_px)/2, f"B: {bottom_margin:.1f}", 
                    color='white', fontsize=8, ha='center', va='center', fontweight='bold')
            
            # Add a title with the composite score
            ax.set_title(f"Margin Score: {row['CompositeMarginScore']:.2f}", fontsize=10)
            
            # Remove axis ticks
            ax.set_xticks([])
            ax.set_yticks([])
            
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            ax.text(0.5, 0.5, f"Image Error: {Path(img_path).name}",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Add a title for the entire figure
    plt.suptitle("Face Margin Visualization: Red = Poor, Yellow = Acceptable, Green = Good", fontsize=16)
    
    # Add a legend
    legend_elements = [
        patches.Patch(facecolor='red', alpha=0.5, label='Poor Margin'),
        patches.Patch(facecolor='yellow', alpha=0.4, label='Acceptable Margin'),
        patches.Patch(facecolor='green', alpha=0.3, label='Good Margin'),
        patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='yellow', facecolor='none', label='Face Bounding Box')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    output_file = os.path.join(output_dir, 'face_margin_visualization.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"✓ Face margin visualization saved to {output_file}")


def visualize_face_position_heatmap(csv_file, output_dir='face_position_visualization'):
    """
    Create a heatmap showing where faces are positioned within images,
    based on the left, right, top, and bottom margin metrics.
    Uses raw values instead of scalar versions.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    df = pd.read_csv(csv_file, sep=';')
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Define the raw column names we want to use
    margin_columns = {
        'LeftwardCropOfFaceImage': 'LeftwardCropOfTheFaceImage',  # Note: fixed name to match data
        'RightwardCropOfFaceImage': 'RightwardCropOfTheFaceImage',  # Note: fixed name to match data
        'MarginAboveOfFaceImage': 'MarginAboveOfTheFaceImage',
        'MarginBelowOfFaceImage': 'MarginBelowOfTheFaceImage'
    }
    
    # Check if all required columns exist
    missing_columns = False
    for col_name in margin_columns.values():
        if col_name not in df.columns:
            print(f"Error: Required column {col_name} not found in dataset")
            missing_columns = True
    
    if 'Filename' not in df.columns or missing_columns:
        print("Error: Missing required columns for face position heatmap")
        return
    
    # Calculate face center position for each image using a different approach
    # For OFIQ dataset, the margins are direct percentages, so we need a different calculation
    
    # Print min and max values of each margin to understand the range
    for col in margin_columns.values():
        print(f"{col} range: {df[col].min()} to {df[col].max()}")
    
    # Calculate face width and height as percentage of image
    df['FaceWidth'] = 100 - df[margin_columns['LeftwardCropOfFaceImage']] - df[margin_columns['RightwardCropOfFaceImage']]
    df['FaceHeight'] = 100 - df[margin_columns['MarginAboveOfFaceImage']] - df[margin_columns['MarginBelowOfFaceImage']]
    
    # Calculate face center position (as percentage of image)
    df['FaceCenterX'] = df[margin_columns['LeftwardCropOfFaceImage']] + (df['FaceWidth'] / 2)
    df['FaceCenterY'] = df[margin_columns['MarginAboveOfFaceImage']] + (df['FaceHeight'] / 2)
    
    # Normalize to 0-1 range (already in percentage, just divide by 100)
    df['NormX'] = df['FaceCenterX'] / 100
    df['NormY'] = df['FaceCenterY'] / 100
    
    # Print statistics to verify our calculations
    print(f"FaceCenterX range: {df['FaceCenterX'].min()} to {df['FaceCenterX'].max()}")
    print(f"FaceCenterY range: {df['FaceCenterY'].min()} to {df['FaceCenterY'].max()}")
    print(f"NormX range: {df['NormX'].min()} to {df['NormX'].max()}")
    print(f"NormY range: {df['NormY'].min()} to {df['NormY'].max()}")
    
    # Add extra diagnostics
    print(f"Total valid face positions: {len(df)}")
    print(f"NaN values in NormX: {df['NormX'].isna().sum()}")
    print(f"NaN values in NormY: {df['NormY'].isna().sum()}")
    
    # Get quality scores - use raw UnifiedQualityScore
    quality_col = 'UnifiedQualityScore'
    if quality_col not in df.columns:
        print(f"Warning: {quality_col} not found. Quality-based analysis will be skipped.")
        quality_col = None
    
    # Filter out any invalid or extreme positions
    # Print before filtering
    print(f"Before filtering - Total records: {len(df)}")
    
    # Filter with more lenient bounds (allowing some margin outside 0-1)
    df_filtered = df[(df['NormX'] >= -0.5) & (df['NormX'] <= 1.5) & 
                    (df['NormY'] >= -0.5) & (df['NormY'] <= 1.5)]
    
    # Print after filtering
    print(f"After filtering - Total records: {len(df_filtered)}")
    
    # Create the basic density heatmap
    plt.figure(figsize=(12, 10))
    
    # Create a 2D histogram of face positions with more bins for better resolution
    h, xedges, yedges = np.histogram2d(
        df_filtered['NormX'], df_filtered['NormY'], 
        bins=[30, 30], 
        range=[[0, 1], [0, 1]]
    )
    
    # Apply Gaussian smoothing to the histogram for better visualization
    from scipy.ndimage import gaussian_filter
    h_smooth = gaussian_filter(h, sigma=1.0)
    
    # Transpose and normalize
    h_smooth = h_smooth.T
    h_norm = h_smooth / h_smooth.max() if h_smooth.max() > 0 else h_smooth
    
    # Create the heatmap with better color mapping and interpolation
    plt.imshow(
        h_norm, 
        extent=[0, 1, 0, 1], 
        origin='lower', 
        cmap='inferno',  # Using inferno for better visibility
        interpolation='gaussian',  # Smooth interpolation
        aspect='auto'
    )
    
    # Add a colorbar with better formatting
    cbar = plt.colorbar(label='Normalized Density')
    cbar.ax.tick_params(labelsize=10)
    
    # Add grid lines for the rule of thirds
    for i in [1/3, 2/3]:
        plt.axvline(i, color='white', linestyle='--', alpha=0.5, linewidth=1.5)
        plt.axhline(i, color='white', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Mark the center of the image more prominently
    plt.plot(0.5, 0.5, 'w+', markersize=12, markeredgewidth=2)
    
    # Add ideal face position markers at rule of thirds intersections
    rule_thirds = [1/3, 2/3]
    for x in rule_thirds:
        for y in rule_thirds:
            plt.plot(x, y, 'wo', markersize=8, alpha=0.7)
    
    # Add axis labels
    plt.xlabel('Horizontal Position (normalized)')
    plt.ylabel('Vertical Position (normalized)')
    
    # Add a title
    plt.title('Face Position Density: All Images', fontsize=14)
    
    # Annotate rule of thirds
    plt.annotate('Rule of Thirds Points', xy=(2/3, 2/3), xytext=(0.75, 0.75),
                arrowprops=dict(facecolor='white', shrink=0.05, width=2, alpha=0.7),
                color='white', fontsize=10)
    
    # Add center annotation
    plt.annotate('Center', xy=(0.5, 0.5), xytext=(0.6, 0.4),
                arrowprops=dict(facecolor='white', shrink=0.05, width=2, alpha=0.7),
                color='white', fontsize=10)
    
    # Save the basic heatmap
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'face_position_heatmap.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"✓ Basic face position heatmap saved to {output_file}")
    
    # If quality scores are available, create a quality-weighted heatmap
    if quality_col:
        plt.figure(figsize=(14, 12))
        
        # Create the subplot grid
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
        
        # 1. Basic density heatmap (top left)
        ax1 = plt.subplot(gs[0, 0])
        ax1.imshow(h_norm, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
        
        # Add grid lines for rule of thirds
        for i in [1/3, 2/3]:
            ax1.axvline(i, color='white', linestyle='--', alpha=0.5)
            ax1.axhline(i, color='white', linestyle='--', alpha=0.5)
        
        ax1.set_title('Face Position Density (All Images)')
        ax1.set_xlabel('Horizontal Position')
        ax1.set_ylabel('Vertical Position')
        
        # 2. Quality-weighted heatmap (top right)
        ax2 = plt.subplot(gs[0, 1])
        
        # Split into high and low quality
        median_quality = df[quality_col].median()
        high_quality = df[df[quality_col] >= median_quality]
        
        # Create heatmap for high quality images
        h_high, _, _ = np.histogram2d(
            high_quality['NormX'], high_quality['NormY'], 
            bins=[20, 20], 
            range=[[0, 1], [0, 1]]
        )
        h_high = h_high.T
        h_high_norm = h_high / h_high.max() if h_high.max() > 0 else h_high
        
        # Display high quality heatmap
        ax2.imshow(h_high_norm, extent=[0, 1, 0, 1], origin='lower', cmap='plasma')
        
        # Add grid lines for rule of thirds
        for i in [1/3, 2/3]:
            ax2.axvline(i, color='white', linestyle='--', alpha=0.5)
            ax2.axhline(i, color='white', linestyle='--', alpha=0.5)
        
        ax2.set_title(f'Face Positions for High Quality Images (>{median_quality:.1f})')
        ax2.set_xlabel('Horizontal Position')
        ax2.set_ylabel('Vertical Position')
        
        # 3. Scatter plot with quality-based coloring (bottom left)
        ax3 = plt.subplot(gs[1, 0])
        
        # Sample data if very large (for better visualization)
        plot_df = df
        if len(df) > 2000:
            plot_df = df.sample(2000, random_state=42)
        
        # Create scatter plot
        scatter = ax3.scatter(
            plot_df['NormX'], plot_df['NormY'],
            c=plot_df[quality_col], cmap='RdYlGn',
            alpha=0.6, s=15
        )
        
        # Add grid lines for rule of thirds
        for i in [1/3, 2/3]:
            ax3.axvline(i, color='black', linestyle='--', alpha=0.3)
            ax3.axhline(i, color='black', linestyle='--', alpha=0.3)
        
        # Add a colorbar
        plt.colorbar(scatter, ax=ax3, label=quality_col)
        
        ax3.set_title('Face Positions Colored by Quality Score')
        ax3.set_xlabel('Horizontal Position')
        ax3.set_ylabel('Vertical Position')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        
        # 4. Difference heatmap (high quality - all) (bottom right)
        ax4 = plt.subplot(gs[1, 1])
        
        # Low quality images
        low_quality = df[df[quality_col] < median_quality]
        
        # Create heatmap for low quality images
        h_low, _, _ = np.histogram2d(
            low_quality['NormX'], low_quality['NormY'], 
            bins=[20, 20], 
            range=[[0, 1], [0, 1]]
        )
        h_low = h_low.T
        h_low_norm = h_low / h_low.max() if h_low.max() > 0 else h_low
        
        # Display the difference
        diff = h_high_norm - h_low_norm
        
        # Create a diverging colormap for the difference
        divnorm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        im = ax4.imshow(diff, extent=[0, 1, 0, 1], origin='lower', cmap='RdBu_r', norm=divnorm)
        
        # Add grid lines for rule of thirds
        for i in [1/3, 2/3]:
            ax4.axvline(i, color='black', linestyle='--', alpha=0.3)
            ax4.axhline(i, color='black', linestyle='--', alpha=0.3)
        
        # Add a colorbar
        plt.colorbar(im, ax=ax4, label='High Quality - Low Quality Difference')
        
        ax4.set_title('Difference: High Quality vs Low Quality Positions')
        ax4.set_xlabel('Horizontal Position')
        ax4.set_ylabel('Vertical Position')
        
        # Add a main title
        plt.suptitle('Face Position Analysis and Quality Correlation', fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save the comprehensive analysis
        output_file = os.path.join(output_dir, 'face_position_quality_analysis.png')
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"✓ Quality-based face position analysis saved to {output_file}")


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def generate_ofiq_summary(csv_file, output_dir=None, create_plots=True):
    """
    Generate a text-based executive summary for an OFIQ dataset.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing OFIQ data
    output_dir : str, optional
        Directory to save plots, if None, plots are just displayed
    create_plots : bool
        Whether to create visualization plots
    
    Returns:
    --------
    str
        Text summary of the dataset
    """
    # Load the data
    try:
        df = pd.read_csv(csv_file, sep=';')
    except Exception as e:
        print(f"Error loading CSV with semicolon delimiter, trying comma: {e}")
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return "ERROR: Failed to load dataset."
    
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Create output directory if specified
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Start building the summary text
    summary = []
    summary.append("=" * 80)
    summary.append(f"OFIQ DATASET EXECUTIVE SUMMARY: {os.path.basename(csv_file)}")
    summary.append("=" * 80)
    summary.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    summary.append(f"Total Images: {df.shape[0]}")
    
    # Print columns for debugging
    summary.append("\nAvailable Columns:")
    column_groups = {
        "Raw Quality Metrics": [col for col in df.columns if not col.endswith('.scalar') and 
                               any(q in col for q in ['Quality', 'Sharpness', 'Background', 'Illumination', 'Colour'])],
        "Raw Face/Head Metrics": [col for col in df.columns if not col.endswith('.scalar') and 
                                 any(q in col for q in ['Face', 'Head', 'Eyes', 'Mouth', 'Pose'])],
        "Scalar Metrics": [col for col in df.columns if col.endswith('.scalar')]
    }
    
    for group, cols in column_groups.items():
        if cols:
            summary.append(f"  {group}: {', '.join(cols)}")
    
    summary.append("")
    
    # 1. Analyze quality scores
    summary.append("-" * 80)
    summary.append("QUALITY SCORE ANALYSIS")
    summary.append("-" * 80)
    
    # Look for UnifiedQualityScore column (prefer scalar)
    if 'UnifiedQualityScore.scalar' in df.columns:
        quality_col = 'UnifiedQualityScore.scalar'
    elif 'UnifiedQualityScore' in df.columns:
        quality_col = 'UnifiedQualityScore'
    else:
        quality_col = None
        summary.append("WARNING: UnifiedQualityScore not found in dataset")
    
    if quality_col:
        # Basic statistics
        q_mean = df[quality_col].mean()
        q_median = df[quality_col].median()
        q_std = df[quality_col].std()
        q_min = df[quality_col].min()
        q_max = df[quality_col].max()
        
        summary.append(f"Mean Quality Score: {q_mean:.2f}")
        summary.append(f"Median Quality Score: {q_median:.2f}")
        summary.append(f"Quality Score Range: {q_min:.2f} - {q_max:.2f}")
        summary.append(f"Quality Score Std Dev: {q_std:.2f}")
        summary.append("")
        
        # Quality distribution
        bins = [0, 20, 40, 60, 80, 100]
        labels = ['Very Poor (0-20)', 'Poor (20-40)', 'Fair (40-60)', 'Good (60-80)', 'Excellent (80-100)']
        df['QualityCategory'] = pd.cut(df[quality_col], bins=bins, labels=labels, include_lowest=True)
        quality_distribution = df['QualityCategory'].value_counts().sort_index()
        
        summary.append("Quality Distribution:")
        for category, count in quality_distribution.items():
            percentage = (count / len(df)) * 100
            summary.append(f"  {category}: {count} images ({percentage:.1f}%)")
        
        # Quality assessment
        poor_quality_pct = ((df[quality_col] < 40).sum() / len(df)) * 100
        good_quality_pct = ((df[quality_col] >= 60).sum() / len(df)) * 100
        
        summary.append("")
        if poor_quality_pct > 30:
            summary.append(f"CRITICAL: {poor_quality_pct:.1f}% of images have poor quality (< 40).")
        elif poor_quality_pct > 15:
            summary.append(f"WARNING: {poor_quality_pct:.1f}% of images have poor quality (< 40).")
        else:
            summary.append(f"GOOD: Only {poor_quality_pct:.1f}% of images have poor quality (< 40).")
        
        if good_quality_pct > 80:
            summary.append(f"EXCELLENT: {good_quality_pct:.1f}% of images have good quality (>= 60).")
        elif good_quality_pct > 60:
            summary.append(f"GOOD: {good_quality_pct:.1f}% of images have good quality (>= 60).")
        else:
            summary.append(f"WARNING: Only {good_quality_pct:.1f}% of images have good quality (>= 60).")
        
        # Create quality histogram if requested
        if create_plots:
            plt.figure(figsize=(10, 6))
            ax = sns.histplot(df[quality_col], bins=20, kde=True)
            ax.axvline(q_mean, color='red', linestyle='--', label=f'Mean: {q_mean:.2f}')
            ax.axvline(q_median, color='green', linestyle='--', label=f'Median: {q_median:.2f}')
            
            plt.title('Distribution of Unified Quality Scores', fontsize=14)
            plt.xlabel('Quality Score (0-100)', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'quality_distribution.png'), dpi=300)
                summary.append(f"[Quality histogram saved to {os.path.join(output_dir, 'quality_distribution.png')}]")
            else:
                # plt.show()
                pass
    
    # 2. Analyze category metrics
    summary.append("")
    summary.append("-" * 80)
    summary.append("METRIC CATEGORY ANALYSIS")
    summary.append("-" * 80)
    
    # Define metric categories with specific handling instructions
    categories = {
        'Image Quality': [
            # Quality metrics - prefer scalar versions
            {'name': 'BackgroundUniformity', 'prefer_scalar': True},
            {'name': 'IlluminationUniformity', 'prefer_scalar': True},
            {'name': 'Sharpness', 'prefer_scalar': True},
            {'name': 'CompressionArtifacts', 'prefer_scalar': True},
            {'name': 'NaturalColour', 'prefer_scalar': True},
            {'name': 'LuminanceMean', 'prefer_scalar': True},
            {'name': 'LuminanceVariance', 'prefer_scalar': True},
            {'name': 'UnderExposurePrevention', 'prefer_scalar': True},
            {'name': 'OverExposurePrevention', 'prefer_scalar': True},
            {'name': 'DynamicRange', 'prefer_scalar': True}
        ],
        'Face Characteristics': [
            # Face characteristics - prefer scalar versions
            {'name': 'EyesOpen', 'prefer_scalar': True},
            {'name': 'MouthClosed', 'prefer_scalar': True},
            {'name': 'EyesVisible', 'prefer_scalar': True},
            {'name': 'ExpressionNeutrality', 'prefer_scalar': True},
            {'name': 'SingleFacePresent', 'prefer_scalar': True},
            {'name': 'MouthOcclusionPrevention', 'prefer_scalar': True},
            {'name': 'FaceOcclusionPrevention', 'prefer_scalar': True},
            {'name': 'InterEyeDistance', 'prefer_scalar': True},
            {'name': 'HeadSize', 'prefer_scalar': True},
            {'name': 'NoHeadCoverings', 'prefer_scalar': True}
        ],
        'Head Pose': [
            # Pose metrics - prefer raw values
            {'name': 'HeadPoseYaw', 'prefer_scalar': False},
            {'name': 'HeadPosePitch', 'prefer_scalar': False},
            {'name': 'HeadPoseRoll', 'prefer_scalar': False}
        ],
        'Face Position': [
            # Margin metrics - prefer raw values
            {'name': 'LeftwardCropOfTheFaceImage', 'prefer_scalar': False},
            {'name': 'RightwardCropOfTheFaceImage', 'prefer_scalar': False},
            {'name': 'MarginAboveOfTheFaceImage', 'prefer_scalar': False},
            {'name': 'MarginBelowOfTheFaceImage', 'prefer_scalar': False}
        ]
    }
    
    # Fix column name variations
    if 'LeftwardCropOfTheFaceImage' not in df.columns and 'LeftwardCropOfFaceImage' in df.columns:
        categories['Face Position'][0] = 'LeftwardCropOfFaceImage'
    
    if 'RightwardCropOfTheFaceImage' not in df.columns and 'RightwardCropOfFaceImage' in df.columns:
        categories['Face Position'][1] = 'RightwardCropOfFaceImage'
    
    # Define thresholds for good quality
    thresholds = {
        'BackgroundUniformity': {'min': 50, 'critical': 30},
        'IlluminationUniformity': {'min': 50, 'critical': 30},
        'Sharpness': {'min': 60, 'critical': 40},
        'CompressionArtifacts': {'min': 60, 'critical': 40},
        'NaturalColour': {'min': 60, 'critical': 40},
        'EyesOpen': {'min': 80, 'critical': 60},
        'MouthClosed': {'min': 80, 'critical': 60},
        'EyesVisible': {'min': 80, 'critical': 60},
        'ExpressionNeutrality': {'min': 70, 'critical': 50},
        'HeadPoseYaw': {'max': 15, 'critical': 30},
        'HeadPosePitch': {'max': 15, 'critical': 30},
        'HeadPoseRoll': {'max': 15, 'critical': 30}
    }
    
    # Analyze each category
    problem_metrics = []
    
    for category, metrics in categories.items():
        summary.append(f"\n{category}:")
        
        for metric_info in metrics:
            metric = metric_info['name']
            prefer_scalar = metric_info['prefer_scalar']
            
            # Try to find the metric in appropriate form (raw or scalar based on preference)
            metric_col = None
            
            if prefer_scalar:
                # For quality metrics - prefer scalar version
                if metric + '.scalar' in df.columns:
                    metric_col = metric + '.scalar'
                elif metric in df.columns:
                    metric_col = metric
            else:
                # For pose/margin metrics - prefer raw version
                if metric in df.columns:
                    metric_col = metric
                elif metric + '.scalar' in df.columns:
                    metric_col = metric + '.scalar'
            
            # Fix column name variations
            if metric == 'LeftwardCropOfTheFaceImage' and metric_col is None:
                if 'LeftwardCropOfFaceImage' in df.columns:
                    metric_col = 'LeftwardCropOfFaceImage'
                    metric = 'LeftwardCropOfFaceImage'
                elif 'LeftwardCropOfFaceImage.scalar' in df.columns and not prefer_scalar:
                    metric_col = 'LeftwardCropOfFaceImage.scalar'
                    metric = 'LeftwardCropOfFaceImage'
            
            if metric == 'RightwardCropOfTheFaceImage' and metric_col is None:
                if 'RightwardCropOfFaceImage' in df.columns:
                    metric_col = 'RightwardCropOfFaceImage'
                    metric = 'RightwardCropOfFaceImage'
                elif 'RightwardCropOfFaceImage.scalar' in df.columns and not prefer_scalar:
                    metric_col = 'RightwardCropOfFaceImage.scalar'
                    metric = 'RightwardCropOfFaceImage'
                    
            if metric_col and pd.api.types.is_numeric_dtype(df[metric_col]):
                # Calculate statistics
                mean_val = df[metric_col].mean()
                median_val = df[metric_col].median()
                min_val = df[metric_col].min()
                max_val = df[metric_col].max()
                
                summary.append(f"  {metric}: mean={mean_val:.2f}, median={median_val:.2f}, range={min_val:.2f}-{max_val:.2f}")
                
                # Check for problems
                if metric in thresholds:
                    if 'min' in thresholds[metric] and mean_val < thresholds[metric]['min']:
                        severity = 'CRITICAL' if mean_val < thresholds[metric].get('critical', 0) else 'WARNING'
                        problem = {
                            'metric': metric,
                            'value': mean_val,
                            'threshold': thresholds[metric]['min'],
                            'direction': 'higher',
                            'severity': severity
                        }
                        problem_metrics.append(problem)
                    
                    elif 'max' in thresholds[metric] and mean_val > thresholds[metric]['max']:
                        severity = 'CRITICAL' if mean_val > thresholds[metric].get('critical', 100) else 'WARNING'
                        problem = {
                            'metric': metric,
                            'value': mean_val,
                            'threshold': thresholds[metric]['max'],
                            'direction': 'lower',
                            'severity': severity
                        }
                        problem_metrics.append(problem)
            else:
                summary.append(f"  {metric}: Not found in dataset")
    
    # 3. Face position analysis
    # Get face position metrics with correct preference (raw values)
    position_cols = {}
    for metric_info in categories['Face Position']:
        metric = metric_info['name']
        
        # Prefer raw version for position metrics
        if metric in df.columns:
            position_cols[metric] = metric
        elif metric == 'LeftwardCropOfTheFaceImage' and 'LeftwardCropOfFaceImage' in df.columns:
            position_cols[metric] = 'LeftwardCropOfFaceImage'
        elif metric == 'RightwardCropOfTheFaceImage' and 'RightwardCropOfFaceImage' in df.columns:
            position_cols[metric] = 'RightwardCropOfFaceImage'
        elif metric + '.scalar' in df.columns:
            position_cols[metric] = metric + '.scalar'
    
    if len(position_cols) == 4:  # We have all the margin metrics
        summary.append("")
        summary.append("-" * 80)
        summary.append("FACE POSITION ANALYSIS")
        summary.append("-" * 80)
        
        # Get column names
        left_col = position_cols.get('LeftwardCropOfTheFaceImage') or position_cols.get('LeftwardCropOfFaceImage')
        right_col = position_cols.get('RightwardCropOfTheFaceImage') or position_cols.get('RightwardCropOfFaceImage')
        top_col = position_cols.get('MarginAboveOfTheFaceImage')
        bottom_col = position_cols.get('MarginBelowOfTheFaceImage')
        
        # Calculate face dimensions
        df['FaceWidth'] = 100 - df[left_col] - df[right_col]
        df['FaceHeight'] = 100 - df[top_col] - df[bottom_col]
        
        # Calculate face center position
        df['FaceCenterX'] = df[left_col] + (df['FaceWidth'] / 2)
        df['FaceCenterY'] = df[top_col] + (df['FaceHeight'] / 2)
        
        # Calculate position statistics
        center_x_mean = df['FaceCenterX'].mean()
        center_y_mean = df['FaceCenterY'].mean()
        face_width_mean = df['FaceWidth'].mean()
        face_height_mean = df['FaceHeight'].mean()
        
        summary.append(f"Average Face Position: ({center_x_mean:.2f}%, {center_y_mean:.2f}%)")
        summary.append(f"Average Face Width: {face_width_mean:.2f}% of image width")
        summary.append(f"Average Face Height: {face_height_mean:.2f}% of image height")
        
        # Check for positioning issues
        # Ideal center position is at the middle or rule of thirds points
        acceptable_dist = 10  # percentage points
        
        # Calculate distance to ideal centers
        center_dist = min(
            # Distance to center
            np.sqrt((center_x_mean - 50)**2 + (center_y_mean - 50)**2),
            # Distance to top-left rule of thirds
            np.sqrt((center_x_mean - 33.3)**2 + (center_y_mean - 33.3)**2),
            # Distance to top-right rule of thirds
            np.sqrt((center_x_mean - 66.7)**2 + (center_y_mean - 33.3)**2),
            # Distance to bottom-left rule of thirds
            np.sqrt((center_x_mean - 33.3)**2 + (center_y_mean - 66.7)**2),
            # Distance to bottom-right rule of thirds
            np.sqrt((center_x_mean - 66.7)**2 + (center_y_mean - 66.7)**2)
        )
        
        if center_dist > acceptable_dist:
            summary.append("")
            summary.append(f"WARNING: Face positioning deviates from ideal composition.")
            summary.append(f"  Average face center is at ({center_x_mean:.1f}%, {center_y_mean:.1f}%)")
            summary.append(f"  Should be closer to center or rule-of-thirds points.")
        
        # Check face size
        ideal_width_min = 60  # percent
        if face_width_mean < ideal_width_min:
            summary.append("")
            summary.append(f"WARNING: Faces are too small in the frame.")
            summary.append(f"  Average face width is {face_width_mean:.1f}% of image width")
            summary.append(f"  Should be at least {ideal_width_min}% for optimal recognition.")
        
        # Create face position heatmap if requested
        if create_plots:
            # Normalize to 0-1 range
            df['NormX'] = df['FaceCenterX'] / 100
            df['NormY'] = df['FaceCenterY'] / 100
            
            # Filter out outliers for better visualization
            df_filtered = df[(df['NormX'] >= 0) & (df['NormX'] <= 1) & 
                            (df['NormY'] >= 0) & (df['NormY'] <= 1)]
            
            if len(df_filtered) > 10:  # Need sufficient data points
                plt.figure(figsize=(8, 8))
                
                # Create a 2D histogram
                h, xedges, yedges = np.histogram2d(
                    df_filtered['NormX'], df_filtered['NormY'], 
                    bins=[30, 30], 
                    range=[[0, 1], [0, 1]]
                )
                
                # Apply smoothing if scipy is available
                try:
                    from scipy.ndimage import gaussian_filter
                    h_smooth = gaussian_filter(h.T, sigma=1.0)
                except ImportError:
                    h_smooth = h.T
                
                h_norm = h_smooth / h_smooth.max() if h_smooth.max() > 0 else h_smooth
                
                # Create the heatmap
                plt.imshow(
                    h_norm, 
                    extent=[0, 1, 0, 1], 
                    origin='lower', 
                    cmap='inferno'
                )
                
                plt.colorbar(label='Normalized Density')
                
                # Add rule of thirds grid
                for i in [1/3, 2/3]:
                    plt.axvline(i, color='white', linestyle='--', alpha=0.5)
                    plt.axhline(i, color='white', linestyle='--', alpha=0.5)
                
                # Mark center of image
                plt.plot(0.5, 0.5, 'w+', markersize=12, markeredgewidth=2)
                
                # Mark mean face position
                plt.plot(df_filtered['NormX'].mean(), df_filtered['NormY'].mean(), 
                        'ro', markersize=10, label='Mean Face Position')
                
                plt.title('Face Position Heatmap', fontsize=14)
                plt.xlabel('Horizontal Position (normalized)', fontsize=12)
                plt.ylabel('Vertical Position (normalized)', fontsize=12)
                plt.legend()
                
                if output_dir:
                    plt.savefig(os.path.join(output_dir, 'face_position_heatmap.png'), dpi=300)
                    summary.append(f"[Face position heatmap saved to {os.path.join(output_dir, 'face_position_heatmap.png')}]")
                else:
                    # plt.show()
                    pass
    
    # 4. Correlation analysis
    if quality_col:
        summary.append("")
        summary.append("-" * 80)
        summary.append("CORRELATION ANALYSIS")
        summary.append("-" * 80)
        
        # Get all numeric columns
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and 
                        col not in ['Filename', 'FaceWidth', 'FaceHeight', 'FaceCenterX', 'FaceCenterY', 
                                  'NormX', 'NormY', 'QualityCategory']]
        
        if len(numeric_cols) > 1:
            # Calculate correlations with quality score
            # Calculate correlations with quality score
            # Filter out any unnamed columns and columns with too many NaN values
            valid_cols = [col for col in numeric_cols if not col.startswith('Unnamed') 
                         and df[col].notna().sum() > len(df) * 0.5]  # Require at least 50% non-NaN values
            
            corr_with_quality = df[valid_cols].corrwith(df[quality_col]).sort_values(ascending=False)
            
            # Remove the quality score itself
            if quality_col in corr_with_quality:
                corr_with_quality = corr_with_quality.drop(quality_col)
            
            # Show top positive correlations
            summary.append("Top Positive Correlations with Quality Score:")
            for feature, corr in corr_with_quality.head(5).items():
                summary.append(f"  {feature}: {corr:.3f}")
            
            # Show top negative correlations
            summary.append("\nTop Negative Correlations with Quality Score:")
            for feature, corr in corr_with_quality.tail(5).items():
                summary.append(f"  {feature}: {corr:.3f}")

            # Add insights based on correlations
            summary.append("")
            strongest_pos = corr_with_quality.head(1)
            
            # Make sure we're not just looking at UnifiedQualityScore itself
            if strongest_pos.index[0] == quality_col or strongest_pos.index[0].replace('.scalar', '') == 'UnifiedQualityScore':
                # If the top correlation is the quality score itself, get the next one
                strongest_pos = corr_with_quality.iloc[1:2]
            
            strongest_neg = corr_with_quality.tail(1)
            
            summary.append(f"INSIGHT: {strongest_pos.index[0]} has the strongest positive influence on quality ({strongest_pos.values[0]:.3f}).")
            summary.append(f"INSIGHT: {strongest_neg.index[0]} has the strongest negative influence on quality ({strongest_neg.values[0]:.3f}).")
                        
           
    # 5. Key problems and recommendations
    summary.append("")
    summary.append("-" * 80)
    summary.append("KEY PROBLEMS AND RECOMMENDATIONS")
    summary.append("-" * 80)
    
    if problem_metrics:
        # Sort problems by severity
        problem_metrics.sort(key=lambda x: 0 if x['severity'] == 'CRITICAL' else 1)
        
        for i, problem in enumerate(problem_metrics):
            direction = "increase" if problem["direction"] == "higher" else "decrease"
            summary.append(f"{i+1}. {problem['severity']}: {problem['metric']} = {problem['value']:.2f}")
            summary.append(f"   Need to {direction} to reach threshold of {problem['threshold']:.2f}")
            
            # Add specific recommendations
            if 'Background' in problem['metric']:
                summary.append("   → Use a plain, uniform background for better face recognition.")
            elif 'Illumination' in problem['metric']:
                summary.append("   → Improve lighting uniformity to avoid shadows on the face.")
            elif 'Sharpness' in problem['metric']:
                summary.append("   → Use better focusing or higher resolution camera.")
            elif 'Compression' in problem['metric']:
                summary.append("   → Use less aggressive image compression settings.")
            elif 'Colour' in problem['metric']:
                summary.append("   → Ensure proper white balance and avoid color filters.")
            elif 'Eyes' in problem['metric']:
                summary.append("   → Ensure subjects keep their eyes open and visible.")
            elif 'Mouth' in problem['metric']:
                summary.append("   → Subjects should keep mouth closed or neutral.")
            elif 'Expression' in problem['metric']:
                summary.append("   → Subjects should maintain a neutral expression.")
            elif 'Pose' in problem['metric']:
                summary.append("   → Ensure subject is facing directly toward the camera with level head.")
    else:
        summary.append("No significant quality issues detected.")
    
    # 6. Overall assessment
    summary.append("")
    summary.append("-" * 80)
    summary.append("OVERALL ASSESSMENT")
    summary.append("-" * 80)
    
    if quality_col:
        # Calculate overall quality rating
        if q_mean >= 80:
            quality_rating = "EXCELLENT"
        elif q_mean >= 60:
            quality_rating = "GOOD"
        elif q_mean >= 40:
            quality_rating = "FAIR"
        else:
            quality_rating = "POOR"
        
        summary.append(f"Overall Quality Rating: {quality_rating} ({q_mean:.2f}/100)")
        
        # Calculate pass rate
        passing_threshold = 40  # Minimum acceptable quality
        pass_rate = (df[quality_col] >= passing_threshold).mean() * 100
        
        summary.append(f"Pass Rate: {pass_rate:.1f}% of images meet minimum quality standards (>= {passing_threshold}).")
        
        # Summary of critical issues
        critical_count = sum(1 for p in problem_metrics if p['severity'] == 'CRITICAL')
        if critical_count > 0:
            summary.append(f"\nCritical Issues: {critical_count} metrics require immediate attention.")
        
        # Final recommendation
        summary.append("\nFinal Recommendation:")
        if quality_rating in ["EXCELLENT", "GOOD"]:
            summary.append("Dataset quality is sufficient for facial recognition tasks.")
        elif pass_rate > 80:
            summary.append("Dataset has acceptable quality overall, but specific improvements recommended.")
        else:
            summary.append("Dataset quality issues should be addressed before use in facial recognition systems.")
    
    # Join the summary
    full_summary = "\n".join(summary)
    
    # Save to file if output directory provided
    if output_dir:
        summary_path = os.path.join(output_dir, 'ofiq_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(full_summary)
        print(f"Summary saved to {summary_path}")
    
    return full_summary

# Example usage in notebook
# summary = generate_ofiq_summary('your_dataset.csv', 'output_dir', create_plots=True)
# print(summary)


# In[6]:


def create_visualization(csv_file, output_dir):
    visualize_pose_angles(csv_file, output_dir)
    visualize_face_boundary_boxes(csv_file, output_dir)
    visualize_face_position_heatmap(csv_file, output_dir)
    generate_ofiq_summary(csv_file, output_dir)


# In[16]:


def main():
    """Main function to run unified pose visualization and summary generation"""
    
    # Base directory for all datasets
    base_dir = '/home/meem/backup/Age Datasets/OFIQ-visualizations/balanced_age_datasets'  # Update this path
    
    # Define all datasets
    datasets = [
        'AFAD', 'APPA-REAL', 'Adience', 'AgeDB', 'FG-NET', 
        'FairFace', 'Groups', 'IMDB-WIKI', 'Juvenile-80K', 
        'LAGENDA', 'MORPH', 'UTKFace'
    ]
    
    print("Loading and combining all datasets...")
    combined_df = pd.DataFrame()
    
    # Load and combine all OFIQ datasets
    for dataset in datasets:
        csv_file = os.path.join(base_dir, dataset, 'ofiq_balanced.csv')
        
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found. Skipping {dataset}")
            continue
        
        try:
            df = pd.read_csv(csv_file, sep=';')
            
            # Clean up the dataframe - remove duplicate columns
            df = clean_dataframe_columns(df)
            
            # Add source dataset column
            df['dataset_source'] = dataset
            
            # Combine with the main dataframe
            if combined_df.empty:
                combined_df = df.copy()
            else:
                # Only keep columns that exist in both dataframes
                common_columns = list(set(combined_df.columns) & set(df.columns))
                combined_df = pd.concat([combined_df[common_columns], df[common_columns]], ignore_index=True)
            
            print(f"✓ Loaded {len(df)} records from {dataset}")
            
        except Exception as e:
            print(f"✗ Error loading {dataset}: {e}")
            continue
    
    if combined_df.empty:
        print("No data loaded. Exiting.")
        return
    
    print(f"\nCombined dataset: {len(combined_df)} total records from {combined_df['dataset_source'].nunique()} datasets")
    print(f"Available columns: {len(combined_df.columns)}")
    
    # Print first few column names to verify
    print("First 10 columns:", combined_df.columns[:10].tolist())
    
    # Save combined dataset for reference
    output_dir = 'unified_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a clean CSV file for the combined data
    temp_csv = os.path.join(output_dir, 'combined_ofiq_clean.csv')
    combined_df.to_csv(temp_csv, sep=';', index=False)
    
    print("Running unified analysis...")
    try:
        # Run the visualization pipeline on combined data
        visualize_pose_angles(temp_csv, output_dir)
        visualize_face_boundary_boxes(temp_csv, output_dir)
        visualize_face_position_heatmap(temp_csv, output_dir)
        generate_ofiq_summary(temp_csv, output_dir)
        
        # Create dataset breakdown summary
        create_dataset_breakdown_summary(combined_df, output_dir)
        
        print(f"✓ Completed unified analysis in {output_dir}")
        
    except Exception as e:
        print(f"✗ Error in unified analysis: {e}")
        import traceback
        traceback.print_exc()

def clean_dataframe_columns(df):
    """Clean dataframe by removing duplicate columns and keeping only the first occurrence"""
    
    # Get the original column names
    original_columns = df.columns.tolist()
    
    # Find unique column names (keep first occurrence)
    seen_columns = set()
    columns_to_keep = []
    
    for col in original_columns:
        if col not in seen_columns:
            seen_columns.add(col)
            columns_to_keep.append(col)
    
    # Create new dataframe with only unique columns
    df_clean = df[columns_to_keep].copy()
    
    print(f"  Cleaned columns: {len(original_columns)} → {len(columns_to_keep)}")
    
    # Also remove any unnamed columns that might be causing issues
    df_clean = df_clean.loc[:, ~df_clean.columns.str.contains('^Unnamed')]
    
    print(f"  After removing unnamed: {len(df_clean.columns)} columns")
    
    return df_clean

def create_dataset_breakdown_summary(combined_df, output_dir):
    """Create additional summary showing breakdown by dataset"""
    
    # Check if we have the quality score column
    quality_col = None
    if 'UnifiedQualityScore' in combined_df.columns:
        quality_col = 'UnifiedQualityScore'
    elif 'UnifiedQualityScore.scalar' in combined_df.columns:
        quality_col = 'UnifiedQualityScore.scalar'
    
    if quality_col is None:
        print("Warning: No quality score column found for dataset breakdown")
        return
    
    # Dataset statistics
    dataset_stats = combined_df.groupby('dataset_source').agg({
        quality_col: ['count', 'mean', 'std'],
        'Filename': 'count'
    }).round(2)
    
    # Flatten column names
    dataset_stats.columns = ['_'.join(col).strip() for col in dataset_stats.columns]
    dataset_stats = dataset_stats.rename(columns={'Filename_count': 'Total_Images'})
    
    # Save dataset breakdown
    dataset_stats.to_csv(os.path.join(output_dir, 'dataset_breakdown.csv'))
    
    # Create visualization of dataset contributions
    plt.figure(figsize=(15, 10))
    
    # Dataset size comparison
    plt.subplot(2, 3, 1)
    dataset_counts = combined_df['dataset_source'].value_counts()
    plt.pie(dataset_counts.values, labels=dataset_counts.index, autopct='%1.1f%%')
    plt.title('Dataset Size Distribution')
    
    # Quality score by dataset
    plt.subplot(2, 3, 2)
    combined_df.boxplot(column=quality_col, by='dataset_source', ax=plt.gca())
    plt.xticks(rotation=45)
    plt.title('Quality Score by Dataset')
    plt.suptitle('')  # Remove default title
    
    # Dataset sizes bar chart
    plt.subplot(2, 3, 3)
    dataset_counts.plot(kind='bar')
    plt.title('Number of Images per Dataset')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    
    # Quality distribution
    plt.subplot(2, 3, 4)
    combined_df[quality_col].hist(bins=50, alpha=0.7)
    plt.title('Overall Quality Score Distribution')
    plt.xlabel('Quality Score')
    plt.ylabel('Frequency')
    
    # Mean quality by dataset
    plt.subplot(2, 3, 5)
    mean_quality = combined_df.groupby('dataset_source')[quality_col].mean().sort_values(ascending=False)
    mean_quality.plot(kind='bar')
    plt.title('Mean Quality Score by Dataset')
    plt.xticks(rotation=45)
    plt.ylabel('Mean Quality Score')
    
    # Summary statistics
    plt.subplot(2, 3, 6)
    stats_text = f"Combined Dataset Summary:\n\n"
    stats_text += f"Total Images: {len(combined_df):,}\n"
    stats_text += f"Number of Datasets: {combined_df['dataset_source'].nunique()}\n"
    stats_text += f"Overall Mean Quality: {combined_df[quality_col].mean():.2f}\n"
    stats_text += f"Quality Std Dev: {combined_df[quality_col].std():.2f}\n"
    stats_text += f"Quality Range: {combined_df[quality_col].min():.1f} - {combined_df[quality_col].max():.1f}\n"
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_breakdown_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Dataset breakdown saved to {output_dir}")

# Alternative main function that loads datasets individually to avoid column duplication
def main_alternative():
    """Alternative main function that processes each dataset individually then combines results"""
    
    # Base directory for all datasets
    base_dir = '/home/meem/backup/Age Datasets/OFIQ-visualizations/balanced_age_datasets'  # Update this path
    
    # Define all datasets
    datasets = [
        'AFAD', 'APPA-REAL', 'Adience', 'AgeDB', 'FG-NET', 
        'FairFace', 'Groups', 'IMDB-WIKI', 'Juvenile-80K', 
        'LAGENDA', 'MORPH', 'UTKFace'
    ]
    
    print("Processing each dataset individually then combining results...")
    
    # Create output directory
    output_dir = 'unified_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    combined_stats = []
    
    # Process each dataset individually
    for dataset in datasets:
        csv_file = os.path.join(base_dir, dataset, 'ofiq_balanced.csv')
        
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found. Skipping {dataset}")
            continue
        
        try:
            print(f"\nProcessing {dataset}...")
            
            # Create individual output directory
            individual_output = os.path.join(output_dir, f'individual_{dataset}')
            os.makedirs(individual_output, exist_ok=True)
            
            # Run analysis on individual dataset
            df = pd.read_csv(csv_file, sep=';')
            
            # Clean the dataframe
            df_clean = clean_dataframe_columns(df)
            
            # Save clean version
            clean_csv = os.path.join(individual_output, f'{dataset}_clean.csv')
            df_clean.to_csv(clean_csv, sep=';', index=False)
            
            # Run individual analysis
            try:
                generate_ofiq_summary(clean_csv, individual_output)
                
                # Extract key statistics
                if 'UnifiedQualityScore' in df_clean.columns:
                    quality_col = 'UnifiedQualityScore'
                elif 'UnifiedQualityScore.scalar' in df_clean.columns:
                    quality_col = 'UnifiedQualityScore.scalar'
                else:
                    quality_col = None
                
                if quality_col:
                    stats = {
                        'dataset': dataset,
                        'count': len(df_clean),
                        'mean_quality': df_clean[quality_col].mean(),
                        'std_quality': df_clean[quality_col].std(),
                        'min_quality': df_clean[quality_col].min(),
                        'max_quality': df_clean[quality_col].max()
                    }
                    combined_stats.append(stats)
                
                all_results[dataset] = {
                    'success': True,
                    'count': len(df_clean),
                    'quality_available': quality_col is not None
                }
                
                print(f"✓ Processed {dataset}: {len(df_clean)} records")
                
            except Exception as e:
                print(f"✗ Error processing {dataset}: {e}")
                all_results[dataset] = {
                    'success': False,
                    'error': str(e)
                }
                
        except Exception as e:
            print(f"✗ Error loading {dataset}: {e}")
            continue
    
    # Create combined summary
    if combined_stats:
        create_combined_summary(combined_stats, output_dir)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("UNIFIED ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    total_images = 0
    successful_datasets = 0
    
    for dataset, result in all_results.items():
        if result['success']:
            status = f"✓ SUCCESS - {result['count']} images"
            total_images += result['count']
            successful_datasets += 1
        else:
            status = f"✗ FAILED - {result.get('error', 'Unknown error')}"
        
        print(f"{dataset:15}: {status}")
    
    print(f"{'='*60}")
    print(f"Total successful datasets: {successful_datasets}")
    print(f"Total images processed: {total_images:,}")
    print(f"Results saved to: {output_dir}")

def create_combined_summary(combined_stats, output_dir):
    """Create a combined summary from individual dataset statistics"""
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(combined_stats)
    
    # Save combined statistics
    stats_df.to_csv(os.path.join(output_dir, 'combined_statistics.csv'), index=False)
    
    # Create combined visualization
    plt.figure(figsize=(15, 10))
    
    # Dataset sizes
    plt.subplot(2, 3, 1)
    plt.bar(stats_df['dataset'], stats_df['count'])
    plt.title('Images per Dataset')
    plt.xticks(rotation=45)
    plt.ylabel('Number of Images')
    
    # Mean quality scores
    plt.subplot(2, 3, 2)
    plt.bar(stats_df['dataset'], stats_df['mean_quality'])
    plt.title('Mean Quality Score by Dataset')
    plt.xticks(rotation=45)
    plt.ylabel('Mean Quality Score')
    
    # Quality range (min-max)
    plt.subplot(2, 3, 3)
    plt.bar(stats_df['dataset'], stats_df['max_quality'] - stats_df['min_quality'])
    plt.title('Quality Score Range by Dataset')
    plt.xticks(rotation=45)
    plt.ylabel('Quality Range')
    
    # Pie chart of dataset sizes
    plt.subplot(2, 3, 4)
    plt.pie(stats_df['count'], labels=stats_df['dataset'], autopct='%1.1f%%')
    plt.title('Dataset Size Distribution')
    
    # Quality vs Size scatter
    plt.subplot(2, 3, 5)
    plt.scatter(stats_df['count'], stats_df['mean_quality'])
    for i, dataset in enumerate(stats_df['dataset']):
        plt.annotate(dataset, (stats_df['count'].iloc[i], stats_df['mean_quality'].iloc[i]))
    plt.xlabel('Dataset Size')
    plt.ylabel('Mean Quality Score')
    plt.title('Dataset Size vs Quality')
    
    # Summary statistics
    plt.subplot(2, 3, 6)
    total_images = stats_df['count'].sum()
    overall_mean = (stats_df['mean_quality'] * stats_df['count']).sum() / total_images
    
    summary_text = f"Combined Analysis Summary:\n\n"
    summary_text += f"Total Datasets: {len(stats_df)}\n"
    summary_text += f"Total Images: {total_images:,}\n"
    summary_text += f"Overall Mean Quality: {overall_mean:.2f}\n"
    summary_text += f"Highest Quality Dataset: {stats_df.loc[stats_df['mean_quality'].idxmax(), 'dataset']}\n"
    summary_text += f"Largest Dataset: {stats_df.loc[stats_df['count'].idxmax(), 'dataset']}\n"
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_analysis_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Combined summary created")


# ==============================================================================
# COMBINED GROUNDTRUTH GENERATION FUNCTIONS
# ==============================================================================

def assign_random_gender(num_records, seed=42):
    """
    Assign random gender values with realistic distribution:
    - 49.5% male ('m')
    - 49.5% female ('f') 
    - 1% non-binary/other ('x')
    """
    np.random.seed(seed)  # For reproducible results
    
    # Create probability distribution
    # 1 in 200 = 0.5% for 'x', split remaining equally
    probabilities = [0.495, 0.495, 0.01]  # m, f, x
    gender_values = ['m', 'f', 'x']
    
    # Generate random gender assignments
    assigned_genders = np.random.choice(
        gender_values, 
        size=num_records, 
        p=probabilities
    )
    
    return assigned_genders.tolist()

def load_and_combine_groundtruth_data(base_dir, datasets):
    """
    Load and combine groundtruth data from multiple datasets,
    handling both single age and age range formats
    """
    
    # Define datasets by type
    single_age_datasets = [
        'AFAD', 'AgeDB', 'FG-NET', 'IMDB-WIKI', 
        'Juvenile-80K', 'LAGENDA', 'MORPH', 'UTKFace'
    ]
    
    age_range_datasets = [
        'APPA-REAL', 'Adience', 'FairFace', 'Groups'
    ]
    
    combined_single_age = pd.DataFrame()
    combined_age_range = pd.DataFrame()
    
    print("Loading groundtruth data from all datasets...")
    
    for dataset in datasets:
        gt_file = os.path.join(base_dir, dataset, 'groundtruth_balanced.csv')
        
        if not os.path.exists(gt_file):
            print(f"Warning: {gt_file} not found. Skipping {dataset}")
            continue
        
        try:
            gt_df = pd.read_csv(gt_file)
            print(f"✓ Loaded {len(gt_df)} groundtruth records from {dataset}")
            
            # Add dataset source column
            gt_df['dataset_source'] = dataset
            
            # Determine if this is single age or age range dataset
            if dataset in single_age_datasets:
                # Standardize single age format
                gt_clean = standardize_single_age_format(gt_df, dataset)
                if gt_clean is not None:
                    combined_single_age = pd.concat([combined_single_age, gt_clean], ignore_index=True)
                    
            elif dataset in age_range_datasets:
                # Standardize age range format
                gt_clean = standardize_age_range_format(gt_df, dataset)
                if gt_clean is not None:
                    combined_age_range = pd.concat([combined_age_range, gt_clean], ignore_index=True)
            
        except Exception as e:
            print(f"✗ Error loading groundtruth from {dataset}: {e}")
            continue
    
    return combined_single_age, combined_age_range

def standardize_single_age_format(df, dataset_name):
    """
    Standardize single age groundtruth to common format:
    [path, age, gender, dataset_source]
    """
    try:
        # Create standardized dataframe
        standardized = pd.DataFrame()
        
        # Map common column variations to standard names
        column_mappings = {
            'path': ['path', 'filepath', 'file_path', 'filename', 'image_path', 'Filename'],
            'age': ['age', 'Age', 'age_label', 'target_age', 'apparent_age'],
            'gender': ['gender', 'Gender', 'sex', 'Sex', 'gender_label']
        }
        
        # Find the appropriate columns
        found_columns = {}
        for standard_name, variations in column_mappings.items():
            for variation in variations:
                if variation in df.columns:
                    found_columns[standard_name] = variation
                    break
        
        # Check if we found all required columns
        if 'path' not in found_columns:
            # Try to use the first column if it looks like a path
            first_col = df.columns[0]
            first_val = str(df.iloc[0, 0])
            if any(ext in first_val.lower() for ext in ['.jpg', '.jpeg', '.png']) or '/' in first_val:
                found_columns['path'] = first_col
                print(f"  Using '{first_col}' as path column for {dataset_name}")
            else:
                print(f"  Warning: Could not find path column for {dataset_name}")
                return None
        
        if 'age' not in found_columns:
            # Look for numeric columns that might be age
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    age_values = df[col].dropna()
                    if len(age_values) > 0 and age_values.min() >= 0 and age_values.max() <= 120:
                        found_columns['age'] = col
                        print(f"  Using '{col}' as age column for {dataset_name}")
                        break
            
            if 'age' not in found_columns:
                print(f"  Warning: Could not find age column for {dataset_name}")
                return None
        
        if 'gender' not in found_columns:
            # Look for categorical columns that might be gender
            for col in df.columns:
                if df[col].dtype == 'object':
                    unique_vals = df[col].dropna().astype(str).str.lower().unique()
                    if any(val in ['m', 'f', 'male', 'female', '0', '1'] for val in unique_vals):
                        found_columns['gender'] = col
                        print(f"  Using '{col}' as gender column for {dataset_name}")
                        break
            
            if 'gender' not in found_columns:
                print(f"  Warning: Could not find gender column for {dataset_name}. Using 'unknown'")
                standardized['gender'] = 'unknown'
        
        # Map the columns
        for standard_name, original_name in found_columns.items():
            standardized[standard_name] = df[original_name]
        
        # Add missing gender column if needed
        if 'gender' not in standardized.columns:
            print(f"  No gender data found for {dataset_name}. Assigning random gender values...")
            standardized['gender'] = assign_random_gender(len(standardized))
        
        # Add dataset source
        standardized['dataset_source'] = dataset_name
        
        # Clean and validate age values
        if 'age' in standardized.columns:
            standardized['age'] = pd.to_numeric(standardized['age'], errors='coerce')
            # Remove invalid ages
            initial_count = len(standardized)
            standardized = standardized[(standardized['age'] >= 0) & (standardized['age'] <= 120)]
            if len(standardized) < initial_count:
                print(f"  Removed {initial_count - len(standardized)} records with invalid ages")
        
        # Standardize gender values
        if 'gender' in standardized.columns:
            gender_mapping = {
                'male': 'm', 'Male': 'm', 'MALE': 'm', 'M': 'm',
                'female': 'f', 'Female': 'f', 'FEMALE': 'f', 'F': 'f',
                '0': 'f', '1': 'm'  # Common numeric encoding
            }
            standardized['gender'] = standardized['gender'].astype(str).map(
                lambda x: gender_mapping.get(x, x.lower() if x.lower() in ['m', 'f'] else 'unknown')
            )
        
        print(f"  Standardized {len(standardized)} records from {dataset_name}")
        return standardized
        
    except Exception as e:
        print(f"  Error standardizing {dataset_name}: {e}")
        return None

def standardize_age_range_format(df, dataset_name):
    """
    Standardize age range groundtruth to common format:
    [path, lower_age, upper_age, mean_age, gender, dataset_source]
    """
    try:
        # Create standardized dataframe
        standardized = pd.DataFrame()
        
        # Map common column variations
        column_mappings = {
            'path': ['path', 'filepath', 'file_path', 'filename', 'image_path', 'Filename'],
            'lower_age': ['lower_age', 'min_age', 'age_min', 'age_lower', 'start_age', 'age_low'],
            'upper_age': ['upper_age', 'max_age', 'age_max', 'age_upper', 'end_age', 'age_high'],
            'mean_age': ['mean_age', 'avg_age', 'age_mean', 'age_avg', 'center_age', 'age'],
            'gender': ['gender', 'Gender', 'sex', 'Sex', 'gender_label']
        }
        
        # Find the appropriate columns
        found_columns = {}
        for standard_name, variations in column_mappings.items():
            for variation in variations:
                if variation in df.columns:
                    found_columns[standard_name] = variation
                    break
        
        # Handle path column
        if 'path' not in found_columns:
            first_col = df.columns[0]
            first_val = str(df.iloc[0, 0])
            if any(ext in first_val.lower() for ext in ['.jpg', '.jpeg', '.png']) or '/' in first_val:
                found_columns['path'] = first_col
                print(f"  Using '{first_col}' as path column for {dataset_name}")
            else:
                print(f"  Warning: Could not find path column for {dataset_name}")
                return None
        
        # Handle age columns - need at least 2 of the 3 age columns
        age_columns_found = sum(1 for col in ['lower_age', 'upper_age', 'mean_age'] if col in found_columns)
        
        if age_columns_found < 2:
            # Try to find numeric columns that might be ages
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            potential_age_cols = []
            
            for col in numeric_cols:
                age_values = df[col].dropna()
                if len(age_values) > 0 and age_values.min() >= 0 and age_values.max() <= 120:
                    potential_age_cols.append(col)
            
            # Sort by mean value to guess which is lower/upper/mean
            if len(potential_age_cols) >= 2:
                col_means = {col: df[col].mean() for col in potential_age_cols}
                sorted_cols = sorted(col_means.items(), key=lambda x: x[1])
                
                if len(sorted_cols) >= 3:
                    found_columns['lower_age'] = sorted_cols[0][0]
                    found_columns['mean_age'] = sorted_cols[1][0]
                    found_columns['upper_age'] = sorted_cols[2][0]
                elif len(sorted_cols) == 2:
                    found_columns['lower_age'] = sorted_cols[0][0]
                    found_columns['upper_age'] = sorted_cols[1][0]
                
                print(f"  Detected age columns for {dataset_name}: {list(found_columns.keys())}")
        
        # Map the found columns
        for standard_name, original_name in found_columns.items():
            standardized[standard_name] = df[original_name]
        
        # Calculate missing age values
        if 'lower_age' in standardized.columns and 'upper_age' in standardized.columns and 'mean_age' not in standardized.columns:
            standardized['mean_age'] = (standardized['lower_age'] + standardized['upper_age']) / 2
            print(f"  Calculated mean_age for {dataset_name}")
        elif 'mean_age' in standardized.columns and ('lower_age' not in standardized.columns or 'upper_age' not in standardized.columns):
            # If we only have mean age, create dummy bounds
            standardized['lower_age'] = standardized['mean_age'] - 2.5
            standardized['upper_age'] = standardized['mean_age'] + 2.5
            print(f"  Created age bounds from mean_age for {dataset_name}")
        
        # Handle gender column
        if 'gender' not in found_columns:
            for col in df.columns:
                if df[col].dtype == 'object':
                    unique_vals = df[col].dropna().astype(str).str.lower().unique()
                    if any(val in ['m', 'f', 'male', 'female', '0', '1'] for val in unique_vals):
                        found_columns['gender'] = col
                        standardized['gender'] = df[col]
                        print(f"  Using '{col}' as gender column for {dataset_name}")
                        break
            
            if 'gender' not in standardized.columns:
                print(f"  No gender column found for {dataset_name}. Assigning random gender values...")
                standardized['gender'] = assign_random_gender(len(standardized))
        
        # Add dataset source
        standardized['dataset_source'] = dataset_name
        
        # Clean and validate data
        for age_col in ['lower_age', 'upper_age', 'mean_age']:
            if age_col in standardized.columns:
                standardized[age_col] = pd.to_numeric(standardized[age_col], errors='coerce')
        
        # Remove invalid records
        initial_count = len(standardized)
        valid_mask = True
        for age_col in ['lower_age', 'upper_age', 'mean_age']:
            if age_col in standardized.columns:
                valid_mask = valid_mask & (standardized[age_col] >= 0) & (standardized[age_col] <= 120)
        
        standardized = standardized[valid_mask]
        if len(standardized) < initial_count:
            print(f"  Removed {initial_count - len(standardized)} records with invalid ages")
        
        # Standardize gender values
        if 'gender' in standardized.columns:
            gender_mapping = {
                'male': 'm', 'Male': 'm', 'MALE': 'm', 'M': 'm',
                'female': 'f', 'Female': 'f', 'FEMALE': 'f', 'F': 'f',
                '0': 'f', '1': 'm'
            }
            standardized['gender'] = standardized['gender'].astype(str).map(
                lambda x: gender_mapping.get(x, x.lower() if x.lower() in ['m', 'f'] else 'unknown')
            )
        
        print(f"  Standardized {len(standardized)} age-range records from {dataset_name}")
        return standardized
        
    except Exception as e:
        print(f"  Error standardizing age-range data for {dataset_name}: {e}")
        return None

def save_combined_groundtruth(combined_single_age, combined_age_range, output_dir):
    """Save the combined groundtruth data to CSV files"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save single age groundtruth
    if not combined_single_age.empty:
        single_age_path = os.path.join(output_dir, 'combined_single_age_groundtruth.csv')
        combined_single_age.to_csv(single_age_path, index=False)
        print(f"✓ Saved combined single-age groundtruth: {len(combined_single_age)} records")
        
        # Print summary
        print(f"  Single-age datasets: {combined_single_age['dataset_source'].nunique()}")
        print(f"  Age range: {combined_single_age['age'].min():.0f}-{combined_single_age['age'].max():.0f}")
        gender_counts = combined_single_age['gender'].value_counts().to_dict()
        print(f"  Gender distribution: {gender_counts}")
        if 'x' in gender_counts:
            print(f"    Non-binary percentage: {gender_counts['x']/len(combined_single_age)*100:.1f}%")
    
    # Save age range groundtruth
    if not combined_age_range.empty:
        age_range_path = os.path.join(output_dir, 'combined_age_range_groundtruth.csv')
        combined_age_range.to_csv(age_range_path, index=False)
        print(f"✓ Saved combined age-range groundtruth: {len(combined_age_range)} records")
        
        # Print summary
        print(f"  Age-range datasets: {combined_age_range['dataset_source'].nunique()}")
        if 'mean_age' in combined_age_range.columns:
            print(f"  Mean age range: {combined_age_range['mean_age'].min():.0f}-{combined_age_range['mean_age'].max():.0f}")
        gender_counts = combined_age_range['gender'].value_counts().to_dict()
        print(f"  Gender distribution: {gender_counts}")
        if 'x' in gender_counts:
            print(f"    Non-binary percentage: {gender_counts['x']/len(combined_age_range)*100:.1f}%")
    
    return single_age_path if not combined_single_age.empty else None, age_range_path if not combined_age_range.empty else None

# ==============================================================================
# UPDATED PLOT SCRIPT MAIN FUNCTION WITH GROUNDTRUTH GENERATION
# ==============================================================================

def main_with_groundtruth_generation():
    """Main function that generates combined groundtruth and runs unified analysis"""
    
    # Base directory for all datasets
    base_dir = '/home/meem/backup/Age Datasets/OFIQ-visualizations/balanced_age_datasets'  # Update this path
    
    # Define all datasets
    datasets = [
        'AFAD', 'APPA-REAL', 'Adience', 'AgeDB', 'FG-NET', 
        'FairFace', 'Groups', 'IMDB-WIKI', 'Juvenile-80K', 
        'LAGENDA', 'MORPH', 'UTKFace'
    ]
    
    # Create output directory
    output_dir = 'unified_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("STEP 1: GENERATING COMBINED GROUNDTRUTH DATA")
    print("=" * 60)
    
    # Load and combine groundtruth data
    combined_single_age, combined_age_range = load_and_combine_groundtruth_data(base_dir, datasets)
    
    # Save combined groundtruth files
    single_age_gt_path, age_range_gt_path = save_combined_groundtruth(
        combined_single_age, combined_age_range, output_dir
    )
    
    print("\n" + "=" * 60)
    print("STEP 2: GENERATING COMBINED OFIQ DATA")
    print("=" * 60)
    
    # Load and combine OFIQ data (reuse from pose script)
    combined_ofiq = pd.DataFrame()
    
    for dataset in datasets:
        csv_file = os.path.join(base_dir, dataset, 'ofiq_balanced.csv')
        
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found. Skipping {dataset}")
            continue
        
        try:
            df = pd.read_csv(csv_file, sep=';')
            df = clean_dataframe_columns(df)  # Clean duplicate columns
            df['dataset_source'] = dataset
            
            if combined_ofiq.empty:
                combined_ofiq = df.copy()
            else:
                common_columns = list(set(combined_ofiq.columns) & set(df.columns))
                combined_ofiq = pd.concat([combined_ofiq[common_columns], df[common_columns]], ignore_index=True)
            
            print(f"✓ Loaded OFIQ data from {dataset}: {len(df)} records")
            
        except Exception as e:
            print(f"✗ Error loading OFIQ from {dataset}: {e}")
            continue
    
    # Save combined OFIQ data
    if not combined_ofiq.empty:
        combined_ofiq_path = os.path.join(output_dir, 'combined_ofiq_clean.csv')
        combined_ofiq.to_csv(combined_ofiq_path, sep=';', index=False)
        print(f"✓ Saved combined OFIQ data: {len(combined_ofiq)} records")
    
    print("\n" + "=" * 60)
    print("STEP 3: RUNNING UNIFIED ANALYSIS")
    print("=" * 60)
    
    # Run analysis on single age datasets
    if single_age_gt_path and not combined_ofiq.empty:
        print("\nRunning single-age analysis...")
        try:
            results = analyze_age_gender_ofiq(
                single_age_gt_path, 
                combined_ofiq_path, 
                os.path.join(output_dir, 'single_age_analysis')
            )
            print(f"✓ Single-age analysis complete: {results['matches_found']} matches")
        except Exception as e:
            print(f"✗ Error in single-age analysis: {e}")
    
    # Run analysis on age range datasets
    if age_range_gt_path and not combined_ofiq.empty:
        print("\nRunning age-range analysis...")
        try:
            results = analyze_age_range_ofiq(
                age_range_gt_path, 
                combined_ofiq_path, 
                os.path.join(output_dir, 'age_range_analysis')
            )
            print(f"✓ Age-range analysis complete: {results['matches_found']} matches")
        except Exception as e:
            print(f"✗ Error in age-range analysis: {e}")
    
    # Create overall comparison
    if not combined_single_age.empty or not combined_age_range.empty:
        print("\nCreating unified comparison...")
        create_unified_comparison(combined_single_age, combined_age_range, output_dir)
    
    print(f"\n✓ Unified analysis complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main_with_groundtruth_generation()


# In[15]:


if __name__ == "__main__":
    # You can choose which main function to run:
    # main()  # Original approach with column cleaning
    main_alternative()  # Individual processing then combining results


# In[57]:


def main():
    """Main function to run for pose visualization"""
    # Specify your CSV file path here
    csv_file = '/home/meem/backup/Age Datasets/AdienceGender Dataset/ofiq.csv'  # Update with your file path
    output_dir = 'pose/adience'  # Output directory for tileboards
    
    # Create the comprehensive comparison tileboards
    create_visualization(csv_file, output_dir)

    # """Main function to run the visualization code"""
    # #1
    # # Specify your CSV file path here
    csv_file = '/home/meem/backup/Age Datasets/fairface/ofiq.csv'  # Update with your file path
    output_dir = 'pose/fairface'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    create_visualization(csv_file, output_dir)

    # #2
    csv_file = '/home/meem/backup/Age Datasets/afad/ofiq.csv'  # Update with your file path
    output_dir = 'pose/afad'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    create_visualization(csv_file, output_dir)

    # #4    
    csv_file = '/home/meem/backup/Age Datasets/agedb/ofiq.csv'  # Update with your file path
    output_dir = 'pose/agedb'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    create_visualization(csv_file, output_dir)

    # #5    
    csv_file = '/home/meem/backup/Age Datasets/appa-real-release/ofiq.csv'  # Update with your file path
    output_dir = 'pose/appa-real'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    create_visualization(csv_file, output_dir)

    # #6    
    csv_file = '/home/meem/backup/Age Datasets/FGNET Dataset/ofiq.csv'  # Update with your file path
    output_dir = 'pose/fgnet'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    create_visualization(csv_file, output_dir)

    # # #7    
    csv_file = '/home/meem/backup/Age Datasets/Groups-of-People Dataset/ofiq.csv'  # Update with your file path
    output_dir = 'pose/groups'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    create_visualization(csv_file, output_dir)

    # #8    
    csv_file = '/home/meem/backup/Age Datasets/IMDB - WIKI/ofiq.csv'  # Update with your file path
    output_dir = 'pose/imdb-wiki'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    create_visualization(csv_file, output_dir)

    # #9     
    csv_file = '/home/meem/backup/Age Datasets/Juvenile_Dataset/Juvenile_Dataset/ofiq.csv'  # Update with your file path
    output_dir = 'pose/juvenil'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    create_visualization(csv_file, output_dir)  # Update with your file path


    # #10     
    csv_file = '/home/meem/backup/Age Datasets/lagenda/ofiq.csv'  # Update with your file path
    output_dir = 'pose/lagenda'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    create_visualization(csv_file, output_dir)  # Update with your file path

    # #11     
    csv_file = '/home/meem/backup/Age Datasets/Morph2 Dataset/ofiq.csv'  # Update with your file path
    output_dir = 'pose/morph2'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    create_visualization(csv_file, output_dir)  # Update with your file path

    # #12     
    csv_file = '/home/meem/backup/Age Datasets/utkface-cropped/ofiq.csv'  # Update with your file path
    output_dir = 'pose/utkface'  # Output directory for visualizations
    
    # # Run the visualization pipeline
    create_visualization(csv_file, output_dir)  # Update with your file path


if __name__ == "__main__":
    main()

