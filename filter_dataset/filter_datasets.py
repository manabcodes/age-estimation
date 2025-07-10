#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import shutil
from typing import Dict, List, Tuple, Optional
import warnings
from collections import Counter
import json
from tqdm.notebook import tqdm
# warnings.filterwarnings('ignore')


# In[3]:


class BalancedAgeDatasetProcessor:
    """
    Comprehensive dataset processor for age estimation with quality filtering,
    pose clustering, and demographic balancing
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize with filtering thresholds and balancing parameters
        """
        self.config = config or {
            # Primary filtering thresholds
            'unified_quality_score_min': 40,
            'expression_neutrality_min': 70,
            'expression_neutrality_relaxed': 50,  # ✓ 
            'sharpness_min': 60,
            'illumination_uniformity_min': 70,
            
            # Pose filtering thresholds
            'frontal_yaw_max': 15,
            'frontal_pitch_max': 10,
            'frontal_roll_max': 10,
            'moderate_pose_max': 30,
            'extreme_pose_max': 45, 
            
            # Age balancing parameters
            'age_bin_size': 5,  # Group ages into 5-year bins
            'min_samples_per_age_bin': 10,  # Minimum samples to keep an age bin
            'target_samples_per_age_bin': None,  # Auto-calculate if None
            
            # Gender balancing (when available)
            'balance_gender': True,
            'gender_ratio_tolerance': 0.1,  # Allow 10% deviation from 50-50
            
            # Pose clustering parameters
            'pose_clusters': 5,
            'balance_poses_within_age': True,
            
            # Quality-based adaptive thresholds
            'adaptive_thresholds': True,
            'preserve_demographic_balance': True,
        }
        if config:
            self.config.update(config)  # This merges instead of replacing
        
        # Store processing statistics
        self.processing_stats = {}
        self.pose_clusters = None
        self.scaler = StandardScaler()
        self.age_bins = None
        
    def load_dataset(self, csv_path: str, groundtruth_path: str) -> pd.DataFrame:
        """
        Load and merge CSV data with groundtruth information
        """
        # Load quality metrics
        df_quality = pd.read_csv(csv_path, sep=';')
        df_quality.columns = df_quality.columns.str.strip()
        
        # Load groundtruth
        df_groundtruth = pd.read_csv(groundtruth_path)
        df_groundtruth.columns = df_groundtruth.columns.str.strip()
        
        # Print available columns for debugging
        print(f"Quality CSV columns: {list(df_quality.columns)}")
        print(f"Groundtruth CSV columns: {list(df_groundtruth.columns)}")
        
        # Standardize age column name
        df_groundtruth = self._standardize_age_column(df_groundtruth)
        
        # Merge datasets - try different merge strategies
        merged_df = self._merge_datasets(df_quality, df_groundtruth)
        
        # Check file existence
        if 'Filename' in merged_df.columns:
            merged_df['file_exists'] = merged_df['Filename'].apply(
                lambda x: os.path.exists(x) if pd.notna(x) else False
            )
            existing_files = merged_df['file_exists'].sum()
            total_files = len(merged_df)
            print(f"Found {existing_files}/{total_files} existing files ({existing_files/total_files*100:.1f}%)")
        
        return merged_df
    
    def _standardize_age_column(self, df_groundtruth: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize age column name to 'age'
        """
        age_columns = ['age', 'Age', 'AGE', 'mean_age', 'Mean_Age', 'MEAN_AGE', 'apparent_age', 'real_age']
        
        found_age_col = None
        for col in age_columns:
            if col in df_groundtruth.columns:
                found_age_col = col
                break
        
        if found_age_col is None:
            print("Warning: No age column found in groundtruth data")
            print(f"Available columns: {list(df_groundtruth.columns)}")
            return df_groundtruth
        
        # Rename to standard 'age' column
        if found_age_col != 'age':
            df_groundtruth = df_groundtruth.rename(columns={found_age_col: 'age'})
            print(f"Renamed age column '{found_age_col}' to 'age'")
        
        return df_groundtruth
    
    def _merge_datasets(self, df_quality: pd.DataFrame, df_groundtruth: pd.DataFrame) -> pd.DataFrame:
        """
        Intelligent merging of quality and groundtruth datasets
        """
        # Extract filenames for matching
        if 'Filename' in df_quality.columns:
            df_quality['base_filename'] = df_quality['Filename'].apply(
                lambda x: os.path.basename(x) if pd.notna(x) else ''
            )
        
        # Try different merge strategies based on available columns
        merge_columns = []
        
        # Strategy 1: Direct filename match
        if 'Filename' in df_groundtruth.columns:
            df_groundtruth['base_filename'] = df_groundtruth['Filename'].apply(
                lambda x: os.path.basename(x) if pd.notna(x) else ''
            )
            merge_columns = ['base_filename']
        
        # Strategy 2: Image ID or similar identifier
        elif 'image_id' in df_groundtruth.columns or 'ImageID' in df_groundtruth.columns:
            id_col = 'image_id' if 'image_id' in df_groundtruth.columns else 'ImageID'
            # Extract ID from filename
            df_quality['image_id'] = df_quality['base_filename'].str.extract(r'(\d+)')
            merge_columns = ['image_id']
        
        # Strategy 3: Index-based merge (if same order)
        else:
            print("Warning: No common identifier found. Using index-based merge.")
            df_quality['merge_index'] = df_quality.index
            df_groundtruth['merge_index'] = df_groundtruth.index
            merge_columns = ['merge_index']
        
        # Perform merge
        if merge_columns:
            merged_df = pd.merge(df_quality, df_groundtruth, on=merge_columns, how='inner', suffixes=('', '_gt'))
            print(f"Successfully merged {len(merged_df)} samples using {merge_columns}")
        else:
            print("Error: Could not determine merge strategy")
            return df_quality
        
        return merged_df
    
    def create_age_bins(self, ages: pd.Series) -> Tuple[pd.Series, Dict]:
        """
        Create age bins for balanced sampling
        """
        min_age = ages.min()
        max_age = ages.max()
        bin_size = self.config['age_bin_size']
        
        # Create age bins
        bins = range(int(min_age), int(max_age) + bin_size, bin_size)
        age_bins = pd.cut(ages, bins=bins, right=False, include_lowest=True)
        
        # Create mapping for bin labels
        bin_mapping = {}
        for i, interval in enumerate(age_bins.cat.categories):
            bin_mapping[interval] = f"{int(interval.left)}-{int(interval.right)-1}"
        
        # Apply mapping
        age_bin_labels = age_bins.map(bin_mapping)
        
        self.age_bins = bin_mapping
        return age_bin_labels, bin_mapping
    
    def analyze_demographic_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Analyze age and gender distribution in the dataset
        """
        analysis = {}
        
        # Age analysis - now standardized to 'age' column
        if 'age' in df.columns:
            ages = df['age'].dropna()
            
            analysis['age'] = {
                'count': len(ages),
                'mean': ages.mean(),
                'median': ages.median(),
                'std': ages.std(),
                'min': ages.min(),
                'max': ages.max(),
                'distribution': ages.value_counts().sort_index().to_dict()
            }
            
            # Create age bins
            age_bin_labels, _ = self.create_age_bins(ages)
            df['age_bin'] = age_bin_labels
            
            bin_distribution = age_bin_labels.value_counts().sort_index()
            analysis['age_bins'] = {
                'distribution': bin_distribution.to_dict(),
                'count': len(bin_distribution),
                'min_samples': bin_distribution.min(),
                'max_samples': bin_distribution.max(),
                'mean_samples': bin_distribution.mean()
            }
        
        # Gender analysis
        gender_cols = ['gender', 'Gender', 'sex', 'Sex']
        gender_col = None
        for col in gender_cols:
            if col in df.columns:
                gender_col = col
                break
        
        if gender_col:
            genders = df[gender_col].dropna()
            analysis['gender'] = {
                'count': len(genders),
                'distribution': genders.value_counts().to_dict(),
                'unique_values': genders.unique().tolist()
            }
        else:
            analysis['gender'] = None
            print("No gender information found in dataset")
        
        return analysis
    
    def apply_quality_filters(self, df: pd.DataFrame, dataset_name: str = "unknown") -> pd.DataFrame:
        """
        Apply quality filters while preserving demographic distribution
        """
        initial_count = len(df)
        filtered_df = df.copy()
        
        print(f"\n=== Quality Filtering for {dataset_name} ===")
        
        # Store initial demographic distribution
        initial_demo = self.analyze_demographic_distribution(filtered_df)
        
        # 1. Unified Quality Score Filter
        if 'UnifiedQualityScore' in df.columns:
            mask_quality = filtered_df['UnifiedQualityScore'] >= self.config['unified_quality_score_min']
            filtered_df = filtered_df[mask_quality]
            print(f"After quality score filter: {len(filtered_df)}/{initial_count} ({len(filtered_df)/initial_count*100:.1f}%)")
        
        # 2. Expression Neutrality Filter (adaptive)
        if 'ExpressionNeutrality' in df.columns:
            wild_datasets = ['adience', 'utkface', 'groups', 'fairface']
            threshold = (self.config['expression_neutrality_relaxed'] 
                        if any(wild in dataset_name.lower() for wild in wild_datasets)
                        else self.config['expression_neutrality_min'])
            
            mask_expression = filtered_df['ExpressionNeutrality'] >= threshold
            filtered_df = filtered_df[mask_expression]
            print(f"After expression filter (threshold={threshold}): {len(filtered_df)}/{initial_count} ({len(filtered_df)/initial_count*100:.1f}%)")
        
        # 3. Sharpness Filter
        if 'Sharpness' in df.columns:
            mask_sharpness = filtered_df['Sharpness'] >= self.config['sharpness_min']
            filtered_df = filtered_df[mask_sharpness]
            print(f"After sharpness filter: {len(filtered_df)}/{initial_count} ({len(filtered_df)/initial_count*100:.1f}%)")
        
        # 4. Illumination Filter
        # if 'IlluminationUniformity' in df.columns:
        #     mask_illumination = filtered_df['IlluminationUniformity'] >= self.config['illumination_uniformity_min']
        #     filtered_df = filtered_df[mask_illumination]
        #     print(f"After illumination filter: {len(filtered_df)}/{initial_count} ({len(filtered_df)/initial_count*100:.1f}%)")
        
        # 5. Pose Filter
        filtered_df = self.apply_pose_filters(filtered_df)
        
        # Analyze demographic impact
        final_demo = self.analyze_demographic_distribution(filtered_df)
        self._compare_demographic_distributions(initial_demo, final_demo, "Quality Filtering")
        
        return filtered_df
    
    def apply_pose_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply pose-based filtering
        """
        filtered_df = df.copy()
        initial_count = len(filtered_df)
        
        pose_cols = ['HeadPoseYaw', 'HeadPosePitch', 'HeadPoseRoll']
        if not all(col in df.columns for col in pose_cols):
            print("Warning: Pose columns not found, skipping pose filtering")
            return filtered_df
        
        # Remove extreme poses
        extreme_pose_mask = (
            (np.abs(filtered_df['HeadPoseYaw']) <= self.config['extreme_pose_max']) &
            (np.abs(filtered_df['HeadPosePitch']) <= self.config['extreme_pose_max']) &
            (np.abs(filtered_df['HeadPoseRoll']) <= self.config['extreme_pose_max'])
        )
        
        filtered_df = filtered_df[extreme_pose_mask]
        print(f"After extreme pose filter: {len(filtered_df)}/{initial_count} ({len(filtered_df)/initial_count*100:.1f}%)")
        
        return filtered_df
    
    def cluster_poses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Gaussian Mixture Model clustering for pose distribution
        """
        pose_cols = ['HeadPoseYaw', 'HeadPosePitch', 'HeadPoseRoll']
        
        if not all(col in df.columns for col in pose_cols):
            print("Warning: Pose columns not found, skipping pose clustering")
            return df
        
        # Prepare pose data
        pose_data = df[pose_cols].values
        valid_mask = ~np.isnan(pose_data).any(axis=1)
        pose_data_clean = pose_data[valid_mask]
        
        if len(pose_data_clean) < self.config['pose_clusters']:
            print(f"Warning: Not enough valid pose data for clustering ({len(pose_data_clean)} samples)")
            return df
        
        # Standardize and cluster
        pose_data_scaled = self.scaler.fit_transform(pose_data_clean)
        self.pose_clusters = GaussianMixture(
            n_components=self.config['pose_clusters'],
            random_state=42,
            covariance_type='full'
        )
        
        cluster_labels = self.pose_clusters.fit_predict(pose_data_scaled)
        
        # Add cluster information
        df_result = df.copy()
        df_result['pose_cluster'] = -1
        df_result.loc[df.index[valid_mask], 'pose_cluster'] = cluster_labels
        
        # Assign meaningful cluster names
        df_result['pose_cluster_name'] = df_result.apply(
            lambda row: self._get_cluster_name(row, df_result), axis=1
        )
        
        return df_result
    
    def _get_cluster_name(self, row: pd.Series, df: pd.DataFrame) -> str:
        """
        Assign meaningful names to pose clusters
        """
        if row['pose_cluster'] < 0:
            return 'Invalid'
        
        yaw, pitch, roll = row['HeadPoseYaw'], row['HeadPosePitch'], row['HeadPoseRoll']
        
        if pd.isna(yaw) or pd.isna(pitch) or pd.isna(roll):
            return 'Invalid'
        
        # Classify based on pose characteristics
        if abs(yaw) < 15 and abs(pitch) < 10 and abs(roll) < 10:
            return 'Frontal'
        elif yaw < -15:
            return 'Left Profile'
        elif yaw > 15:
            return 'Right Profile'
        elif pitch > 10:
            return 'Looking Up'
        elif pitch < -10:
            return 'Looking Down'
        else:
            return f'Mixed_{row["pose_cluster"]}'
    
    def balance_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance dataset across age bins, gender, and pose clusters
        """
        print(f"\n=== Demographic Balancing ===")
        
        # Check required columns - age is now standardized
        if 'age' not in df.columns:
            print("Error: No age column found for balancing")
            return df
        
        gender_cols = ['gender', 'Gender', 'sex', 'Sex']
        gender_col = None
        for col in gender_cols:
            if col in df.columns:
                gender_col = col
                break
        
        # Create age bins if not already created
        if 'age_bin' not in df.columns:
            age_bin_labels, _ = self.create_age_bins(df['age'])
            df['age_bin'] = age_bin_labels
        
        # Remove age bins with too few samples
        age_bin_counts = df['age_bin'].value_counts()
        valid_age_bins = age_bin_counts[age_bin_counts >= self.config['min_samples_per_age_bin']].index
        df_filtered = df[df['age_bin'].isin(valid_age_bins)]
        
        print(f"Kept {len(valid_age_bins)} age bins with >= {self.config['min_samples_per_age_bin']} samples")
        print(f"Removed {len(df) - len(df_filtered)} samples from small age bins")
        
        # Determine target samples per age bin
        target_samples = self.config['target_samples_per_age_bin']
        if target_samples is None:
            # Use minimum samples in any valid age bin
            target_samples = age_bin_counts[valid_age_bins].min()
        
        print(f"Target samples per age bin: {target_samples}")
        
        # Balance by age, gender, and pose
        balanced_dfs = []
        
        for age_bin in valid_age_bins:
            age_bin_data = df_filtered[df_filtered['age_bin'] == age_bin]
            
            if gender_col and self.config['balance_gender']:
                # Balance within each age bin by gender
                balanced_age_data = self._balance_by_gender_and_pose(
                    age_bin_data, gender_col, target_samples
                )
            else:
                # Balance only by pose within age bin
                balanced_age_data = self._balance_by_pose(age_bin_data, target_samples)
            
            if len(balanced_age_data) > 0:
                balanced_dfs.append(balanced_age_data)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        # Print balancing results
        print(f"\nBalancing Results:")
        print(f"Original samples: {len(df):,}")
        print(f"Balanced samples: {len(balanced_df):,}")
        print(f"Retention rate: {len(balanced_df)/len(df)*100:.1f}%")
        
        # Show age distribution
        final_age_dist = balanced_df['age_bin'].value_counts().sort_index()
        print(f"\nFinal age distribution:")
        for age_bin, count in final_age_dist.items():
            print(f"  {age_bin}: {count} samples")
        
        # Show gender distribution if available
        if gender_col:
            final_gender_dist = balanced_df[gender_col].value_counts()
            print(f"\nFinal gender distribution:")
            for gender, count in final_gender_dist.items():
                print(f"  {gender}: {count} samples ({count/len(balanced_df)*100:.1f}%)")
        
        # Show pose distribution if available
        if 'pose_cluster_name' in balanced_df.columns:
            final_pose_dist = balanced_df['pose_cluster_name'].value_counts()
            print(f"\nFinal pose distribution:")
            for pose, count in final_pose_dist.items():
                print(f"  {pose}: {count} samples ({count/len(balanced_df)*100:.1f}%)")
        
        return balanced_df
    
    def _balance_by_gender_and_pose(self, df: pd.DataFrame, gender_col: str, target_samples: int) -> pd.DataFrame:
        """
        Balance samples by gender and pose within an age bin
        """
        genders = df[gender_col].unique()
        target_per_gender = target_samples // len(genders)
        
        gender_dfs = []
        for gender in genders:
            gender_data = df[df[gender_col] == gender]
            
            if len(gender_data) >= target_per_gender:
                # Balance by pose within gender
                balanced_gender_data = self._balance_by_pose(gender_data, target_per_gender)
                gender_dfs.append(balanced_gender_data)
            else:
                # Use all available samples if not enough
                gender_dfs.append(gender_data)
        
        return pd.concat(gender_dfs, ignore_index=True) if gender_dfs else pd.DataFrame()
    
    def _balance_by_pose(self, df: pd.DataFrame, target_samples: int) -> pd.DataFrame:
        """
        Balance samples by pose clusters
        """
        if 'pose_cluster_name' not in df.columns or not self.config['balance_poses_within_age']:
            # Random sampling without pose balancing
            if len(df) >= target_samples:
                return df.sample(n=target_samples, random_state=42)
            else:
                return df
        
        pose_clusters = df['pose_cluster_name'].unique()
        valid_clusters = [cluster for cluster in pose_clusters if cluster != 'Invalid']
        
        if len(valid_clusters) == 0:
            return df.sample(n=min(len(df), target_samples), random_state=42)
        
        target_per_pose = target_samples // len(valid_clusters)
        
        pose_dfs = []
        for pose in valid_clusters:
            pose_data = df[df['pose_cluster_name'] == pose]
            
            if len(pose_data) >= target_per_pose:
                sampled_pose_data = pose_data.sample(n=target_per_pose, random_state=42)
            else:
                sampled_pose_data = pose_data
            
            pose_dfs.append(sampled_pose_data)
        
        return pd.concat(pose_dfs, ignore_index=True) if pose_dfs else pd.DataFrame()
    
    def _compare_demographic_distributions(self, before: Dict, after: Dict, stage: str):
        """
        Compare demographic distributions before and after processing
        """
        print(f"\n--- {stage} Impact ---")
        
        # Age distribution comparison
        if 'age_bins' in before and 'age_bins' in after:
            before_bins = before['age_bins']['distribution']
            after_bins = after['age_bins']['distribution']
            
            print("Age bin changes:")
            all_bins = set(before_bins.keys()) | set(after_bins.keys())
            for age_bin in sorted(all_bins):
                before_count = before_bins.get(age_bin, 0)
                after_count = after_bins.get(age_bin, 0)
                change = after_count - before_count
                if before_count > 0:
                    pct_change = (change / before_count) * 100
                    print(f"  {age_bin}: {before_count} → {after_count} ({change:+d}, {pct_change:+.1f}%)")
        
        # Gender distribution comparison
        if before.get('gender') and after.get('gender'):
            before_gender = before['gender']['distribution']
            after_gender = after['gender']['distribution']
            
            print("Gender changes:")
            all_genders = set(before_gender.keys()) | set(after_gender.keys())
            for gender in all_genders:
                before_count = before_gender.get(gender, 0)
                after_count = after_gender.get(gender, 0)
                change = after_count - before_count
                if before_count > 0:
                    pct_change = (change / before_count) * 100
                    print(f"  {gender}: {before_count} → {after_count} ({change:+d}, {pct_change:+.1f}%)")
    
    def visualize_balanced_dataset(self, df_original: pd.DataFrame, df_balanced: pd.DataFrame, 
                                 dataset_name: str = "Dataset") -> plt.Figure:
        """
        Create comprehensive visualizations of the balanced dataset
        """
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'{dataset_name} - Complete Processing Analysis', fontsize=16)
        
        # Age column is now standardized to 'age'
        age_col = 'age' if 'age' in df_balanced.columns else None
        gender_col = None
        for col in ['gender', 'Gender', 'sex', 'Sex']:
            if col in df_balanced.columns:
                gender_col = col
                break
        
        # 1. Age distribution comparison
        if age_col:
            axes[0, 0].hist(df_original[age_col].dropna(), bins=50, alpha=0.7, label='Original', density=True)
            axes[0, 0].hist(df_balanced[age_col].dropna(), bins=50, alpha=0.7, label='Balanced', density=True)
            axes[0, 0].set_xlabel('Age')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].set_title('Age Distribution')
            axes[0, 0].legend()
        
        # 2. Age bin distribution
        if 'age_bin' in df_balanced.columns:
            age_bin_counts = df_balanced['age_bin'].value_counts().sort_index()
            axes[0, 1].bar(range(len(age_bin_counts)), age_bin_counts.values)
            axes[0, 1].set_xlabel('Age Bin')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Balanced Age Bin Distribution')
            axes[0, 1].set_xticks(range(len(age_bin_counts)))
            axes[0, 1].set_xticklabels(age_bin_counts.index, rotation=45)
        
        # 3. Gender distribution (if available)
        if gender_col:
            gender_counts = df_balanced[gender_col].value_counts()
            axes[0, 2].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
            axes[0, 2].set_title('Gender Distribution')
        else:
            axes[0, 2].text(0.5, 0.5, 'No Gender\nInformation', ha='center', va='center', 
                           transform=axes[0, 2].transAxes, fontsize=12)
            axes[0, 2].set_title('Gender Distribution')
        
        # 4. Quality Score Distribution
        if 'UnifiedQualityScore' in df_balanced.columns:
            axes[1, 0].hist(df_original['UnifiedQualityScore'].dropna(), bins=50, alpha=0.7, 
                           label='Original', density=True)
            axes[1, 0].hist(df_balanced['UnifiedQualityScore'].dropna(), bins=50, alpha=0.7, 
                           label='Balanced', density=True)
            axes[1, 0].axvline(self.config['unified_quality_score_min'], color='red', 
                              linestyle='--', label='Threshold')
            axes[1, 0].set_xlabel('Quality Score')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].set_title('Quality Score Distribution')
            axes[1, 0].legend()
        
        # 5. Pose distribution
        pose_cols = ['HeadPoseYaw', 'HeadPosePitch', 'HeadPoseRoll']
        if all(col in df_balanced.columns for col in pose_cols):
            scatter = axes[1, 1].scatter(df_balanced['HeadPoseYaw'], df_balanced['HeadPosePitch'], 
                                       c=df_balanced.get('pose_cluster', 0), alpha=0.6, cmap='viridis')
            axes[1, 1].set_xlabel('Head Pose Yaw')
            axes[1, 1].set_ylabel('Head Pose Pitch')
            axes[1, 1].set_title('Pose Distribution')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Pose cluster distribution
        if 'pose_cluster_name' in df_balanced.columns:
            pose_counts = df_balanced['pose_cluster_name'].value_counts()
            axes[1, 2].bar(pose_counts.index, pose_counts.values)
            axes[1, 2].set_xlabel('Pose Cluster')
            axes[1, 2].set_ylabel('Count')
            axes[1, 2].set_title('Pose Cluster Distribution')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        # 7. Age vs Quality relationship
        if age_col and 'UnifiedQualityScore' in df_balanced.columns:
            axes[2, 0].scatter(df_balanced[age_col], df_balanced['UnifiedQualityScore'], alpha=0.6)
            axes[2, 0].set_xlabel('Age')
            axes[2, 0].set_ylabel('Quality Score')
            axes[2, 0].set_title('Age vs Quality Relationship')
            axes[2, 0].grid(True, alpha=0.3)
        
        # 8. Age vs Pose relationship
        if age_col and 'HeadPoseYaw' in df_balanced.columns:
            axes[2, 1].scatter(df_balanced[age_col], np.abs(df_balanced['HeadPoseYaw']), alpha=0.6)
            axes[2, 1].set_xlabel('Age')
            axes[2, 1].set_ylabel('Absolute Yaw Angle')
            axes[2, 1].set_title('Age vs Pose Relationship')
            axes[2, 1].grid(True, alpha=0.3)
        
        # 9. Sample count by age and gender (if available)
        if age_col and gender_col and 'age_bin' in df_balanced.columns:
            crosstab = pd.crosstab(df_balanced['age_bin'], df_balanced[gender_col])
            im = axes[2, 2].imshow(crosstab.values, cmap='Blues', aspect='auto')
            axes[2, 2].set_xticks(range(len(crosstab.columns)))
            axes[2, 2].set_xticklabels(crosstab.columns)
            axes[2, 2].set_yticks(range(len(crosstab.index)))
            axes[2, 2].set_yticklabels(crosstab.index, rotation=0)
            axes[2, 2].set_xlabel('Gender')
            axes[2, 2].set_ylabel('Age Bin')
            axes[2, 2].set_title('Samples by Age and Gender')
            
            # Add text annotations
            for i in range(len(crosstab.index)):
                for j in range(len(crosstab.columns)):
                    axes[2, 2].text(j, i, str(crosstab.values[i, j]), 
                                   ha='center', va='center', color='white' if crosstab.values[i, j] > crosstab.values.max()/2 else 'black')
        else:
            axes[2, 2].text(0.5, 0.5, 'Age-Gender\nCrosstab\nNot Available', 
                           ha='center', va='center', transform=axes[2, 2].transAxes, fontsize=12)
            axes[2, 2].set_title('Age-Gender Distribution')
        
        plt.tight_layout()
        return fig
    
    def generate_comprehensive_report(self, dataset_name: str, df_original: pd.DataFrame, 
                                    df_balanced: pd.DataFrame) -> Dict:
        """
        Generate comprehensive processing report
        """
        report = {
            'dataset_name': dataset_name,
            'processing_timestamp': pd.Timestamp.now().isoformat(),
            'sample_counts': {
                'original': len(df_original),
                'balanced': len(df_balanced),
                'retention_rate': len(df_balanced) / len(df_original) * 100
            },
            'demographics': {},
            'quality_metrics': {},
            'pose_metrics': {},
            'balancing_effectiveness': {},
            'recommendations': []
        }
        
        # Demographic analysis - age is now standardized
        if 'age' in df_balanced.columns:
            ages = df_balanced['age'].dropna()
            report['demographics']['age'] = {
                'count': len(ages),
                'mean': float(ages.mean()),
                'median': float(ages.median()),
                'std': float(ages.std()),
                'range': [float(ages.min()), float(ages.max())],
                'distribution': ages.value_counts().sort_index().to_dict()
            }
            
            if 'age_bin' in df_balanced.columns:
                age_bins = df_balanced['age_bin'].value_counts().sort_index()
                report['demographics']['age_bins'] = {
                    'count': len(age_bins),
                    'distribution': age_bins.to_dict(),
                    'uniformity_score': float(1 - age_bins.std() / age_bins.mean()),  # Higher is more uniform
                    'min_samples': int(age_bins.min()),
                    'max_samples': int(age_bins.max())
                }
        
        # Gender analysis
        gender_col = None
        for col in ['gender', 'Gender', 'sex', 'Sex']:
            if col in df_balanced.columns:
                gender_col = col
                break
        
        if gender_col:
            genders = df_balanced[gender_col].dropna()
            gender_dist = genders.value_counts()
            report['demographics']['gender'] = {
                'count': len(genders),
                'distribution': gender_dist.to_dict(),
                'balance_score': float(gender_dist.min() / gender_dist.max()),  # Closer to 1 is more balanced
                'unique_values': genders.unique().tolist()
            }
        
        # Quality metrics analysis
        quality_cols = ['UnifiedQualityScore', 'Sharpness', 'ExpressionNeutrality', 'IlluminationUniformity']
        for col in quality_cols:
            if col in df_balanced.columns:
                values = df_balanced[col].dropna()
                report['quality_metrics'][col] = {
                    'mean': float(values.mean()),
                    'median': float(values.median()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'above_threshold': float((values >= self.config.get(f'{col.lower()}_min', 0)).mean() * 100)
                }
        
        # Pose metrics analysis
        pose_cols = ['HeadPoseYaw', 'HeadPosePitch', 'HeadPoseRoll']
        for col in pose_cols:
            if col in df_balanced.columns:
                values = df_balanced[col].dropna()
                report['pose_metrics'][col] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'abs_mean': float(np.abs(values).mean()),
                    'frontal_percentage': float((np.abs(values) <= 15).mean() * 100)
                }
        
        if 'pose_cluster_name' in df_balanced.columns:
            pose_clusters = df_balanced['pose_cluster_name'].value_counts()
            report['pose_metrics']['cluster_distribution'] = pose_clusters.to_dict()
            report['pose_metrics']['cluster_balance_score'] = float(pose_clusters.min() / pose_clusters.max())
        
        # Balancing effectiveness
        if 'age_bin' in df_balanced.columns:
            age_bins = df_balanced['age_bin'].value_counts()
            cv = age_bins.std() / age_bins.mean()  # Coefficient of variation
            report['balancing_effectiveness']['age_uniformity'] = {
                'coefficient_of_variation': float(cv),
                'uniformity_grade': 'Excellent' if cv < 0.1 else 'Good' if cv < 0.2 else 'Fair' if cv < 0.3 else 'Poor'
            }
        
        if gender_col:
            gender_counts = df_balanced[gender_col].value_counts()
            balance_ratio = gender_counts.min() / gender_counts.max()
            report['balancing_effectiveness']['gender_balance'] = {
                'balance_ratio': float(balance_ratio),
                'balance_grade': 'Excellent' if balance_ratio > 0.9 else 'Good' if balance_ratio > 0.8 else 'Fair' if balance_ratio > 0.7 else 'Poor'
            }
        
        # Generate recommendations
        retention_rate = report['sample_counts']['retention_rate']
        if retention_rate < 20:
            report['recommendations'].append("Very low retention rate - consider relaxing quality thresholds")
        elif retention_rate < 40:
            report['recommendations'].append("Low retention rate - review filtering criteria")
        
        if 'age_uniformity' in report['balancing_effectiveness']:
            if report['balancing_effectiveness']['age_uniformity']['uniformity_grade'] in ['Fair', 'Poor']:
                report['recommendations'].append("Age distribution not well balanced - consider increasing target samples per bin")
        
        if 'gender_balance' in report['balancing_effectiveness']:
            if report['balancing_effectiveness']['gender_balance']['balance_grade'] in ['Fair', 'Poor']:
                report['recommendations'].append("Gender distribution imbalanced - review gender balancing strategy")
        
        if 'cluster_balance_score' in report['pose_metrics']:
            if report['pose_metrics']['cluster_balance_score'] < 0.5:
                report['recommendations'].append("Pose clusters are imbalanced - consider pose-specific sampling")
        
        return report
    
    def copy_images_and_create_csvs(self, df_balanced: pd.DataFrame, dataset_name: str, 
                                   output_dir: str) -> Dict[str, str]:
        """
        Copy selected images to dataset subdirectory and create updated CSVs with new paths
        """
        # Create dataset subdirectory
        dataset_dir = os.path.abspath(os.path.join(output_dir, dataset_name))
        images_dir = os.path.join(dataset_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        print(f"\n=== Copying Images for {dataset_name} ===")
        
        # Copy images and update paths
        new_paths = []
        failed_copies = []
        
        for idx, row in tqdm(df_balanced.iterrows(), total=len(df_balanced), desc="Copying images"):
            original_path = row['Filename']
            
            if pd.isna(original_path) or not os.path.exists(original_path):
                failed_copies.append(original_path)
                continue
            
            # Create new filename
            original_filename = os.path.basename(original_path)
            new_path = os.path.join(images_dir, original_filename)
            
            # Handle duplicate filenames
            counter = 1
            base_name, ext = os.path.splitext(original_filename)
            while os.path.exists(new_path):
                new_filename = f"{base_name}_{counter}{ext}"
                new_path = os.path.join(images_dir, new_filename)
                counter += 1
            
            try:
                shutil.copy2(original_path, new_path)
                new_paths.append(new_path)
            except Exception as e:
                print(f"Failed to copy {original_path}: {str(e)}")
                failed_copies.append(original_path)
                new_paths.append(None)
        
        print(f"Successfully copied {len([p for p in new_paths if p is not None])}/{len(df_balanced)} images")
        if failed_copies:
            print(f"Failed to copy {len(failed_copies)} images")
        
        # Update dataframe with new paths
        df_balanced_copy = df_balanced.copy()
        df_balanced_copy['new_filename'] = new_paths
        
        # Remove rows where image copy failed
        df_balanced_copy = df_balanced_copy[df_balanced_copy['new_filename'].notna()]
        
        # Create OFIQ CSV with new paths
        ofiq_columns = [col for col in df_balanced_copy.columns if col not in ['age', 'gender', 'Gender', 'sex', 'Sex']]
        df_ofiq = df_balanced_copy[ofiq_columns].copy()
        df_ofiq['Filename'] = df_balanced_copy['new_filename']  # Update to new paths
        
        # Create groundtruth CSV with new paths
        groundtruth_columns = ['new_filename', 'age']
        
        # Add gender column if available
        gender_col = None
        for col in ['gender', 'Gender', 'sex', 'Sex']:
            if col in df_balanced_copy.columns:
                gender_col = col
                groundtruth_columns.append(col)
                break
        
        df_groundtruth = df_balanced_copy[groundtruth_columns].copy()
        df_groundtruth = df_groundtruth.rename(columns={'new_filename': 'Filename'})
        
        # Save CSVs
        # ofiq_csv_path = os.path.join(dataset_dir, f"{dataset_name}_ofiq_balanced.csv")
        # groundtruth_csv_path = os.path.join(dataset_dir, f"{dataset_name}_groundtruth_balanced.csv")


        ofiq_csv_path = os.path.join(dataset_dir, f"ofiq_balanced.csv")
        groundtruth_csv_path = os.path.join(dataset_dir, f"groundtruth_balanced.csv")
        

        df_ofiq.to_csv(ofiq_csv_path, index=False)
        df_groundtruth.to_csv(groundtruth_csv_path, index=False)
        
        print(f"Created OFIQ CSV: {os.path.basename(ofiq_csv_path)} ({len(df_ofiq)} samples)")
        print(f"Created Groundtruth CSV: {os.path.basename(groundtruth_csv_path)} ({len(df_groundtruth)} samples)")
        
        return {
            'dataset_dir': dataset_dir,
            'images_dir': images_dir,
            'ofiq_csv': ofiq_csv_path,
            'groundtruth_csv': groundtruth_csv_path,
            'images_copied': len([p for p in new_paths if p is not None]),
            'images_failed': len(failed_copies)
        }
    
    def save_results(self, df_balanced: pd.DataFrame, report: Dict, output_dir: str, dataset_name: str):
        """
        Save all results to organized dataset subdirectory
        """
        # Create dataset subdirectory structure
        dataset_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Copy images and create updated CSVs
        copy_results = self.copy_images_and_create_csvs(df_balanced, dataset_name, output_dir)
        
        # Save comprehensive report
        report_path = os.path.join(dataset_dir, f"{dataset_name}_processing_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save processing statistics
        stats_path = os.path.join(dataset_dir, f"{dataset_name}_statistics.txt")
        with open(stats_path, 'w') as f:
            f.write(f"Dataset Processing Statistics: {dataset_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Original samples: {report['sample_counts']['original']:,}\n")
            f.write(f"Balanced samples: {report['sample_counts']['balanced']:,}\n")
            f.write(f"Retention rate: {report['sample_counts']['retention_rate']:.1f}%\n")
            f.write(f"Images copied: {copy_results['images_copied']:,}\n")
            f.write(f"Copy failures: {copy_results['images_failed']:,}\n\n")
            
            if 'age_bins' in report['demographics']:
                f.write("Age Distribution:\n")
                for age_bin, count in report['demographics']['age_bins']['distribution'].items():
                    f.write(f"  {age_bin}: {count} samples\n")
                f.write("\n")
            
            if report.get('recommendations'):
                f.write("Recommendations:\n")
                for rec in report['recommendations']:
                    f.write(f"  - {rec}\n")
        
        # Create README file
        readme_path = os.path.join(dataset_dir, "README.md")
        with open(readme_path, 'w') as f:
            f.write(f"# {dataset_name} - Balanced Age Estimation Dataset\n\n")
            f.write("## Dataset Structure\n\n")
            f.write("```\n")
            f.write(f"{dataset_name}/\n")
            f.write("├── images/                     # Balanced image set\n")
            f.write(f"├── {dataset_name}_ofiq_balanced.csv      # OFIQ quality metrics\n")
            f.write(f"├── {dataset_name}_groundtruth_balanced.csv # Age and gender labels\n")
            f.write(f"├── {dataset_name}_processing_report.json  # Detailed processing report\n")
            f.write(f"├── {dataset_name}_statistics.txt          # Processing statistics\n")
            f.write(f"├── {dataset_name}_analysis.png            # Visualization plots\n")
            f.write("└── README.md                   # This file\n")
            f.write("```\n\n")
            f.write("## Dataset Statistics\n\n")
            f.write(f"- **Total Images**: {report['sample_counts']['balanced']:,}\n")
            f.write(f"- **Retention Rate**: {report['sample_counts']['retention_rate']:.1f}%\n")
            
            if 'age' in report['demographics']:
                f.write(f"- **Age Range**: {report['demographics']['age']['range'][0]:.0f}-{report['demographics']['age']['range'][1]:.0f} years\n")
                f.write(f"- **Mean Age**: {report['demographics']['age']['mean']:.1f} years\n")
            
            if 'age_bins' in report['demographics']:
                f.write(f"- **Age Bins**: {report['demographics']['age_bins']['count']}\n")
                f.write(f"- **Samples per Bin**: {report['demographics']['age_bins']['min_samples']}-{report['demographics']['age_bins']['max_samples']}\n")
            
            if 'gender' in report['demographics'] and report['demographics']['gender']:
                f.write(f"- **Gender Balance**: {len(report['demographics']['gender']['unique_values'])} categories\n")
            
            f.write("\n## Quality Metrics\n\n")
            if 'UnifiedQualityScore' in report['quality_metrics']:
                f.write(f"- **Mean Quality Score**: {report['quality_metrics']['UnifiedQualityScore']['mean']:.1f}\n")
                f.write(f"- **Quality Range**: {report['quality_metrics']['UnifiedQualityScore']['min']:.1f}-{report['quality_metrics']['UnifiedQualityScore']['max']:.1f}\n")
            
            f.write("\n## Usage\n\n")
            f.write("```python\n")
            f.write("import pandas as pd\n\n")
            f.write("# Load balanced dataset\n")
            f.write(f"ofiq_data = pd.read_csv('{dataset_name}_ofiq_balanced.csv')\n")
            f.write(f"groundtruth_data = pd.read_csv('{dataset_name}_groundtruth_balanced.csv')\n\n")
            f.write("# Get image paths and labels\n")
            f.write("image_paths = groundtruth_data['Filename'].values\n")
            f.write("ages = groundtruth_data['age'].values\n")
            f.write("```\n")
        
        return {
            'dataset_dir': dataset_dir,
            'ofiq_csv': copy_results['ofiq_csv'],
            'groundtruth_csv': copy_results['groundtruth_csv'],
            'images_dir': copy_results['images_dir'],
            'report': report_path,
            'statistics': stats_path,
            'readme': readme_path,
            'images_copied': copy_results['images_copied'],
            'images_failed': copy_results['images_failed']
        }

def process_age_estimation_dataset(csv_path: str, groundtruth_path: str, dataset_name: str,
                                  output_dir: str = "balanced_datasets", config: Dict = None) -> Dict:
    """
    Complete processing pipeline for age estimation datasets
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing {dataset_name} for Age Estimation")
    print(f"{'='*60}")
    
    # Initialize processor
    processor = BalancedAgeDatasetProcessor(config)
    
    # Load and merge datasets
    print("1. Loading and merging datasets...")
    df_original = processor.load_dataset(csv_path, groundtruth_path)
    print(f"Loaded {len(df_original):,} samples")
    
    # Analyze initial demographics
    print("\n2. Analyzing initial demographics...")
    initial_demo = processor.analyze_demographic_distribution(df_original)
    
    # Apply quality filters
    print("\n3. Applying quality filters...")
    df_filtered = processor.apply_quality_filters(df_original, dataset_name)
    
    # Cluster poses
    print("\n4. Clustering poses...")
    df_clustered = processor.cluster_poses(df_filtered)
    
    # Balance demographics
    print("\n5. Balancing demographics...")
    df_balanced = processor.balance_demographics(df_clustered)
    
    # Generate visualizations
    print("\n6. Generating visualizations...")
    fig = processor.visualize_balanced_dataset(df_original, df_balanced, dataset_name)
    
    # Generate comprehensive report
    print("\n7. Generating report...")
    report = processor.generate_comprehensive_report(dataset_name, df_original, df_balanced)
    
    # Save results
    print("\n8. Saving results...")
    output_paths = processor.save_results(df_balanced, report, output_dir, dataset_name)
    
    # Save visualization in dataset directory
    fig_path = os.path.join(output_paths['dataset_dir'], f"{dataset_name}_analysis.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    output_paths['visualization'] = fig_path
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"Processing Complete: {dataset_name}")
    print(f"{'='*60}")
    print(f"Original samples: {len(df_original):,}")
    print(f"Balanced samples: {len(df_balanced):,}")
    print(f"Retention rate: {len(df_balanced)/len(df_original)*100:.1f}%")
    print(f"Images copied: {output_paths['images_copied']:,}")
    if output_paths['images_failed'] > 0:
        print(f"Copy failures: {output_paths['images_failed']:,}")
    
    if report['recommendations']:
        print(f"\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
    
    print(f"\nDataset directory: {output_paths['dataset_dir']}")
    print(f"  - Images: images/ ({output_paths['images_copied']:,} files)")
    print(f"  - OFIQ CSV: {os.path.basename(output_paths['ofiq_csv'])}")
    print(f"  - Groundtruth CSV: {os.path.basename(output_paths['groundtruth_csv'])}")
    print(f"  - Analysis plot: {os.path.basename(output_paths['visualization'])}")
    
    return {
        'dataset_name': dataset_name,
        'balanced_dataframe': df_balanced,
        'report': report,
        'output_paths': output_paths,
        'processor': processor
    }

def process_multiple_datasets(dataset_configs: List[Dict], output_dir: str = "balanced_datasets") -> Dict:
    """
    Process multiple datasets for age estimation
    
    Args:
        dataset_configs: List of dicts with keys: 'csv_path', 'groundtruth_path', 'dataset_name', 'config'
        output_dir: Output directory for all results
    """
    all_results = {}
    summary_stats = []
    
    for i, dataset_config in enumerate(dataset_configs):
        print(f"\n\nProcessing dataset {i+1}/{len(dataset_configs)}: {dataset_config['dataset_name']}")
        
        try:
            result = process_age_estimation_dataset(
                csv_path=dataset_config['csv_path'],
                groundtruth_path=dataset_config['groundtruth_path'],
                dataset_name=dataset_config['dataset_name'],
                output_dir=output_dir,
                config=dataset_config.get('config')
            )
            
            all_results[dataset_config['dataset_name']] = result
            
            # Collect summary statistics
            summary_stats.append({
                'dataset': dataset_config['dataset_name'],
                'original_samples': len(result['balanced_dataframe']) + 1000,  # Approximate
                'balanced_samples': len(result['balanced_dataframe']),
                'retention_rate': result['report']['sample_counts']['retention_rate'],
                'age_bins': len(result['report']['demographics'].get('age_bins', {}).get('distribution', {})),
                'quality_score': result['report']['quality_metrics'].get('UnifiedQualityScore', {}).get('mean', 0)
            })
            
        except Exception as e:
            print(f"Error processing {dataset_config['dataset_name']}: {str(e)}")
            continue
    
    # Create summary report
    summary_df = pd.DataFrame(summary_stats)
    summary_path = os.path.join(output_dir, "processing_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # Create combined visualization
    if len(summary_stats) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Multi-Dataset Processing Summary', fontsize=16)
        
        # Retention rates
        axes[0, 0].bar(summary_df['dataset'], summary_df['retention_rate'])
        axes[0, 0].set_ylabel('Retention Rate (%)')
        axes[0, 0].set_title('Retention Rate by Dataset')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Sample counts
        axes[0, 1].bar(summary_df['dataset'], summary_df['balanced_samples'])
        axes[0, 1].set_ylabel('Balanced Samples')
        axes[0, 1].set_title('Final Sample Count by Dataset')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Age bins
        axes[1, 0].bar(summary_df['dataset'], summary_df['age_bins'])
        axes[1, 0].set_ylabel('Number of Age Bins')
        axes[1, 0].set_title('Age Bins Retained by Dataset')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Quality scores
        axes[1, 1].bar(summary_df['dataset'], summary_df['quality_score'])
        axes[1, 1].set_ylabel('Mean Quality Score')
        axes[1, 1].set_title('Final Quality Score by Dataset')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        summary_viz_path = os.path.join(output_dir, "multi_dataset_summary.png")
        fig.savefig(summary_viz_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"\n\n{'='*80}")
    print(f"ALL DATASETS PROCESSED")
    print(f"{'='*80}")
    print(f"Successfully processed: {len(all_results)}/{len(dataset_configs)} datasets")
    print(f"Summary saved to: {summary_path}")
    
    return {
        'results': all_results,
        'summary': summary_df,
        'summary_path': summary_path
    }

# Configuration presets for different dataset types
DATASET_QUALITY_CONFIGS = {
    'high_quality': {  # For MORPH, AgeDB - but still relaxed
        'unified_quality_score_min': 25,           # Relaxed from 40
        'expression_neutrality_min': 30,           # Relaxed from 70
        'expression_neutrality_relaxed': 20,       # Relaxed from 50
        'sharpness_min': 40,                       # Relaxed from 60
        'illumination_uniformity_min': 50,         # Relaxed from 70
        'extreme_pose_max': 60,                    # Relaxed from 45
        'frontal_yaw_max': 25,                     # Relaxed from 15
        'frontal_pitch_max': 20,                   # Relaxed from 10
        'frontal_roll_max': 20,                    # Relaxed from 10
        'moderate_pose_max': 45,                   # Relaxed from 30
        'age_bin_size': 5,
        'min_samples_per_age_bin': 20,             # Reduced from 50
        'target_samples_per_age_bin': 100,         # Reduced from 200
        'balance_gender': True,
        'gender_ratio_tolerance': 0.2,
        'pose_clusters': 15,
        'balance_poses_within_age': True,
        'adaptive_thresholds': True,
        'preserve_demographic_balance': True,
    },
    
    'moderate_quality': {  # For APPA-REAL, IMDB-WIKI, etc.
        'unified_quality_score_min': 20,           # Very relaxed
        'expression_neutrality_min': 20,           # Very relaxed
        'expression_neutrality_relaxed': 15,       # Very relaxed
        'sharpness_min': 25,                       # Very relaxed
        'illumination_uniformity_min': 30,         # Very relaxed
        'extreme_pose_max': 75,                    # Very relaxed
        'frontal_yaw_max': 35,
        'frontal_pitch_max': 25,
        'frontal_roll_max': 25,
        'moderate_pose_max': 50,
        'age_bin_size': 5,
        'min_samples_per_age_bin': 15,
        'target_samples_per_age_bin': 80,
        'balance_gender': True,
        'gender_ratio_tolerance': 0.3,
        'pose_clusters': 15,
        'balance_poses_within_age': True,
        'adaptive_thresholds': True,
        'preserve_demographic_balance': True,
    },
    
    'low_quality': {  # For Adience, UTKFace, FG-NET
        'unified_quality_score_min': 15,           # Extremely relaxed
        'expression_neutrality_min': 10,           # Extremely relaxed
        'expression_neutrality_relaxed': 5,        # Extremely relaxed
        'sharpness_min': 10,                       # Extremely relaxed
        'illumination_uniformity_min': 20,         # Extremely relaxed
        'extreme_pose_max': 90,                    # Almost no pose filtering
        'frontal_yaw_max': 45,
        'frontal_pitch_max': 35,
        'frontal_roll_max': 35,
        'moderate_pose_max': 60,
        'age_bin_size': 10,                        # Larger age bins
        'min_samples_per_age_bin': 3,              # low threshold
        'target_samples_per_age_bin': 100,
        'balance_gender': True,
        'gender_ratio_tolerance': 0.3,             # Very tolerant
        'pose_clusters': 15,
        'balance_poses_within_age': True,
        'adaptive_thresholds': True,
        'preserve_demographic_balance': True,
    }
}
# Example: Process a single dataset
# result = process_age_estimation_dataset(
#     csv_path="appa_real_quality.csv",
#     groundtruth_path="appa_real_groundtruth.csv", 
#     dataset_name="APPA-REAL",
#     config=DATASET_QUALITY_CONFIGS['moderate_quality']
# )


# In[4]:


def main():
    """Main function to process all age estimation datasets"""
    
    # Define all 12 datasets with their configurations
    dataset_configs = [
        {
            'csv_path': '/home/meem/backup/Age Datasets/AdienceGender Dataset/ofiq.csv',
            'groundtruth_path': '/home/meem/backup/Age Datasets/AdienceGender Dataset/groundtruth.csv',
            'dataset_name': 'Adience',
            'config': DATASET_QUALITY_CONFIGS['low_quality']
        },
        {
            'csv_path': '/home/meem/backup/Age Datasets/fairface/ofiq.csv',
            'groundtruth_path': '/home/meem/backup/Age Datasets/fairface/groundtruth.csv',
            'dataset_name': 'FairFace',
            'config': DATASET_QUALITY_CONFIGS['low_quality']
        },
        {
            'csv_path': '/home/meem/backup/Age Datasets/afad/ofiq.csv',
            'groundtruth_path': '/home/meem/backup/Age Datasets/afad/groundtruth.csv',
            'dataset_name': 'AFAD',
            'config': DATASET_QUALITY_CONFIGS['low_quality']
        },
        {
            'csv_path': '/home/meem/backup/Age Datasets/agedb/ofiq.csv',
            'groundtruth_path': '/home/meem/backup/Age Datasets/agedb/groundtruth.csv',
            'dataset_name': 'AgeDB',
            'config': DATASET_QUALITY_CONFIGS['low_quality']
        },
        {
            'csv_path': '/home/meem/backup/Age Datasets/appa-real-release/ofiq.csv',
            'groundtruth_path': '/home/meem/backup/Age Datasets/appa-real-release/groundtruth.csv',
            'dataset_name': 'APPA-REAL',
            'config': DATASET_QUALITY_CONFIGS['low_quality']
        },
        {
            'csv_path': '/home/meem/backup/Age Datasets/FGNET Dataset/ofiq.csv',
            'groundtruth_path': '/home/meem/backup/Age Datasets/FGNET Dataset/groundtruth.csv',
            'dataset_name': 'FG-NET',
            'config': DATASET_QUALITY_CONFIGS['low_quality']
        },
        {
            'csv_path': '/home/meem/backup/Age Datasets/Groups-of-People Dataset/ofiq.csv',
            'groundtruth_path': '/home/meem/backup/Age Datasets/Groups-of-People Dataset/groundtruth.csv',
            'dataset_name': 'Groups',
            'config': DATASET_QUALITY_CONFIGS['low_quality']
        },
        {
            'csv_path': '/home/meem/backup/Age Datasets/IMDB - WIKI/ofiq.csv',
            'groundtruth_path': '/home/meem/backup/Age Datasets/IMDB - WIKI/groundtruth.csv',
            'dataset_name': 'IMDB-WIKI',
            'config': DATASET_QUALITY_CONFIGS['low_quality']
        },
        {
            'csv_path': '/home/meem/backup/Age Datasets/Juvenile_Dataset/Juvenile_Dataset/ofiq.csv',
            'groundtruth_path': '/home/meem/backup/Age Datasets/Juvenile_Dataset/Juvenile_Dataset/groundtruth.csv',
            'dataset_name': 'Juvenile-80K',
            'config': DATASET_QUALITY_CONFIGS['low_quality']
        },
        {
            'csv_path': '/home/meem/backup/Age Datasets/lagenda/ofiq.csv',
            'groundtruth_path': '/home/meem/backup/Age Datasets/lagenda/groundtruth.csv',
            'dataset_name': 'LAGENDA',
            'config': DATASET_QUALITY_CONFIGS['low_quality']
        },
        {
            'csv_path': '/home/meem/backup/Age Datasets/Morph2 Dataset/ofiq.csv',
            'groundtruth_path': '/home/meem/backup/Age Datasets/Morph2 Dataset/groundtruth.csv',
            'dataset_name': 'MORPH',
            'config': DATASET_QUALITY_CONFIGS['low_quality']
        },
        {
            'csv_path': '/home/meem/backup/Age Datasets/utkface-cropped/ofiq.csv',
            'groundtruth_path': '/home/meem/backup/Age Datasets/utkface-cropped/groundtruth.csv',
            'dataset_name': 'UTKFace',
            'config': DATASET_QUALITY_CONFIGS['low_quality']
        }
    ]
    
    # Output directory for all balanced datasets
    output_dir = 'balanced_age_datasets'
    
    # Process all datasets
    print("Starting processing of all 12 age estimation datasets...")
    results = process_multiple_datasets(dataset_configs, output_dir)
    
    print(f"\n🎉 All datasets processed successfully!")
    print(f"Results saved in: {output_dir}/")
    print(f"Successfully processed: {len(results['results'])}/12 datasets")

if __name__ == "__main__":
    main()

