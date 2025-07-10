#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
from typing import Dict, List, Tuple


# In[8]:


def create_unified_dataset(balanced_datasets_dir: str, output_dir: str = "unified_age_dataset", 
                           exclude_datasets: List[str] = None, 
                          train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15) -> Dict:
    """
    Merge all 12 balanced datasets into unified train/val/test with simple CSV format
    
    Args:
        balanced_datasets_dir: Directory containing all balanced dataset folders
        output_dir: Output directory for unified dataset
        train_ratio: Fraction of data for training (default 0.7)
        val_ratio: Fraction of data for validation (default 0.15)
        test_ratio: Fraction of data for testing (default 0.15)
    
    Returns:
        Dictionary with paths to created files and directories
    """

    os.makedirs(output_dir, exist_ok=True)
    if exclude_datasets is None:
        exclude_datasets = []
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    # Create output structure
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val') 
    test_dir = os.path.join(output_dir, 'test')
    
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"Creating unified dataset in: {output_dir}")
    print(f"Split ratios - Train: {train_ratio:.1%}, Val: {val_ratio:.1%}, Test: {test_ratio:.1%}")
    
    # Collect all datasets from groundtruth.csv files
    all_data = []
    dataset_names = []
    dataset_stats = {}
    
    print("\nCollecting data from all balanced datasets...")
    for dataset_name in sorted(os.listdir(balanced_datasets_dir)):
        dataset_path = os.path.join(balanced_datasets_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        if exclude_datasets and dataset_name in exclude_datasets:
            print(f"⏭️  Skipping excluded dataset: {dataset_name}")
            continue
            
        # Look for groundtruth CSV
        groundtruth_csv = os.path.join(dataset_path, f"groundtruth_balanced.csv")
        if not os.path.exists(groundtruth_csv):
            print(f"⚠️  Warning: {groundtruth_csv} not found, skipping {dataset_name}")
            continue
            
        # Load and validate data
        try:
            df = pd.read_csv(groundtruth_csv)
            
            # Validate required columns
            if 'Filename' not in df.columns or 'age' not in df.columns:
                print(f"⚠️  Warning: Missing required columns in {dataset_name}, skipping")
                continue
            
            # Add dataset source for tracking
            df['dataset_source'] = dataset_name
            
            # Store dataset statistics
            dataset_stats[dataset_name] = {
                'samples': len(df),
                'age_range': [int(df['age'].min()), int(df['age'].max())],
                'mean_age': float(df['age'].mean())
            }
            
            all_data.append(df)
            dataset_names.append(dataset_name)
            print(f"  ✅ {dataset_name}: {len(df):,} samples (age: {df['age'].min():.0f}-{df['age'].max():.0f})")
            
        except Exception as e:
            print(f"⚠️  Error loading {dataset_name}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid datasets found!")
    
    # Combine all datasets
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Total combined samples: {len(combined_df):,}")
    print(f"   Age range: {combined_df['age'].min():.0f}-{combined_df['age'].max():.0f} years")
    print(f"   Mean age: {combined_df['age'].mean():.1f} years")
    
    # Check for gender information
    has_gender = any(col in combined_df.columns for col in ['gender', 'Gender', 'sex', 'Sex'])
    if has_gender:
        gender_col = next(col for col in ['gender', 'Gender', 'sex', 'Sex'] if col in combined_df.columns)
        gender_dist = combined_df[gender_col].value_counts()
        print(f"   Gender distribution: {dict(gender_dist)}")
    
    def create_stratified_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create stratified train/val/test splits maintaining age distribution"""
        
        # Create age groups for stratification (ensure each group has enough samples)
        n_groups = min(10, len(df) // 100)  # At least 100 samples per group
        n_groups = max(3, n_groups)  # At least 3 groups
        
        df = df.copy()
        df['age_group'] = pd.cut(df['age'], bins=n_groups, labels=False)
        
        print(f"\n🎯 Creating stratified splits using {n_groups} age groups...")
        
        # First split: train vs (val+test)
        train_df, temp_df = train_test_split(
            df, 
            test_size=(val_ratio + test_ratio),
            stratify=df['age_group'],
            random_state=42
        )
        
        # Second split: val vs test
        if test_ratio > 0:
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            val_df, test_df = train_test_split(
                temp_df,
                test_size=(1 - val_ratio_adjusted),
                stratify=temp_df['age_group'],
                random_state=42
            )
        else:
            val_df = temp_df
            test_df = pd.DataFrame()
        
        return train_df, val_df, test_df
    
    # Create splits
    train_df, val_df, test_df = create_stratified_splits(combined_df)
    
    print(f"   Train: {len(train_df):,} samples ({len(train_df)/len(combined_df)*100:.1f}%)")
    print(f"   Val: {len(val_df):,} samples ({len(val_df)/len(combined_df)*100:.1f}%)")
    if len(test_df) > 0:
        print(f"   Test: {len(test_df):,} samples ({len(test_df)/len(combined_df)*100:.1f}%)")
    
    def process_split(df: pd.DataFrame, split_name: str, target_dir: str) -> List[Dict]:
        """Copy images and create annotation list for a data split"""
        
        print(f"\n📁 Processing {split_name} split...")
        
        annotations = []
        failed_copies = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying {split_name} images"):
            original_path = row['Filename']
            age = row['age']
            dataset_source = row['dataset_source']
            
            if pd.isna(original_path) or not os.path.exists(original_path):
                failed_copies += 1
                continue
            
            # Create new filename with dataset prefix to avoid conflicts between datasets
            original_filename = os.path.basename(original_path)
            new_filename = f"{dataset_source}_{original_filename}"
            new_path = os.path.join(target_dir, new_filename)
            
            # Handle filename conflicts within the same split
            counter = 1
            base_name, ext = os.path.splitext(new_filename)
            while os.path.exists(new_path):
                new_filename = f"{base_name}_dup{counter}{ext}"
                new_path = os.path.join(target_dir, new_filename)
                counter += 1
            
            # Copy image
            try:
                shutil.copy2(original_path, new_path)
                
                # Store annotation with just filename (since we provide train_dir/val_dir separately)
                annotations.append({
                    'img_path': new_filename,
                    'age': int(age)
                })
                
            except Exception as e:
                print(f"Failed to copy {original_path}: {e}")
                failed_copies += 1
                continue
        
        if failed_copies > 0:
            print(f"⚠️  Failed to copy {failed_copies} images in {split_name} split")
        
        print(f"✅ Successfully processed {len(annotations):,} images for {split_name}")
        return annotations
    
    # Process each split
    train_annotations = process_split(train_df, "train", train_dir)
    val_annotations = process_split(val_df, "val", val_dir)
    
    test_annotations = []
    if len(test_df) > 0:
        test_annotations = process_split(test_df, "test", test_dir)
    
    def save_annotations(annotations: List[Dict], csv_path: str, split_name: str):
        """Save annotations in simple img_path,age format"""
        if not annotations:
            print(f"⚠️  No annotations to save for {split_name}")
            return
        
        df = pd.DataFrame(annotations)
        df.to_csv(csv_path, index=False)
        print(f"💾 Saved {len(annotations):,} annotations to {os.path.basename(csv_path)}")
        
        # Show preview
        print(f"   Preview of {split_name}_annotations.csv:")
        print("   img_path,age")
        for i, item in enumerate(annotations[:3]):
            print(f"   {item['img_path']},{item['age']}")
        if len(annotations) > 3:
            print("   ...")
    
    # Save annotation CSVs
    train_csv_path = os.path.join(output_dir, 'train_annotations.csv')
    val_csv_path = os.path.join(output_dir, 'val_annotations.csv')
    test_csv_path = os.path.join(output_dir, 'test_annotations.csv') if test_annotations else None
    
    save_annotations(train_annotations, train_csv_path, "train")
    save_annotations(val_annotations, val_csv_path, "val")
    if test_annotations:
        save_annotations(test_annotations, test_csv_path, "test")
    
    # Create comprehensive summary
    summary = {
        'creation_timestamp': pd.Timestamp.now().isoformat(),
        'source_directory': balanced_datasets_dir,
        'output_directory': output_dir,
        'split_ratios': {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio
        },
        'total_samples': len(combined_df),
        'split_samples': {
            'train': len(train_annotations),
            'val': len(val_annotations),
            'test': len(test_annotations) if test_annotations else 0
        },
        'datasets_included': dataset_stats,
        'age_statistics': {
            'overall': {
                'min': float(combined_df['age'].min()),
                'max': float(combined_df['age'].max()),
                'mean': float(combined_df['age'].mean()),
                'std': float(combined_df['age'].std())
            },
            'train': {
                'min': float(train_df['age'].min()),
                'max': float(train_df['age'].max()),
                'mean': float(train_df['age'].mean()),
                'std': float(train_df['age'].std())
            },
            'val': {
                'min': float(val_df['age'].min()),
                'max': float(val_df['age'].max()),
                'mean': float(val_df['age'].mean()),
                'std': float(val_df['age'].std())
            }
        }
    }
    
    if test_annotations:
        summary['age_statistics']['test'] = {
            'min': float(test_df['age'].min()),
            'max': float(test_df['age'].max()),
            'mean': float(test_df['age'].mean()),
            'std': float(test_df['age'].std())
        }
    
    # Add gender statistics if available
    if has_gender:
        summary['gender_statistics'] = {
            'overall': dict(combined_df[gender_col].value_counts()),
            'train': dict(train_df[gender_col].value_counts()),
            'val': dict(val_df[gender_col].value_counts())
        }
        if test_annotations:
            summary['gender_statistics']['test'] = dict(test_df[gender_col].value_counts())
    
    # Save summary
    summary_path = os.path.join(output_dir, 'dataset_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Create comprehensive README
    readme_content = f"""# Unified Age Estimation Dataset

## 📊 Dataset Statistics

### Overall
- **Total Samples**: {len(combined_df):,}
- **Age Range**: {combined_df['age'].min():.0f}-{combined_df['age'].max():.0f} years
- **Mean Age**: {combined_df['age'].mean():.1f} ± {combined_df['age'].std():.1f} years

### Data Splits
- **Train**: {len(train_annotations):,} samples ({len(train_annotations)/len(combined_df)*100:.1f}%)
- **Validation**: {len(val_annotations):,} samples ({len(val_annotations)/len(combined_df)*100:.1f}%)"""

    if test_annotations:
        readme_content += f"\n- **Test**: {len(test_annotations):,} samples ({len(test_annotations)/len(combined_df)*100:.1f}%)"

    readme_content += f"""

## 🗂️ Source Datasets
{chr(10).join([f"- **{name}**: {stats['samples']:,} samples (age: {stats['age_range'][0]}-{stats['age_range'][1]})" for name, stats in dataset_stats.items()])}

## 📁 Directory Structure
```
{os.path.basename(output_dir)}/
├── train/                    # Training images ({len(train_annotations):,} files)
├── val/                      # Validation images ({len(val_annotations):,} files)"""

    if test_annotations:
        readme_content += f"\n├── test/                     # Test images ({len(test_annotations):,} files)"

    readme_content += f"""
├── train_annotations.csv     # Training labels
├── val_annotations.csv       # Validation labels"""

    if test_annotations:
        readme_content += f"\n├── test_annotations.csv      # Test labels"

    readme_content += f"""
├── dataset_summary.json      # Detailed statistics
└── README.md                 # This file
```

## 📋 CSV Format
Each annotation CSV contains two columns:
```
img_path,age
{train_annotations[0]['img_path'] if train_annotations else 'example_image.jpg'},{train_annotations[0]['age'] if train_annotations else '25'}
```

## 🚀 Usage Example
```python
# Update your training configuration:
config = {{
    'train_csv': '{train_csv_path}',
    'val_csv': '{val_csv_path}',
    'train_dir': '{train_dir}',
    'val_dir': '{val_dir}',
}}

# Load data
import pandas as pd
train_df = pd.read_csv(config['train_csv'])
val_df = pd.read_csv(config['val_csv'])

# Image paths are relative to train_dir/val_dir
train_image_path = os.path.join(config['train_dir'], train_df.iloc[0]['img_path'])
```

## 🎯 Data Balance
The dataset maintains age distribution balance across train/validation/test splits using stratified sampling.

### Age Distribution by Split
- **Train**: {summary['age_statistics']['train']['mean']:.1f} ± {summary['age_statistics']['train']['std']:.1f} years
- **Val**: {summary['age_statistics']['val']['mean']:.1f} ± {summary['age_statistics']['val']['std']:.1f} years"""

    if test_annotations:
        readme_content += f"\n- **Test**: {summary['age_statistics']['test']['mean']:.1f} ± {summary['age_statistics']['test']['std']:.1f} years"

    if has_gender:
        readme_content += f"""

### Gender Distribution
- **Overall**: {dict(combined_df[gender_col].value_counts())}
- **Train**: {dict(train_df[gender_col].value_counts())}
- **Val**: {dict(val_df[gender_col].value_counts())}"""
        if test_annotations:
            readme_content += f"\n- **Test**: {dict(test_df[gender_col].value_counts())}"

    readme_content += f"""

## 📝 Notes
- Images are prefixed with dataset name to avoid filename conflicts
- Stratified sampling ensures age distribution balance across splits
- All image paths in CSV files are relative to their respective directories
- Original dataset sources are preserved in the summary file
"""

    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    # Print final summary
    print(f"\n🎉 Unified dataset created successfully!")
    print(f"📍 Location: {output_dir}")
    print(f"📊 Total samples: {len(combined_df):,}")
    print(f"🏷️  Train: {len(train_annotations):,} | Val: {len(val_annotations):,}", end="")
    if test_annotations:
        print(f" | Test: {len(test_annotations):,}")
    else:
        print()
    
    print(f"\n📋 Configuration for your training script:")
    print(f"'train_csv': '{train_csv_path}'")
    print(f"'val_csv': '{val_csv_path}'")
    print(f"'train_dir': '{train_dir}'")
    print(f"'val_dir': '{val_dir}'")
    
    # Return paths and summary
    result = {
        'train_csv': train_csv_path,
        'val_csv': val_csv_path,
        'train_dir': train_dir,
        'val_dir': val_dir,
        'summary_json': summary_path,
        'readme': readme_path,
        'summary': summary
    }
    
    if test_annotations:
        result['test_csv'] = test_csv_path
        result['test_dir'] = test_dir
    
    return result

def analyze_unified_dataset(output_dir: str):
    """
    Analyze the created unified dataset and print detailed statistics
    """
    summary_path = os.path.join(output_dir, 'dataset_summary.json')
    
    if not os.path.exists(summary_path):
        print(f"Summary file not found: {summary_path}")
        return
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    print(f"\n📊 Unified Dataset Analysis: {os.path.basename(output_dir)}")
    print("=" * 60)
    
    # Overall statistics
    print(f"Total samples: {summary['total_samples']:,}")
    print(f"Age range: {summary['age_statistics']['overall']['min']:.0f}-{summary['age_statistics']['overall']['max']:.0f} years")
    print(f"Mean age: {summary['age_statistics']['overall']['mean']:.1f} ± {summary['age_statistics']['overall']['std']:.1f}")
    
    # Split breakdown
    print(f"\nSplit breakdown:")
    for split, count in summary['split_samples'].items():
        if count > 0:
            percentage = count / summary['total_samples'] * 100
            print(f"  {split.capitalize()}: {count:,} ({percentage:.1f}%)")
    
    # Dataset contributions
    print(f"\nDataset contributions:")
    for dataset, stats in summary['datasets_included'].items():
        percentage = stats['samples'] / summary['total_samples'] * 100
        print(f"  {dataset}: {stats['samples']:,} ({percentage:.1f}%)")
    
    # Age distribution by split
    print(f"\nAge distribution by split:")
    for split in ['train', 'val', 'test']:
        if split in summary['age_statistics']:
            stats = summary['age_statistics'][split]
            print(f"  {split.capitalize()}: {stats['mean']:.1f} ± {stats['std']:.1f} years")

if __name__ == "__main__":
    # Example usage
    unified_paths = create_unified_dataset(
        balanced_datasets_dir='balanced_age_datasets',
        output_dir='unified_age_dataset',
        exclude_datasets=['LAGENDA'],
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )
    
    # Analyze the created dataset
    analyze_unified_dataset('unified_age_dataset')

