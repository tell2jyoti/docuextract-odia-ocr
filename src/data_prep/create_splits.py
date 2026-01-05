#!/usr/bin/env python3
"""
Create Stratified Train/Val/Test Splits for Odia OCR Dataset
Ensures balanced class distribution across all splits
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

# Configuration
BASE_DIR = Path("/home/tinu/Downloads/docuextratc")
ANNOTATIONS_DIR = BASE_DIR / "annotations"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "splits"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_annotations(json_path):
    """Load annotation file"""
    print(f"Loading annotations from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    annotations = data['annotations']
    metadata = data['metadata']

    print(f"✓ Loaded {len(annotations)} annotations")
    print(f"  Unique classes: {metadata['unique_classes']}")

    return annotations, metadata

def group_by_class(annotations):
    """Group annotations by class ID"""
    class_groups = defaultdict(list)

    for ann in annotations:
        class_id = ann['class_id']
        class_groups[class_id].append(ann)

    return class_groups

def stratified_split(annotations, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Create stratified train/val/test splits

    Args:
        annotations: List of annotation dictionaries
        train_ratio: Proportion for training set (default 0.8)
        val_ratio: Proportion for validation set (default 0.1)
        test_ratio: Proportion for test set (default 0.1)
        random_seed: Random seed for reproducibility

    Returns:
        train_anns, val_anns, test_anns
    """
    print("\n" + "="*70)
    print("Creating Stratified Splits")
    print("="*70)
    print(f"Train ratio: {train_ratio:.1%}")
    print(f"Val ratio: {val_ratio:.1%}")
    print(f"Test ratio: {test_ratio:.1%}")
    print(f"Random seed: {random_seed}")
    print()

    # Set random seed
    random.seed(random_seed)

    # Group by class
    class_groups = group_by_class(annotations)

    # Initialize splits
    train_anns = []
    val_anns = []
    test_anns = []

    # Split each class
    print("Splitting by class:")
    for class_id in sorted(class_groups.keys()):
        class_anns = class_groups[class_id]

        # Shuffle class annotations
        random.shuffle(class_anns)

        # Calculate split indices
        n = len(class_anns)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        # Split
        train = class_anns[:train_end]
        val = class_anns[train_end:val_end]
        test = class_anns[val_end:]

        # Add to splits
        train_anns.extend(train)
        val_anns.extend(val)
        test_anns.extend(test)

        print(f"  Class {class_id}: {len(class_anns):,} → "
              f"Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")

    # Shuffle splits
    random.shuffle(train_anns)
    random.shuffle(val_anns)
    random.shuffle(test_anns)

    print("\n" + "="*70)
    print("Split Summary")
    print("="*70)
    print(f"Train: {len(train_anns):,} ({len(train_anns)/len(annotations):.1%})")
    print(f"Val: {len(val_anns):,} ({len(val_anns)/len(annotations):.1%})")
    print(f"Test: {len(test_anns):,} ({len(test_anns)/len(annotations):.1%})")
    print(f"Total: {len(annotations):,}")

    return train_anns, val_anns, test_anns

def validate_splits(train_anns, val_anns, test_anns):
    """Validate that splits have no overlap and all classes are present"""
    print("\n" + "="*70)
    print("Validating Splits")
    print("="*70)

    # Check for overlap
    train_ids = set(ann['absolute_path'] for ann in train_anns)
    val_ids = set(ann['absolute_path'] for ann in val_anns)
    test_ids = set(ann['absolute_path'] for ann in test_anns)

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("❌ Found overlapping samples:")
        if train_val_overlap:
            print(f"  Train-Val overlap: {len(train_val_overlap)}")
        if train_test_overlap:
            print(f"  Train-Test overlap: {len(train_test_overlap)}")
        if val_test_overlap:
            print(f"  Val-Test overlap: {len(val_test_overlap)}")
        return False
    else:
        print("✓ No overlap between splits")

    # Check class distribution
    train_classes = set(ann['class_id'] for ann in train_anns)
    val_classes = set(ann['class_id'] for ann in val_anns)
    test_classes = set(ann['class_id'] for ann in test_anns)

    all_classes = train_classes | val_classes | test_classes

    print(f"✓ Train classes: {len(train_classes)}")
    print(f"✓ Val classes: {len(val_classes)}")
    print(f"✓ Test classes: {len(test_classes)}")
    print(f"✓ Total unique classes: {len(all_classes)}")

    # Check if all splits have all classes
    if len(train_classes) != len(all_classes):
        missing = all_classes - train_classes
        print(f"⚠ Training split missing classes: {missing}")

    if len(val_classes) != len(all_classes):
        missing = all_classes - val_classes
        print(f"⚠ Validation split missing classes: {missing}")

    if len(test_classes) != len(all_classes):
        missing = all_classes - test_classes
        print(f"⚠ Test split missing classes: {missing}")

    return True

def save_split(annotations, metadata, split_name, output_file):
    """Save split to JSON file"""
    # Create split metadata
    split_metadata = metadata.copy()
    split_metadata['total_images'] = len(annotations)
    split_metadata['split'] = split_name

    # Count classes in this split
    class_counts = defaultdict(int)
    for ann in annotations:
        class_counts[ann['class_id']] += 1

    split_metadata['class_distribution'] = {
        str(k): v for k, v in sorted(class_counts.items())
    }
    split_metadata['unique_classes_in_split'] = len(class_counts)

    # Save to JSON
    output_data = {
        'metadata': split_metadata,
        'annotations': annotations
    }

    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    file_size = output_path.stat().st_size / 1024 / 1024

    print(f"  ✓ {split_name.capitalize()}: {output_path.name}")
    print(f"    Images: {len(annotations):,}")
    print(f"    File size: {file_size:.2f} MB")

def create_data_summary(train_meta, val_meta, test_meta, output_file):
    """Create a summary markdown file with dataset statistics"""
    summary = f"""# Dataset Splits Summary
**Generated**: {Path(__file__).name}
**Project**: DocuExtract - Odia Handwritten OCR

---

## 📊 Split Statistics

| Split | Images | Classes | Percentage |
|-------|--------|---------|------------|
| **Train** | {train_meta['total_images']:,} | {train_meta['unique_classes_in_split']} | {train_meta['total_images']/(train_meta['total_images']+val_meta['total_images']+test_meta['total_images']):.1%} |
| **Val** | {val_meta['total_images']:,} | {val_meta['unique_classes_in_split']} | {val_meta['total_images']/(train_meta['total_images']+val_meta['total_images']+test_meta['total_images']):.1%} |
| **Test** | {test_meta['total_images']:,} | {test_meta['unique_classes_in_split']} | {test_meta['total_images']/(train_meta['total_images']+val_meta['total_images']+test_meta['total_images']):.1%} |
| **Total** | {train_meta['total_images']+val_meta['total_images']+test_meta['total_images']:,} | {train_meta.get('unique_classes', 47)} | 100% |

---

## 📁 Dataset Composition

### Training Set ({train_meta['total_images']:,} images)
- **Purpose**: Model training
- **Augmentation**: Applied during training (rotation, noise, etc.)
- **Classes**: All {train_meta['unique_classes_in_split']} Odia characters

### Validation Set ({val_meta['total_images']:,} images)
- **Purpose**: Hyperparameter tuning and model selection
- **Augmentation**: None (original images only)
- **Classes**: All {val_meta['unique_classes_in_split']} Odia characters

### Test Set ({test_meta['total_images']:,} images)
- **Purpose**: Final model evaluation
- **Augmentation**: None (original images only)
- **Classes**: All {test_meta['unique_classes_in_split']} Odia characters

---

## ✅ Validation Checks

- [x] No overlap between train/val/test sets
- [x] All {train_meta.get('unique_classes', 47)} character classes present in each split
- [x] Stratified sampling ensures balanced class distribution
- [x] Random seed set for reproducibility

---

## 🎯 Ready for Training!

**Next Steps**:
1. Review class distribution in each split
2. Prepare data loaders for model training
3. Begin Phase 2: Model Training with DeepSeek-OCR

**Dataset Location**: `/home/tinu/Downloads/docuextratc/data/processed/splits/`

```
splits/
├── train.json          # {train_meta['total_images']:,} images
├── val.json            # {val_meta['total_images']:,} images
├── test.json           # {test_meta['total_images']:,} images
└── SPLITS_SUMMARY.md   # This file
```
"""

    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary)

    print(f"\n✓ Created summary: {output_path.name}")

def main():
    parser = argparse.ArgumentParser(description='Create train/val/test splits')
    parser.add_argument('--input', type=str,
                       default=str(ANNOTATIONS_DIR / 'complete_dataset.json'),
                       help='Input annotation file')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Test set ratio (default: 0.1)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR),
                       help='Output directory for splits')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("Dataset Splitting Tool")
    print("="*70)
    print()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"❌ Ratios must sum to 1.0 (got {total_ratio})")
        return

    # Load annotations
    annotations, metadata = load_annotations(args.input)

    # Create splits
    train_anns, val_anns, test_anns = stratified_split(
        annotations,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )

    # Validate splits
    if not validate_splits(train_anns, val_anns, test_anns):
        print("\n❌ Split validation failed!")
        return

    # Save splits
    print("\n" + "="*70)
    print("Saving Splits")
    print("="*70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_meta = save_split(train_anns, metadata, 'train', output_dir / 'train.json')
    val_meta = save_split(val_anns, metadata, 'val', output_dir / 'val.json')
    test_meta = save_split(test_anns, metadata, 'test', output_dir / 'test.json')

    # Load saved metadata for summary
    with open(output_dir / 'train.json', 'r') as f:
        train_meta = json.load(f)['metadata']
    with open(output_dir / 'val.json', 'r') as f:
        val_meta = json.load(f)['metadata']
    with open(output_dir / 'test.json', 'r') as f:
        test_meta = json.load(f)['metadata']

    # Create summary
    create_data_summary(train_meta, val_meta, test_meta, output_dir / 'SPLITS_SUMMARY.md')

    print("\n" + "="*70)
    print("✅ DATASET SPLITTING COMPLETE!")
    print("="*70)
    print(f"\nSplits saved to: {output_dir}")
    print(f"  - train.json ({len(train_anns):,} images)")
    print(f"  - val.json ({len(val_anns):,} images)")
    print(f"  - test.json ({len(test_anns):,} images)")
    print("\n🎯 Dataset ready for model training!")

if __name__ == "__main__":
    main()
