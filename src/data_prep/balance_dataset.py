#!/usr/bin/env python3
"""
Balance Dataset Script for DocuExtract Project
- Downsample overrepresented classes
- Augment underrepresented classes
- Create balanced annotation file
"""

import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import argparse

# Configuration
BASE_DIR = Path("/home/tinu/Downloads/docuextratc")
ANNOTATIONS_DIR = BASE_DIR / "annotations"
OUTPUT_DIR = BASE_DIR / "data" / "balanced"

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
    print(f"  Classes: {metadata['unique_classes']}")

    return annotations, metadata

def group_by_class(annotations):
    """Group annotations by class ID"""
    class_groups = defaultdict(list)

    for ann in annotations:
        class_id = ann['class_id']
        class_groups[class_id].append(ann)

    return class_groups

def augment_image(image_path, output_path, aug_type='rotation'):
    """
    Apply augmentation to an image

    Augmentation types:
    - rotation: Small rotation (±5-10 degrees)
    - elastic: Elastic deformation
    - noise: Add Gaussian noise
    - brightness: Adjust brightness/contrast
    """
    try:
        # Load image
        img = Image.open(image_path)

        if aug_type == 'rotation':
            # Random rotation between -10 and 10 degrees
            angle = random.uniform(-10, 10)
            img = img.rotate(angle, fillcolor='white', expand=False)

        elif aug_type == 'brightness':
            # Random brightness adjustment
            enhancer = ImageEnhance.Brightness(img)
            factor = random.uniform(0.7, 1.3)
            img = enhancer.enhance(factor)

            # Random contrast adjustment
            enhancer = ImageEnhance.Contrast(img)
            factor = random.uniform(0.8, 1.2)
            img = enhancer.enhance(factor)

        elif aug_type == 'noise':
            # Convert to numpy array
            img_array = np.array(img)

            # Add Gaussian noise
            noise = np.random.normal(0, 10, img_array.shape).astype(np.uint8)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

            img = Image.fromarray(img_array)

        elif aug_type == 'blur':
            # Slight Gaussian blur
            radius = random.uniform(0.5, 1.5)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        elif aug_type == 'elastic':
            # Simple elastic deformation using slight shift
            img_array = np.array(img)

            # Random small shifts
            shift_x = random.randint(-2, 2)
            shift_y = random.randint(-2, 2)

            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            img_array = cv2.warpAffine(img_array, M, (img_array.shape[1], img_array.shape[0]),
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=255)

            img = Image.fromarray(img_array)

        # Save augmented image
        img.save(output_path)
        return True

    except Exception as e:
        print(f"⚠ Failed to augment {image_path}: {e}")
        return False

def downsample_class(annotations, max_samples):
    """Randomly downsample annotations to max_samples"""
    if len(annotations) <= max_samples:
        return annotations

    return random.sample(annotations, max_samples)

def augment_class(annotations, target_samples, char_id, character):
    """Augment class to reach target_samples"""
    current_count = len(annotations)

    if current_count >= target_samples:
        return annotations

    needed_samples = target_samples - current_count
    augmented_annotations = annotations.copy()

    print(f"  Augmenting class {char_id} ({character}): {current_count} → {target_samples}")

    # Create augmented images directory
    aug_dir = OUTPUT_DIR / "augmented_images" / f"class_{char_id}"
    aug_dir.mkdir(parents=True, exist_ok=True)

    # Augmentation types to cycle through
    aug_types = ['rotation', 'brightness', 'noise', 'blur', 'elastic']

    samples_created = 0
    max_attempts = needed_samples * 3  # Prevent infinite loop
    attempts = 0

    while samples_created < needed_samples and attempts < max_attempts:
        # Randomly select an original image
        source_ann = random.choice(annotations)
        source_path = BASE_DIR / source_ann['image_path']

        # Select augmentation type
        aug_type = aug_types[samples_created % len(aug_types)]

        # Create augmented filename
        aug_filename = f"aug_{char_id}_{aug_type}_{samples_created}.jpg"
        aug_path = aug_dir / aug_filename

        # Apply augmentation
        if augment_image(source_path, aug_path, aug_type):
            # Create new annotation
            new_ann = source_ann.copy()
            new_ann['image_path'] = str(aug_path.relative_to(BASE_DIR))
            new_ann['absolute_path'] = str(aug_path)
            new_ann['filename'] = aug_filename
            new_ann['augmentation'] = f'balanced_{aug_type}'

            augmented_annotations.append(new_ann)
            samples_created += 1

        attempts += 1

    print(f"    Created {samples_created} augmented samples")

    return augmented_annotations

def balance_dataset(input_path, max_samples_per_class=10000, min_samples_per_class=3000):
    """
    Balance dataset by downsampling and augmenting

    Args:
        input_path: Path to input annotation JSON
        max_samples_per_class: Maximum samples per class (downsample above this)
        min_samples_per_class: Minimum samples per class (augment below this)
    """
    print("\n" + "="*70)
    print("Dataset Balancing")
    print("="*70)
    print(f"Max samples per class: {max_samples_per_class}")
    print(f"Min samples per class: {min_samples_per_class}")
    print()

    # Load annotations
    annotations, metadata = load_annotations(input_path)

    # Group by class
    class_groups = group_by_class(annotations)

    print(f"\nOriginal class distribution:")
    for class_id in sorted(class_groups.keys()):
        count = len(class_groups[class_id])
        char = metadata['character_mapping'][str(class_id)]
        status = ""
        if count > max_samples_per_class:
            status = f" → downsample to {max_samples_per_class}"
        elif count < min_samples_per_class:
            status = f" → augment to {min_samples_per_class}"
        print(f"  Class {class_id} ({char}): {count:,}{status}")

    # Balance each class
    balanced_annotations = []
    balanced_stats = defaultdict(int)

    print("\n" + "="*70)
    print("Balancing Classes")
    print("="*70)

    for class_id in sorted(class_groups.keys()):
        class_anns = class_groups[class_id]
        char = metadata['character_mapping'][str(class_id)]
        original_count = len(class_anns)

        # Downsample if needed
        if original_count > max_samples_per_class:
            print(f"\nDownsampling class {class_id} ({char}): {original_count:,} → {max_samples_per_class:,}")
            class_anns = downsample_class(class_anns, max_samples_per_class)

        # Augment if needed
        elif original_count < min_samples_per_class:
            class_anns = augment_class(class_anns, min_samples_per_class, class_id, char)

        balanced_annotations.extend(class_anns)
        balanced_stats[class_id] = len(class_anns)

    # Update metadata
    balanced_metadata = metadata.copy()
    balanced_metadata['total_images'] = len(balanced_annotations)
    balanced_metadata['balancing_applied'] = True
    balanced_metadata['max_samples_per_class'] = max_samples_per_class
    balanced_metadata['min_samples_per_class'] = min_samples_per_class

    print("\n" + "="*70)
    print("Balanced Class Distribution")
    print("="*70)
    for class_id in sorted(balanced_stats.keys()):
        char = metadata['character_mapping'][str(class_id)]
        original = len(class_groups[class_id])
        balanced = balanced_stats[class_id]
        change = balanced - original
        change_str = f"+{change}" if change > 0 else str(change)
        print(f"  Class {class_id} ({char}): {balanced:,} ({change_str})")

    return balanced_annotations, balanced_metadata

def save_balanced_annotations(annotations, metadata, output_path):
    """Save balanced annotations to JSON"""
    print("\n" + "="*70)
    print("Saving Balanced Annotations")
    print("="*70)

    output_data = {
        'metadata': metadata,
        'annotations': annotations
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    file_size = output_path.stat().st_size / 1024 / 1024

    print(f"✓ Saved: {output_path}")
    print(f"  Total images: {len(annotations):,}")
    print(f"  File size: {file_size:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description='Balance OCR dataset')
    parser.add_argument('--input', type=str,
                       default=str(ANNOTATIONS_DIR / 'handwritten_characters_full.json'),
                       help='Input annotation file')
    parser.add_argument('--max-samples-per-class', type=int, default=10000,
                       help='Maximum samples per class')
    parser.add_argument('--min-samples-per-class', type=int, default=3000,
                       help='Minimum samples per class')
    parser.add_argument('--output', type=str,
                       default=str(ANNOTATIONS_DIR / 'balanced_annotations.json'),
                       help='Output annotation file')

    args = parser.parse_args()

    # Balance dataset
    balanced_anns, balanced_meta = balance_dataset(
        args.input,
        max_samples_per_class=args.max_samples_per_class,
        min_samples_per_class=args.min_samples_per_class
    )

    # Save
    save_balanced_annotations(balanced_anns, balanced_meta, Path(args.output))

    print("\n" + "="*70)
    print("✅ DATASET BALANCING COMPLETE!")
    print("="*70)
    print(f"\nBalanced annotations: {args.output}")
    print(f"Total images: {len(balanced_anns):,}")
    print("\nNext step: Generate synthetic data for missing classes")

if __name__ == "__main__":
    main()
