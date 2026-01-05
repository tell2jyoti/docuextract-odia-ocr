#!/usr/bin/env python3
"""
Create Annotation Files for Odia Handwritten Character Dataset
Maps image filenames to Odia character labels using the label CSV
"""

import os
import json
import csv
from pathlib import Path
from collections import defaultdict
import re

# Configuration
BASE_DIR = Path("/home/tinu/Downloads/docuextratc")
HANDWRITTEN_DIR = BASE_DIR / "Odia_handwritten"
LABEL_CSV = BASE_DIR / "dataset" / "odia_charcter" / "odia_characters_label.csv"
OUTPUT_DIR = BASE_DIR / "annotations"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_character_mapping():
    """Load character label mapping from CSV"""
    print("Loading character label mapping...")

    char_map = {}
    with open(LABEL_CSV, 'r', encoding='utf-8-sig') as f:  # utf-8-sig to handle BOM
        reader = csv.DictReader(f)
        for row in reader:
            class_id = int(row['class'])
            character = row['label']
            char_map[class_id] = character

    print(f"✓ Loaded {len(char_map)} character mappings")
    for i in range(min(10, len(char_map))):
        if i in char_map:
            print(f"  Class {i}: {char_map[i]}")

    return char_map

def extract_class_from_filename(filename, char_type):
    """
    Extract class ID from filename based on character type

    Simple chars:
      - 0000010000_resized.jpg -> class 1
      - aug_0_1001.jpeg -> class 0
      - 0000010000_resized_binary.jpg -> class 1
      - 0000010000_resized_inverted_binary.jpg -> class 1
    Compound chars:
      - 1001.jpg -> class 10
    """
    # Remove various suffixes and extensions
    name = filename
    # Remove all possible suffixes
    for suffix in ['_resized_inverted_binary.jpg', '_resized_binary.jpg', '_resized.jpg', '.jpg', '.jpeg']:
        name = name.replace(suffix, '')

    # Check for augmented filename pattern: aug_CLASS_SAMPLE
    aug_match = re.match(r'aug_(\d+)_\d+', name)
    if aug_match:
        class_id = int(aug_match.group(1))
        return class_id

    if char_type == 'simple':
        # Simple characters: 10-digit format
        # Pattern: 00000CXXXX where C is class ID (1-2 digits)
        # Examples: 0000010000 -> class 1, 0000120000 -> class 12

        # Remove leading zeros and extract class
        # The class ID appears after the initial zeros
        match = re.match(r'0*(\d+?)(\d{4})$', name)
        if match:
            class_id = int(match.group(1))
            return class_id

    elif char_type == 'compound':
        # Compound characters: variable length
        # Pattern: CCXXX where CC is class ID (2 digits typically)
        # Examples: 1001 -> class 10, 44123 -> class 44

        # Try to extract first 2 digits as class
        if len(name) >= 2:
            # Check if first 2 digits form a valid class
            potential_class = int(name[:2])
            return potential_class

    return None

def process_simple_characters(char_map):
    """Process simple character dataset"""
    print("\n" + "="*70)
    print("Processing Simple Characters")
    print("="*70)

    simple_dir = HANDWRITTEN_DIR / "Odia_Handwritten_Simple_Character" / "Odia_simple_character"

    if not simple_dir.exists():
        print(f"⚠ Simple character directory not found: {simple_dir}")
        return []

    annotations = []
    stats = defaultdict(int)
    class_distribution = defaultdict(int)

    # Process each augmentation variant
    augmentation_folders = [
        'grayscale',
        'binary',
        'inverted',
        'aug_grayscale',
        'aug_binary',
        'aug_inverted',
        'Odia Simple Character',
        'Inverted_Odia_SimpleCharacter'
    ]

    for aug_folder in augmentation_folders:
        folder_path = simple_dir / aug_folder

        if not folder_path.exists():
            continue

        print(f"\nProcessing: {aug_folder}")

        # Get both .jpg and .jpeg files
        image_files = list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.jpeg'))
        print(f"  Found {len(image_files)} images")

        for img_path in image_files:
            filename = img_path.name

            # Extract class ID from filename
            class_id = extract_class_from_filename(filename, 'simple')

            if class_id is not None and class_id in char_map:
                character = char_map[class_id]

                # Create annotation entry
                annotation = {
                    'image_path': str(img_path.relative_to(BASE_DIR)),
                    'absolute_path': str(img_path),
                    'filename': filename,
                    'class_id': class_id,
                    'character': character,
                    'type': 'simple',
                    'augmentation': aug_folder
                }

                annotations.append(annotation)
                stats[aug_folder] += 1
                class_distribution[class_id] += 1
            else:
                # Debug: show unmapped files
                if len(annotations) < 5:  # Only show first few
                    print(f"    ⚠ Could not map: {filename} (class_id: {class_id})")

    # Print statistics
    print(f"\n✓ Processed {len(annotations)} simple character images")
    print(f"\nAugmentation breakdown:")
    for aug, count in sorted(stats.items()):
        print(f"  {aug}: {count:,}")

    print(f"\nClass distribution (top 10):")
    for class_id, count in sorted(class_distribution.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  Class {class_id} ({char_map[class_id]}): {count:,}")

    return annotations

def process_compound_characters(char_map):
    """Process compound character dataset"""
    print("\n" + "="*70)
    print("Processing Compound Characters")
    print("="*70)

    compound_base = HANDWRITTEN_DIR / "Odia_Compound_Character" / "Odia_Compound_Character"

    if not compound_base.exists():
        print(f"⚠ Compound character directory not found: {compound_base}")
        return []

    annotations = []
    stats = defaultdict(int)
    class_distribution = defaultdict(int)

    # Find all subdirectories
    for root, dirs, files in os.walk(compound_base):
        root_path = Path(root)
        folder_name = root_path.name

        # Skip if not an augmentation or character folder
        image_files = [f for f in files if f.endswith('.jpg') or f.endswith('.jpeg')]

        if not image_files:
            continue

        print(f"\nProcessing: {folder_name} ({len(image_files)} images)")

        for filename in image_files:
            img_path = root_path / filename

            # Extract class ID from filename
            class_id = extract_class_from_filename(filename, 'compound')

            if class_id is not None and class_id in char_map:
                character = char_map[class_id]

                # Create annotation entry
                annotation = {
                    'image_path': str(img_path.relative_to(BASE_DIR)),
                    'absolute_path': str(img_path),
                    'filename': filename,
                    'class_id': class_id,
                    'character': character,
                    'type': 'compound',
                    'augmentation': folder_name
                }

                annotations.append(annotation)
                stats[folder_name] += 1
                class_distribution[class_id] += 1

    # Print statistics
    print(f"\n✓ Processed {len(annotations)} compound character images")

    print(f"\nClass distribution (top 10):")
    for class_id, count in sorted(class_distribution.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  Class {class_id} ({char_map[class_id]}): {count:,}")

    return annotations

def validate_character_coverage(annotations, char_map):
    """Validate that all expected characters are present"""
    print("\n" + "="*70)
    print("Validating Character Coverage")
    print("="*70)

    # Get unique class IDs in annotations
    present_classes = set(ann['class_id'] for ann in annotations)
    expected_classes = set(char_map.keys())

    print(f"\nExpected classes: {len(expected_classes)}")
    print(f"Present classes: {len(present_classes)}")

    # Missing classes
    missing_classes = expected_classes - present_classes
    if missing_classes:
        print(f"\n⚠ Missing {len(missing_classes)} character classes:")
        for class_id in sorted(missing_classes):
            print(f"  Class {class_id}: {char_map[class_id]}")
    else:
        print("\n✓ All character classes present!")

    # Extra classes (shouldn't happen)
    extra_classes = present_classes - expected_classes
    if extra_classes:
        print(f"\n⚠ Found {len(extra_classes)} unexpected classes:")
        for class_id in sorted(extra_classes):
            print(f"  Class {class_id}")

    return {
        'expected': len(expected_classes),
        'present': len(present_classes),
        'missing': sorted(list(missing_classes)),
        'complete': len(missing_classes) == 0
    }

def save_annotations(annotations, char_map, coverage_report):
    """Save annotations to JSON file"""
    print("\n" + "="*70)
    print("Saving Annotations")
    print("="*70)

    # Calculate statistics
    total_images = len(annotations)
    unique_classes = len(set(ann['class_id'] for ann in annotations))

    # Count by type
    simple_count = sum(1 for ann in annotations if ann['type'] == 'simple')
    compound_count = sum(1 for ann in annotations if ann['type'] == 'compound')

    # Create metadata
    metadata = {
        'total_images': total_images,
        'unique_classes': unique_classes,
        'expected_classes': len(char_map),
        'simple_characters': simple_count,
        'compound_characters': compound_count,
        'character_mapping': {str(k): v for k, v in char_map.items()},
        'coverage': coverage_report
    }

    # Save full annotations
    output_file = OUTPUT_DIR / 'handwritten_characters_full.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': metadata,
            'annotations': annotations
        }, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved full annotations: {output_file}")
    print(f"  Total images: {total_images:,}")
    print(f"  Unique classes: {unique_classes}")
    print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

    # Save metadata only
    metadata_file = OUTPUT_DIR / 'metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved metadata: {metadata_file}")

    # Save class distribution
    class_dist = defaultdict(int)
    for ann in annotations:
        class_dist[ann['class_id']] += 1

    dist_file = OUTPUT_DIR / 'class_distribution.json'
    with open(dist_file, 'w', encoding='utf-8') as f:
        json.dump({
            str(k): {
                'character': char_map[k],
                'count': v
            } for k, v in sorted(class_dist.items())
        }, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved class distribution: {dist_file}")

def main():
    print("\n" + "="*70)
    print("Odia Handwritten Character Annotation Generator")
    print("="*70)
    print()

    # Step 1: Load character mapping
    char_map = load_character_mapping()

    # Step 2: Process simple characters
    simple_annotations = process_simple_characters(char_map)

    # Step 3: Process compound characters
    compound_annotations = process_compound_characters(char_map)

    # Step 4: Combine annotations
    all_annotations = simple_annotations + compound_annotations

    print("\n" + "="*70)
    print(f"Total Annotations: {len(all_annotations):,}")
    print("="*70)

    if not all_annotations:
        print("⚠ No annotations created! Please check the data paths.")
        return

    # Step 5: Validate coverage
    coverage_report = validate_character_coverage(all_annotations, char_map)

    # Step 6: Save annotations
    save_annotations(all_annotations, char_map, coverage_report)

    print("\n" + "="*70)
    print("✅ ANNOTATION GENERATION COMPLETE!")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Review class_distribution.json for data balance")
    print("  2. Check metadata.json for coverage report")
    print("  3. Use handwritten_characters_full.json for dataset splits")
    print("  4. Run: python create_splits.py (to be created next)")

if __name__ == "__main__":
    main()
