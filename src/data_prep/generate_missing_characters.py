#!/usr/bin/env python3
"""
Generate Synthetic Handwritten-Style Images for Missing Odia Characters
Creates realistic handwritten character images using fonts and transformations
"""

import json
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import argparse

# Configuration
BASE_DIR = Path("/home/tinu/Downloads/docuextratc")
FONTS_DIR = BASE_DIR / "fonts" / "odia"
OUTPUT_DIR = BASE_DIR / "data" / "synthetic" / "missing_characters"
LABEL_CSV = BASE_DIR / "dataset" / "odia_charcter" / "odia_characters_label.csv"

# Image dimensions (matching existing dataset)
IMG_WIDTH = 32
IMG_HEIGHT = 32

def load_character_mapping():
    """Load character mapping from CSV"""
    import csv

    char_map = {}
    with open(LABEL_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_id = int(row['class'])
            character = row['label']
            char_map[class_id] = character

    return char_map

def apply_handwriting_effects(img, intensity='medium'):
    """
    Apply transformations to make synthetic text look more handwritten

    Effects:
    - Random rotation
    - Slight elastic deformation via shifting
    - Noise
    - Blur
    - Brightness/contrast variation
    - Random thickness variation
    """
    # Convert to numpy array
    img_array = np.array(img)

    # 1. Random rotation (-15 to +15 degrees for more variation)
    if intensity in ['medium', 'high']:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, fillcolor=255, expand=False, resample=Image.BICUBIC)
        img_array = np.array(img)

    # 2. Random shift (simulate slight writing variations)
    if intensity == 'high':
        shift_x = random.randint(-2, 2)
        shift_y = random.randint(-2, 2)
        img_array = np.roll(img_array, shift_x, axis=1)
        img_array = np.roll(img_array, shift_y, axis=0)

    # 3. Add slight Gaussian noise
    if random.random() > 0.5:
        noise = np.random.normal(0, 5, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

    # 4. Random blur (simulate pen stroke blur)
    img = Image.fromarray(img_array)
    if random.random() > 0.6:
        blur_radius = random.uniform(0.3, 0.8)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # 5. Brightness/contrast variation
    if random.random() > 0.5:
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.8, 1.2)
        img = enhancer.enhance(factor)

    if random.random() > 0.5:
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(0.9, 1.3)
        img = enhancer.enhance(factor)

    # 6. Random thickness variation using simple erosion/dilation
    img_array = np.array(img)
    if random.random() > 0.7:
        # Simple morphological operation without scipy
        # Thicken strokes by darkening adjacent pixels
        h, w = img_array.shape
        padded = np.pad(img_array, 1, mode='constant', constant_values=255)

        # Simple averaging with neighbors to thicken strokes
        result = np.minimum(
            np.minimum(padded[:-2, 1:-1], padded[2:, 1:-1]),
            np.minimum(padded[1:-1, :-2], padded[1:-1, 2:])
        )
        img_array = np.minimum(img_array, result)

    return Image.fromarray(img_array)

def generate_character_image(character, font_path, size=32):
    """
    Generate a single character image

    Args:
        character: Odia character to render
        font_path: Path to TTF font file
        size: Image size (default 32x32)

    Returns:
        PIL Image
    """
    # Create blank white image
    img = Image.new('L', (size, size), color=255)
    draw = ImageDraw.Draw(img)

    # Load font (try different sizes to fit well)
    font_size = int(size * 0.8)  # 80% of image size
    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except Exception as e:
        print(f"⚠ Failed to load font {font_path}: {e}")
        return None

    # Get text bounding box to center it
    bbox = draw.textbbox((0, 0), character, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Calculate position to center text
    x = (size - text_width) // 2 - bbox[0]
    y = (size - text_height) // 2 - bbox[1]

    # Draw character
    draw.text((x, y), character, font=font, fill=0)

    return img

def generate_synthetic_dataset(missing_class_ids, samples_per_class=3000, output_dir=OUTPUT_DIR):
    """
    Generate synthetic handwritten-style images for missing character classes

    Args:
        missing_class_ids: List of missing class IDs to generate
        samples_per_class: Number of samples to generate per class
        output_dir: Output directory for generated images
    """
    print("\n" + "="*70)
    print("Synthetic Character Generation")
    print("="*70)
    print(f"Missing classes: {len(missing_class_ids)}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Total images to generate: {len(missing_class_ids) * samples_per_class:,}")
    print()

    # Load character mapping
    char_map = load_character_mapping()

    # Get available fonts
    font_files = list(FONTS_DIR.glob("*.ttf"))
    if not font_files:
        print("❌ No fonts found in", FONTS_DIR)
        return None, None

    print(f"✓ Found {len(font_files)} fonts:")
    for font_file in font_files:
        print(f"  - {font_file.name}")
    print()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Store annotations
    annotations = []
    stats = {}

    # Generate images for each missing class
    for class_id in missing_class_ids:
        if class_id not in char_map:
            print(f"⚠ Class {class_id} not in character mapping, skipping")
            continue

        character = char_map[class_id]
        print(f"Generating class {class_id} ({character}): {samples_per_class} samples")

        # Create class directory
        class_dir = output_dir / f"class_{class_id}"
        class_dir.mkdir(exist_ok=True)

        samples_created = 0

        for i in range(samples_per_class):
            # Randomly select font
            font_file = random.choice(font_files)

            # Generate base character image
            img = generate_character_image(character, font_file, size=IMG_WIDTH)

            if img is None:
                continue

            # Apply handwriting effects with varying intensity
            intensity = random.choice(['medium', 'medium', 'high'])  # More medium than high
            img = apply_handwriting_effects(img, intensity=intensity)

            # Save image
            filename = f"synth_{class_id}_{i:05d}.jpg"
            img_path = class_dir / filename
            img.save(img_path, quality=95)

            # Create annotation
            annotation = {
                'image_path': str(img_path.relative_to(BASE_DIR)),
                'absolute_path': str(img_path),
                'filename': filename,
                'class_id': class_id,
                'character': character,
                'type': 'synthetic',
                'augmentation': f'synthetic_font_{font_file.stem}'
            }

            annotations.append(annotation)
            samples_created += 1

        stats[class_id] = samples_created
        print(f"  ✓ Created {samples_created} samples for class {class_id} ({character})")

    print("\n" + "="*70)
    print(f"✓ Generated {len(annotations):,} synthetic images")
    print("="*70)

    return annotations, stats

def save_synthetic_annotations(annotations, stats, output_file):
    """Save synthetic annotations to JSON"""
    print("\nSaving synthetic annotations...")

    # Create metadata
    metadata = {
        'total_images': len(annotations),
        'unique_classes': len(stats),
        'samples_per_class': stats,
        'type': 'synthetic_missing_characters',
        'characters': {}
    }

    # Add character info
    char_map = load_character_mapping()
    for class_id in stats.keys():
        metadata['characters'][str(class_id)] = {
            'character': char_map[class_id],
            'count': stats[class_id]
        }

    # Save to JSON
    output_data = {
        'metadata': metadata,
        'annotations': annotations
    }

    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    file_size = output_path.stat().st_size / 1024 / 1024

    print(f"✓ Saved: {output_path}")
    print(f"  Total images: {len(annotations):,}")
    print(f"  File size: {file_size:.2f} MB")

def merge_with_balanced_dataset(synthetic_file, balanced_file, output_file):
    """Merge synthetic data with balanced dataset"""
    print("\n" + "="*70)
    print("Merging Synthetic Data with Balanced Dataset")
    print("="*70)

    # Load both datasets
    with open(synthetic_file, 'r', encoding='utf-8') as f:
        synthetic_data = json.load(f)

    with open(balanced_file, 'r', encoding='utf-8') as f:
        balanced_data = json.load(f)

    # Merge annotations
    merged_annotations = balanced_data['annotations'] + synthetic_data['annotations']

    # Update metadata
    merged_metadata = balanced_data['metadata'].copy()
    merged_metadata['total_images'] = len(merged_annotations)
    merged_metadata['unique_classes'] = balanced_data['metadata']['unique_classes'] + synthetic_data['metadata']['unique_classes']
    merged_metadata['synthetic_classes_added'] = list(synthetic_data['metadata']['samples_per_class'].keys())

    # Save merged dataset
    output_data = {
        'metadata': merged_metadata,
        'annotations': merged_annotations
    }

    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    file_size = output_path.stat().st_size / 1024 / 1024

    print(f"✓ Merged dataset saved: {output_path}")
    print(f"  Total images: {len(merged_annotations):,}")
    print(f"  Unique classes: {merged_metadata['unique_classes']}")
    print(f"  File size: {file_size:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic handwritten Odia characters')
    parser.add_argument('--missing-classes', type=str,
                       default='4,5,6,7,8,9,36,37,38,39,40,41,42,43,44,45,46',
                       help='Comma-separated list of missing class IDs')
    parser.add_argument('--samples-per-class', type=int, default=3000,
                       help='Number of samples to generate per class')
    parser.add_argument('--output', type=str,
                       default=str(OUTPUT_DIR),
                       help='Output directory for generated images')
    parser.add_argument('--no-merge', action='store_true',
                       help='Do not merge with balanced dataset')

    args = parser.parse_args()

    # Parse missing class IDs
    missing_classes = [int(x.strip()) for x in args.missing_classes.split(',')]

    # Generate synthetic data
    annotations, stats = generate_synthetic_dataset(
        missing_classes,
        samples_per_class=args.samples_per_class,
        output_dir=args.output
    )

    if annotations is None:
        print("❌ Failed to generate synthetic data")
        return

    # Save synthetic annotations
    synthetic_json = BASE_DIR / "annotations" / "synthetic_missing_chars.json"
    save_synthetic_annotations(annotations, stats, synthetic_json)

    # Merge with balanced dataset
    if not args.no_merge:
        balanced_json = BASE_DIR / "annotations" / "balanced_annotations.json"
        merged_json = BASE_DIR / "annotations" / "complete_dataset.json"

        merge_with_balanced_dataset(synthetic_json, balanced_json, merged_json)

    print("\n" + "="*70)
    print("✅ SYNTHETIC DATA GENERATION COMPLETE!")
    print("="*70)
    print("\nNext step: Create train/val/test splits")

if __name__ == "__main__":
    main()
