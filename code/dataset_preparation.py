import os
import shutil
import random
import json
from pathlib import Path
import argparse
from collections import defaultdict
import yaml

def load_yolo_classes(data_yaml_path):
    """Load class names from data.yaml file"""
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('names', [])

def split_yolo_dataset(dataset_path, output_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Split YOLO format dataset into train/val/test sets
    
    Args:
        dataset_path: Path to original dataset (should contain images/ and labels/ folders)
        output_path: Path where split dataset will be saved
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set  
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
    """
    
    # Set random seed
    random.seed(seed)
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    # Setup paths
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    # Find images and labels directories
    images_dir = None
    labels_dir = None
    
    # Common Roboflow directory structures
    possible_image_dirs = ['images', 'train/images', 'valid/images', 'test/images']
    possible_label_dirs = ['labels', 'train/labels', 'valid/labels', 'test/labels']
    
    for img_dir in possible_image_dirs:
        if (dataset_path / img_dir).exists():
            images_dir = dataset_path / img_dir
            break
    
    for lbl_dir in possible_label_dirs:
        if (dataset_path / lbl_dir).exists():
            labels_dir = dataset_path / lbl_dir
            break
    
    # If not found, look for any images/labels directories
    if images_dir is None:
        for item in dataset_path.rglob('*'):
            if item.is_dir() and 'image' in item.name.lower():
                images_dir = item
                break
    
    if labels_dir is None:
        for item in dataset_path.rglob('*'):
            if item.is_dir() and 'label' in item.name.lower():
                labels_dir = item
                break
    
    if images_dir is None or labels_dir is None:
        print("Available directories in dataset:")
        for item in dataset_path.iterdir():
            if item.is_dir():
                print(f"  - {item.name}/")
                for subitem in item.iterdir():
                    if subitem.is_dir():
                        print(f"    - {subitem.name}/")
        raise FileNotFoundError(f"Could not find images and labels directories in {dataset_path}")
    
    print(f"Found images directory: {images_dir}")
    print(f"Found labels directory: {labels_dir}")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(f'*{ext}')))
        image_files.extend(list(images_dir.glob(f'*{ext.upper()}')))
    
    print(f"Found {len(image_files)} images")
    
    # Filter images that have corresponding label files
    valid_pairs = []
    missing_labels = []
    
    for image_file in image_files:
        label_file = labels_dir / f"{image_file.stem}.txt"
        if label_file.exists():
            valid_pairs.append((image_file, label_file))
        else:
            missing_labels.append(image_file.name)
    
    if missing_labels:
        print(f"Warning: {len(missing_labels)} images don't have corresponding labels:")
        for missing in missing_labels[:5]:  # Show first 5
            print(f"  - {missing}")
        if len(missing_labels) > 5:
            print(f"  ... and {len(missing_labels) - 5} more")
    
    print(f"Found {len(valid_pairs)} valid image-label pairs")
    
    if len(valid_pairs) == 0:
        raise ValueError("No valid image-label pairs found!")
    
    # Shuffle the pairs
    random.shuffle(valid_pairs)
    
    # Calculate split indices
    total_count = len(valid_pairs)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    test_count = total_count - train_count - val_count  # Remaining goes to test
    
    print(f"\nDataset split:")
    print(f"  Train: {train_count} samples ({train_count/total_count:.1%})")
    print(f"  Val:   {val_count} samples ({val_count/total_count:.1%})")
    print(f"  Test:  {test_count} samples ({test_count/total_count:.1%})")
    
    # Split the data
    train_pairs = valid_pairs[:train_count]
    val_pairs = valid_pairs[train_count:train_count + val_count]
    test_pairs = valid_pairs[train_count + val_count:]
    
    # Create output directories
    splits = {
        'train': train_pairs,
        'val': val_pairs,
        'test': test_pairs
    }
    
    for split_name, pairs in splits.items():
        if len(pairs) == 0:
            continue
            
        split_images_dir = output_path / split_name / 'images'
        split_labels_dir = output_path / split_name / 'labels'
        
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCopying {len(pairs)} files to {split_name} set...")
        
        for image_file, label_file in pairs:
            # Copy image
            shutil.copy2(image_file, split_images_dir / image_file.name)
            # Copy label
            shutil.copy2(label_file, split_labels_dir / label_file.name)
    
    # Copy data.yaml if it exists and create new one for the split dataset
    data_yaml_path = dataset_path / 'data.yaml'
    if data_yaml_path.exists():
        print(f"\nFound data.yaml, creating new configuration...")
        
        # Load original data.yaml
        with open(data_yaml_path, 'r') as f:
            original_data = yaml.safe_load(f)
        
        # Create new data.yaml for split dataset
        new_data = {
            'path': str(output_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': original_data.get('nc', len(original_data.get('names', []))),
            'names': original_data.get('names', [])
        }
        
        # Save new data.yaml
        with open(output_path / 'data.yaml', 'w') as f:
            yaml.dump(new_data, f, default_flow_style=False)
        
        print(f"Created new data.yaml with {new_data['nc']} classes")
    
    # Create class distribution analysis
    analyze_class_distribution(splits, output_path)
    
    print(f"\n✅ Dataset successfully split and saved to: {output_path}")
    return output_path

def analyze_class_distribution(splits, output_path):
    """Analyze and save class distribution across splits"""
    
    print(f"\nAnalyzing class distribution...")
    
    split_stats = {}
    overall_stats = defaultdict(int)
    
    for split_name, pairs in splits.items():
        if len(pairs) == 0:
            continue
            
        class_counts = defaultdict(int)
        total_annotations = 0
        
        for _, label_file in pairs:
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
                            total_annotations += 1
                            overall_stats[class_id] += 1
        
        split_stats[split_name] = {
            'images': len(pairs),
            'annotations': total_annotations,
            'classes': dict(class_counts),
            'avg_annotations_per_image': total_annotations / len(pairs) if pairs else 0
        }
    
    # Print statistics
    print(f"\n{'Split':<10} {'Images':<8} {'Annotations':<12} {'Avg/Image':<10}")
    print("-" * 45)
    
    for split_name, stats in split_stats.items():
        print(f"{split_name:<10} {stats['images']:<8} {stats['annotations']:<12} {stats['avg_annotations_per_image']:<10.2f}")
    
    # Save detailed statistics
    stats_file = output_path / 'dataset_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump({
            'split_statistics': split_stats,
            'overall_class_distribution': dict(overall_stats)
        }, f, indent=2)
    
    print(f"\nDetailed statistics saved to: {stats_file}")

def convert_to_coco_format(yolo_dataset_path, output_path):
    """Convert the split YOLO dataset to COCO format for NanoDet training"""
    
    from PIL import Image
    
    yolo_path = Path(yolo_dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load class names
    data_yaml = yolo_path / 'data.yaml'
    if not data_yaml.exists():
        raise FileNotFoundError("data.yaml not found. Cannot determine class names.")
    
    class_names = load_yolo_classes(data_yaml)
    
    # Create COCO categories
    categories = []
    for i, name in enumerate(class_names):
        categories.append({
            "id": i + 1,  # COCO format starts from 1
            "name": name,
            "supercategory": "road_sign"
        })
    
    # Process each split
    for split in ['train', 'val', 'test']:
        split_path = yolo_path / split
        if not split_path.exists():
            continue
            
        print(f"\nConverting {split} set to COCO format...")
        
        images_dir = split_path / 'images'
        labels_dir = split_path / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"Skipping {split} - missing images or labels directory")
            continue
        
        # Create COCO annotation structure
        coco_data = {
            "info": {
                        "description": "Roadsign Dataset",
                        "version": "1.0",
                        "year": 2025,
                        "contributor": "Cynthia, Priestly",
                        "date_created": "2025-09-15"
                    },
            "licenses": [
                    {"url": "http://creativecommons.org/licenses/by/2.0/","id": 4,"name": "Attribution License"}
            ], 
            "images": [],
            "annotations": [],
            "categories": categories
        }
        
        annotation_id = 1
        
        # Process each image
        image_files = list(images_dir.glob('*'))
        image_files = [f for f in image_files if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
        
        for image_id, image_file in enumerate(image_files, 1):
            # Get image info
            try:
                img = Image.open(image_file)
                width, height = img.size
            except Exception as e:
                print(f"Error reading {image_file}: {e}")
                continue
            
            # Add image info
            image_info = {
                "id": image_id,
                "file_name": image_file.name,
                "width": width,
                "height": height
            }
            coco_data["images"].append(image_info)
            
            # Read corresponding YOLO annotation
            label_file = labels_dir / f"{image_file.stem}.txt"
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            bbox_width = float(parts[3])
                            bbox_height = float(parts[4])
                            
                            # Convert to COCO format (x, y, width, height)
                            x = (x_center - bbox_width / 2) * width
                            y = (y_center - bbox_height / 2) * height
                            w = bbox_width * width
                            h = bbox_height * height
                            
                            # Add annotation
                            annotation = {
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": class_id + 1,  # COCO format starts from 1
                                "bbox": [x, y, w, h],
                                "area": w * h,
                                "iscrowd": 0
                            }
                            coco_data["annotations"].append(annotation)
                            annotation_id += 1
        
        # Save COCO annotation file
        output_file = output_path / f"{split}.json"
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"Saved {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations to {output_file}")
    
    # Copy images to output directory
    print(f"\nCopying images to output directory...")
    for split in ['train', 'val', 'test']:
        src_images = yolo_path / split / 'images'
        dst_images = output_path / 'images' / split
        
        if src_images.exists():
            dst_images.mkdir(parents=True, exist_ok=True)
            for img_file in src_images.glob('*'):
                if img_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                    shutil.copy2(img_file, dst_images / img_file.name)
    
    print(f"\n✅ COCO format dataset created at: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Split Roboflow dataset into train/val/test sets')
    parser.add_argument('dataset_path', type=str, help='Path to the original dataset')
    parser.add_argument('output_path', type=str, help='Path for the output split dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set ratio (default: 0.2)')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--create_coco', action='store_true', help='Also create COCO format annotations for NanoDet')
    
    args = parser.parse_args()
    
    try:
        # Split the dataset
        split_dataset_path = split_yolo_dataset(
            dataset_path=args.dataset_path,
            output_path=args.output_path,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
        
        # Optionally create COCO format
        if args.create_coco:
            coco_output_path = Path(args.output_path).parent / f"{Path(args.output_path).name}_coco"
            convert_to_coco_format(split_dataset_path, coco_output_path)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())