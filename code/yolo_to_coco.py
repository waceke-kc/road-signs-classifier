import json
import os
from PIL import Image
import cv2

def yolo_to_coco(yolo_annotations_dir, images_dir, output_json):
    """Convert YOLO format annotations to COCO format"""
    
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Define your road sign categories
    categories = [
        {"id": 0, "name": "tunnel"},
        {"id": 1, "name": "Speedlimit_50"},
        {"id": 2, "name": "Speedlimit_100"},
        {"id": 3, "name": "intersection"},
        {"id": 4, "name": "parking"},
        {"id": 5, "name": "right"},
        {"id": 6, "name": "left"},
        {"id": 7, "name": "construction"},
        {"id": 8, "name": "stop"},
        {"id": 9, "name": "traffic_light"}
    ]
    
    coco_format["categories"] = categories
    
    annotation_id = 1
    
    for image_id, filename in enumerate(os.listdir(images_dir)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        # Get image info
        img_path = os.path.join(images_dir, filename)
        img = Image.open(img_path)
        width, height = img.size
        
        image_info = {
            "id": image_id + 1,
            "file_name": filename,
            "width": width,
            "height": height
        }
        coco_format["images"].append(image_info)
        
        # Read corresponding YOLO annotation
        txt_filename = filename.rsplit('.', 1)[0] + '.txt'
        txt_path = os.path.join(yolo_annotations_dir, txt_filename)
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0]) + 1  # COCO format starts from 1
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        bbox_width = float(parts[3])
                        bbox_height = float(parts[4])
                        
                        # Convert to COCO format (x, y, width, height)
                        x = (x_center - bbox_width / 2) * width
                        y = (y_center - bbox_height / 2) * height
                        w = bbox_width * width
                        h = bbox_height * height
                        
                        annotation = {
                            "id": annotation_id,
                            "image_id": image_id + 1,
                            "category_id": class_id,
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0
                        }
                        coco_format["annotations"].append(annotation)
                        annotation_id += 1
    
    # Save COCO format annotation
    with open(output_json, 'w') as f:
        json.dump(coco_format, f, indent=2)

# Usage
yolo_to_coco('data/Dataset/train/labels', 'data/Dataset/train/images', 'annotations.json')
#yolo_to_coco('path/to/yolo/labels', 'path/to/images', 'annotations.json')