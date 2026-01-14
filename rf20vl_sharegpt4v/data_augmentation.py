import os
import cv2
import numpy as np
import albumentations as A
import argparse
import json
import random
import copy
from tqdm import tqdm
from pathlib import Path
import shutil

def get_albumentations_pipeline(width=640, height=640):
    """
    Maps the user's requested list to Albumentations classes.
    Updated for Albumentations >= 1.4.0 compatibility.
    """
    return A.Compose([
        A.RandomResizedCrop(size=(height, width), scale=(0.5, 1.0), p=0.5),
        A.Resize(height=height, width=width, p=0.8),
        A.HorizontalFlip(p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=10, 
            sat_shift_limit=30, 
            val_shift_limit=30, 
            p=0.5
        ),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], min_visibility=0.1))

def get_annotations(coco_json):
    res = {}
    for annotation in coco_json["annotations"]:
        image_id = annotation["image_id"]
        if image_id not in res:
            res[image_id] = []
        res[image_id].append(annotation)
    return res 

def visualize_dataset(json_path, image_dir, output_vis_dir, num_samples=10):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    os.makedirs(output_vis_dir, exist_ok=True)
    
    cat_map = {cat['id']: cat['name'] for cat in data.get('categories', [])}

    img_to_anns = {}
    for ann in data['annotations']:
        iid = ann['image_id']
        if iid not in img_to_anns: img_to_anns[iid] = []
        img_to_anns[iid].append(ann)
        
    all_images = data['images']
    if len(all_images) > num_samples:
        images = random.sample(all_images, num_samples)
    else:
        images = all_images
    
    for img_info in images:
        file_name = img_info['file_name']
        img_path = os.path.join(image_dir, file_name)
            
        image = cv2.imread(img_path)        
        anns = img_to_anns.get(img_info['id'], [])
        
        for ann in anns:
            x, y, w, h = map(int, ann['bbox'])
            cat_id = ann['category_id']
            
            class_name = cat_map.get(cat_id, str(cat_id))            
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)            
            label = f"{class_name}"
            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x, y - 20), (x + w_text, y), (0, 255, 0), -1)
            cv2.putText(image, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        cv2.imwrite(os.path.join(output_vis_dir, f"vis_{file_name}"), image)

def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    num_epochs = args.num_epochs
    no_aug_epochs = args.no_aug_epochs
    target_size = (640, 640)
    
    os.makedirs(output_dir, exist_ok=True)
    
    transform_pipeline = get_albumentations_pipeline(width=target_size[1], height=target_size[0])
    
    for dataset in tqdm(os.listdir(input_dir), desc="Datasets"):
        shutil.copytree(os.path.join(input_dir, dataset, "valid"), os.path.join(output_dir, dataset, "valid"), dirs_exist_ok=True)
        shutil.copytree(os.path.join(input_dir, dataset, "test"), os.path.join(output_dir, dataset, "test"), dirs_exist_ok=True)
        shutil.copyfile(os.path.join(input_dir, dataset, "README.dataset.txt"), os.path.join(output_dir, dataset, "README.dataset.txt"))
            
        dataset_path = os.path.join(input_dir, dataset)
        if not os.path.isdir(dataset_path): continue

        train_dir = os.path.join(dataset_path, "train")
        anno_file = os.path.join(train_dir, "_annotations.coco.json")
        
        os.makedirs(os.path.join(output_dir, dataset, "train"), exist_ok=True)
        
        with open(anno_file, 'r') as f:
            train_json = json.load(f)

        augmented_json = {
            "info": train_json.get("info", {}),
            "licenses": train_json.get("licenses", []),
            "images": [],
            "annotations": [],
            "categories": train_json["categories"]
        }
        
        annotations_dict = get_annotations(train_json)
        
        # ID Counters
        new_img_id = 0
        new_ann_id = 0
        
        for epoch in range(num_epochs):            
            for image_info in train_json["images"]:
                image_id = image_info["id"]
                image_filename = image_info["file_name"]
                image_path = os.path.join(train_dir, image_filename)
                
                image = cv2.imread(image_path)
                
                anns = annotations_dict.get(image_id, [])
                bboxes = [a['bbox'] for a in anns]
                category_ids = [a['category_id'] for a in anns]

                final_image = image.copy()
                final_bboxes = copy.deepcopy(bboxes)
                final_cats = copy.deepcopy(category_ids)
                
                if epoch > no_aug_epochs:
                    transformed = transform_pipeline(
                        image=final_image, 
                        bboxes=final_bboxes, 
                        category_ids=final_cats
                    )
                    final_image = transformed['image']
                    final_bboxes = transformed['bboxes']
                    final_cats = transformed['category_ids']
                  
                new_filename = f"{os.path.splitext(image_filename)[0]}_ep{epoch}.jpg"
                output_path = os.path.join(output_dir, dataset, "train", new_filename)
                cv2.imwrite(output_path, final_image)
                
                augmented_json["images"].append({
                    "id": new_img_id,
                    "file_name": new_filename,
                    "height": final_image.shape[0],
                    "width": final_image.shape[1]
                })
                
                for bbox, cat_id in zip(final_bboxes, final_cats):
                    if isinstance(bbox, np.ndarray):
                        bbox = bbox.tolist()
                    
                    bbox = [float(x) for x in bbox]
                    area = bbox[2] * bbox[3]
                    
                    augmented_json["annotations"].append({
                        "id": int(new_ann_id),           
                        "image_id": int(new_img_id),     
                        "category_id": int(cat_id),      
                        "bbox": bbox,                    
                        "area": float(area),             
                        "iscrowd": 0
                    })
                    new_ann_id += 1
                
                new_img_id += 1
        
        output_json_path = os.path.join(output_dir, dataset, "train", "_annotations.coco.json")
        with open(output_json_path, "w") as f:
            json.dump(augmented_json, f)
        
        if args.visualize:
            visualize_dataset(output_json_path, os.path.join(output_dir, dataset, "train"), 
                              os.path.join(output_dir, dataset, "visualize"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Augmentation for Image Dataset")
    parser.add_argument("--input_dir", default="/home/nperi/Workspace/LLaMA-Factory/data/rf20vl")
    parser.add_argument("--output_dir", default="/home/nperi/Workspace/LLaMA-Factory/data/rf20vl_augmented")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--no_aug_epochs", type=int, default=1)
    parser.add_argument("--visualize", action='store_true', help="Visualize output samples")
    
    args = parser.parse_args()
    main(args)
