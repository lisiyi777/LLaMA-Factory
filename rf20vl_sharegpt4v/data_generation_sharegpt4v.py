import json
import os
from typing import Dict, List, Any
from collections import defaultdict

def load_coco_annotations(json_path: str) -> Dict:
    """Load COCO format annotations."""
    with open(json_path, 'r') as file:
        return json.load(file)

def load_prompts(prompts_path: str) -> Dict:
    """Load generated prompts."""
    with open(prompts_path, 'r') as file:
        return json.load(file)

def get_class_name_mapping(coco_data: Dict) -> Dict[int, str]:
    """Create mapping from category ID to class name."""
    return {cat['id']: cat['name'] for cat in coco_data['categories'] if cat['supercategory'] != 'none'}

def get_image_info_mapping(coco_data: Dict) -> Dict[int, Dict]:
    """Create mapping from image ID to image info."""
    return {img['id']: img for img in coco_data['images']}

def group_annotations_by_image(coco_data: Dict) -> Dict[int, List]:
    """Group annotations by image ID."""
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)
    return annotations_by_image

def group_annotations_by_image_and_class(coco_data: Dict) -> Dict[int, Dict[int, List]]:
    """Group annotations by image ID and then by class ID."""
    annotations_by_image_class = defaultdict(lambda: defaultdict(list))
    for ann in coco_data['annotations']:
        annotations_by_image_class[ann['image_id']][ann['category_id']].append(ann)
    return annotations_by_image_class

def convert_bbox_to_xyxy(bbox: List[float]) -> List[float]:
    """Convert COCO bbox [x, y, width, height] to [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]

def format_detections_response(annotations: List[Dict], class_mapping: Dict[int, str]) -> str:
    """Format annotations as JSON response - always returns a list."""
    detections = []
    for ann in annotations:
        bbox_xyxy = convert_bbox_to_xyxy(ann['bbox'])
        detection = {
            "bbox_2d": bbox_xyxy,
            "label": class_mapping[ann['category_id']]
        }
        detections.append(detection)
    
    return json.dumps(detections)

def generate_by_image_datasets(coco_data: Dict, prompts: Dict, image_url_base: str = "") -> tuple:
    """Generate by_image datasets (label_only and with_description)."""
    class_mapping = get_class_name_mapping(coco_data)
    image_mapping = get_image_info_mapping(coco_data)
    annotations_by_image = group_annotations_by_image(coco_data)
    
    label_only_data = []
    with_description_data = []
    
    for image_id, annotations in annotations_by_image.items():
        if image_id not in image_mapping:
            continue
            
        image_info = image_mapping[image_id]
        image_url = f"{image_url_base}{image_info['file_name']}" if image_url_base else image_info['file_name']
        
        # Format response
        response = format_detections_response(annotations, class_mapping)
        
        # Label only dataset
        label_only_conversation = {
            "messages": [
                {
                    "content": f"<image>{prompts['overall']['label_only']}",
                    "role": "user"
                },
                {
                    "content": response,
                    "role": "assistant"
                }
            ],
            "images": [image_url]
        }
        label_only_data.append(label_only_conversation)
        
        # With description dataset
        with_description_conversation = {
            "messages": [
                {
                    "content": f"<image>{prompts['overall']['with_description']}",
                    "role": "user"
                },
                {
                    "content": response,
                    "role": "assistant"
                }
            ],
            "images": [image_url]
        }
        with_description_data.append(with_description_conversation)
    
    return label_only_data, with_description_data

def generate_by_class_datasets(coco_data: Dict, prompts: Dict, image_url_base: str = "") -> tuple:
    """Generate by_class datasets (label_only and with_description)."""
    class_mapping = get_class_name_mapping(coco_data)
    image_mapping = get_image_info_mapping(coco_data)
    annotations_by_image_class = group_annotations_by_image_and_class(coco_data)
    
    label_only_data = []
    with_description_data = []
    
    for image_id, class_annotations in annotations_by_image_class.items():
        if image_id not in image_mapping:
            continue
            
        image_info = image_mapping[image_id]
        image_url = f"{image_url_base}{image_info['file_name']}" if image_url_base else image_info['file_name']
        
        # For each class present in this image
        for class_id, annotations in class_annotations.items():
            if class_id not in class_mapping:
                continue
                
            class_name = class_mapping[class_id]
            
            # Skip if no prompts for this class
            if class_name not in prompts:
                continue
            
            # Format response for this specific class
            response = format_detections_response(annotations, class_mapping)
            
            # Label only dataset
            label_only_conversation = {
                "messages": [
                    {
                        "content": f"<image>{prompts[class_name]['label_only']}",
                        "role": "user"
                    },
                    {
                        "content": response,
                        "role": "assistant"
                    }
                ],
                "images": [image_url]
            }
            label_only_data.append(label_only_conversation)
            
            # With description dataset
            with_description_conversation = {
                "messages": [
                    {
                        "content": f"<image>{prompts[class_name]['with_description']}",
                        "role": "user"
                    },
                    {
                        "content": response,
                        "role": "assistant"
                    }
                ],
                "images": [image_url]
            }
            with_description_data.append(with_description_conversation)
    
    return label_only_data, with_description_data

def save_dataset(data: List[Dict], output_path: str):
    """Save dataset to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} samples to {output_path}")

def generate_all_datasets_for_split(coco_path: str, prompts_path: str, output_dir: str, base_path: str, dataset_name: str, split: str):
    """Generate all 4 datasets for a specific dataset split."""
    # Load data
    coco_data = load_coco_annotations(coco_path)
    prompts = load_prompts(prompts_path)
    
    # Create output directory for this dataset/split
    split_output_dir = os.path.join(output_dir, dataset_name, split)
    os.makedirs(split_output_dir, exist_ok=True)
    
    print(f"Generating datasets for {dataset_name}/{split}...")
    
    print("  - by_image datasets...")
    by_image_label_only, by_image_with_description = generate_by_image_datasets(
        coco_data, prompts, base_path
    )
    
    print("  - by_class datasets...")
    by_class_label_only, by_class_with_description = generate_by_class_datasets(
        coco_data, prompts, base_path
    )
    
    # Save datasets
    datasets = {
        "by_image_label_only": by_image_label_only,
        "by_image_with_description": by_image_with_description,
        "by_class_label_only": by_class_label_only,
        "by_class_with_description": by_class_with_description
    }
    
    for dataset_type, dataset_data in datasets.items():
        output_path = os.path.join(split_output_dir, f"{dataset_type}.json")
        save_dataset(dataset_data, output_path)
    
    # Print statistics
    print(f"  Statistics for {dataset_name}/{split}:")
    for name, data in datasets.items():
        print(f"    {name}: {len(data)} samples")
    
    return datasets

def process_all_datasets(dataset_configs: List[Dict], output_base_dir: str):
    """Process multiple datasets with multiple splits."""
    
    print(f"Processing {len(dataset_configs)} datasets...")
    
    total_stats = {}
    
    for config in dataset_configs:
        dataset_name = config['name']
        base_data_path = config['base_data_path']
        base_image_path = config['base_image_path']
        prompts_path = config['prompts_path']
        splits = config.get('splits', ['train', 'valid', 'test'])
        
        print(f"\n{'='*50}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*50}")
        
        total_stats[dataset_name] = {}
        
        for split in splits:
            print(f"\nProcessing split: {split}")
            
            # Construct paths for this split
            coco_path = os.path.join(base_data_path, split, "_annotations.coco.json")
            image_base_path = os.path.join(base_image_path, split)
            
            # Check if files exist
            if not os.path.exists(coco_path):
                print(f"  Warning: COCO file not found: {coco_path}")
                continue
            
            if not os.path.exists(image_base_path):
                print(f"  Warning: Image directory not found: {image_base_path}")
                continue
                
            if not os.path.exists(prompts_path):
                print(f"  Warning: Prompts file not found: {prompts_path}")
                continue
            
            try:
                # Generate datasets for this split
                datasets = generate_all_datasets_for_split(
                    coco_path=coco_path,
                    prompts_path=prompts_path,
                    output_dir=output_base_dir,
                    base_path=image_base_path,
                    dataset_name=dataset_name,
                    split=split
                )
                
                # Store statistics
                total_stats[dataset_name][split] = {
                    name: len(data) for name, data in datasets.items()
                }
                
            except Exception as e:
                print(f"  Error processing {dataset_name}/{split}: {str(e)}")
                continue
    
    # Print overall statistics
    print(f"\n{'='*60}")
    print("OVERALL STATISTICS")
    print(f"{'='*60}")
    
    for dataset_name, dataset_stats in total_stats.items():
        print(f"\n{dataset_name}:")
        for split, split_stats in dataset_stats.items():
            print(f"  {split}:")
            for dataset_type, count in split_stats.items():
                print(f"    {dataset_type}: {count} samples")
    
    return total_stats

def main():
    """Main function to process multiple datasets and splits."""
    import os 
    root_dir = "/home/nperi/Workspace/LLaMA-Factory/data/rf20vl"
    output_base_dir = "sharegpt4v_datasets"

    dataset_configs = []
    
    for dataset in os.listdir(root_dir):
        dataset_configs.append({
            'name': dataset,
            'base_data_path': os.path.join(root_dir, dataset),  # Contains train/, valid/, test/ subdirs
            'base_image_path': os.path.join(root_dir, dataset),  # Contains train/, valid/, test/ subdirs
            'prompts_path': os.path.join(root_dir, dataset, f"{dataset}_prompts.json"),  # Prompts file for this dataset
            'splits': ['train/', 'valid/', 'test/']
        })
    
    ## Configuration for multiple datasets
    #dataset_configs = [
    #    {
    #        'name': 'soda_bottles',
    #        'base_data_path': './soda_bottles',  # Contains train/, valid/, test/ subdirs
    #        'base_image_path': '/data3/cmitra/FSOD/data/soda_bottles',  # Contains train/, valid/, test/ subdirs
    #        'prompts_path': 'soda_bottles_prompts.json',  # Prompts file for this dataset
    #        'splits': ['train/', 'valid/', 'test/']
    #    },
    #    {
    #        'name': 'water_meter_numbers',
    #        'base_data_path': './water_meter_numbers',
    #        'base_image_path': '/data3/cmitra/FSOD/data/water_meter_numbers',
    #        'prompts_path': 'water_meter_numbers_prompts.json',
    #        'splits': ['train/', 'valid/', 'test/']
    #    }
    #]
    
    # Output directory for all generated datasets
    
    # Process all datasets
    stats = process_all_datasets(dataset_configs, output_base_dir)
    
    print(f"\nDataset generation complete!")
    print(f"All datasets saved to: {output_base_dir}")

if __name__ == "__main__":
    main()
