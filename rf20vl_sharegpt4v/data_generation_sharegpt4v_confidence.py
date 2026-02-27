# change log: 2026-02-08, added annotation resize for qwen3vl
# change log: 2026-02-12, added alias mapping for class names
import json
import os
from typing import Dict, List, Any
from collections import defaultdict
import re
from typing import Optional
from prompt_generation import ALIASES
import random
import math
COORD_BASE = 1000.0

def jitter_with_strength(bbox_gt_xyxy, img_w, img_h, rng, strength: float):
    """
    strength in [0, 1]. maps to shift/scale magnitudes.
    0 => almost no jitter, 1 => strong jitter.
    """
    # tuned to hit roughly IoU ~ [0.5, 1.0] across typical boxes
    shift_frac = 0.02 + 0.22 * strength     # 0.02 .. 0.24
    scale_log  = 0.03 + 0.35 * strength     # 0.03 .. 0.38
    return jitter_bbox_xyxy(
        bbox_gt_xyxy, img_w, img_h, rng,
        shift_frac=shift_frac,
        scale_log=scale_log,
    )    

def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def jitter_bbox_xyxy(
    bbox_xyxy,
    img_w: int,
    img_h: int,
    rng: random.Random,
    shift_frac: float = 0.08,     # center shift up to 8% of w/h
    scale_log: float = 0.12,      # log-scale jitter
    min_size: float = 2.0
):
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    w = max(min_size, x2 - x1)
    h = max(min_size, y2 - y1)
    cx, cy = x1 + 0.5 * w, y1 + 0.5 * h

    dx = rng.uniform(-shift_frac * w, shift_frac * w)
    dy = rng.uniform(-shift_frac * h, shift_frac * h)
    sw = math.exp(rng.uniform(-scale_log, scale_log))
    sh = math.exp(rng.uniform(-scale_log, scale_log))

    nw = max(min_size, w * sw)
    nh = max(min_size, h * sh)

    ncx, ncy = cx + dx, cy + dy

    nx1 = ncx - 0.5 * nw
    ny1 = ncy - 0.5 * nh
    nx2 = ncx + 0.5 * nw
    ny2 = ncy + 0.5 * nh

    # clamp to image bounds
    nx1 = max(0.0, min(img_w - 1.0, nx1))
    ny1 = max(0.0, min(img_h - 1.0, ny1))
    nx2 = max(0.0, min(img_w - 1.0, nx2))
    ny2 = max(0.0, min(img_h - 1.0, ny2))

    # ensure valid
    if nx2 <= nx1:
        nx2 = min(img_w - 1.0, nx1 + min_size)
    if ny2 <= ny1:
        ny2 = min(img_h - 1.0, ny1 + min_size)

    return [nx1, ny1, nx2, ny2]

def clamp(v, lo=0.0, hi=COORD_BASE):
    return max(lo, min(hi, v))

def xyxy_to_1000(xyxy, img_w, img_h, coord_base=1000):
    x1, y1, x2, y2 = map(float, xyxy)
    x1 = round(x1 / img_w * coord_base)
    x2 = round(x2 / img_w * coord_base)
    y1 = round(y1 / img_h * coord_base)
    y2 = round(y2 / img_h * coord_base)

    # clamp
    x1 = max(0, min(coord_base, x1))
    x2 = max(0, min(coord_base, x2))
    y1 = max(0, min(coord_base, y1))
    y2 = max(0, min(coord_base, y2))

    # ensure valid ordering
    if x2 <= x1: x2 = min(coord_base, x1 + 1)
    if y2 <= y1: y2 = min(coord_base, y1 + 1)

    return [int(x1), int(y1), int(x2), int(y2)]

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

def format_detections_response(
    annotations: List[Dict],
    class_mapping: Dict[int, str],
    img_w: int,
    img_h: int,
    coord_base: float = 1000.0,
    extra_min: int = 1,
    extra_max: int = 3,
) -> str:
    detections = []

    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    def _round(x: float) -> float:
        return round(_clip01(x), 4)

    for ann in annotations:
        bbox_gt_xyxy = convert_bbox_to_xyxy(ann["bbox"])  # pixel xyxy

        # deterministic per-annotation RNG
        ann_id = ann.get("id", None)
        if ann_id is None:
            ann_id = hash((ann["image_id"], ann["category_id"], *ann["bbox"])) & 0xFFFFFFFF
        seed = (int(ann["image_id"]) * 1000003 + int(ann_id)) & 0xFFFFFFFF
        rng = random.Random(seed)

        label = class_mapping[ann["category_id"]]

        # total proposals for this object
        extra_k = rng.randint(extra_min, extra_max)  # 1..3
        K = 1 + extra_k

        proposals = []

        # --- Proposal 0: "high confidence" in [0.95, 1.0] ---
        # Use GT box but DO NOT force confidence=1.0; add label smoothing.
        conf0 = _round(rng.uniform(0.95, 1.0))
        proposals.append((bbox_gt_xyxy, conf0))

        # --- Proposals 1..K-1: jittered, confidence derived from IoU, then nudged to [0.5, 1.0] ---
        # Use increasing jitter strength so later ones tend to be worse, but not deterministic tiers.
        for j in range(extra_k):
            # increasing strength, plus randomness so it's not fixed patterns
            base_strength = (j + 1) / (extra_k + 1)      # e.g. 0.33,0.66,0.75
            strength = _clip01(base_strength + rng.uniform(-0.15, 0.15))
            b = jitter_with_strength(bbox_gt_xyxy, img_w, img_h, rng, strength)

            iou = iou_xyxy(b, bbox_gt_xyxy)

            # map IoU -> confidence but avoid overfitting to exact IoU:
            # - clip to [0.5, 1.0]
            # - add tiny noise
            conf = _clip01(iou + rng.uniform(-0.02, 0.02))
            conf = max(0.50, min(1.0, conf))
            proposals.append((b, _round(conf)))

        # Sort to ensure top confidence comes first and overall descending
        proposals.sort(key=lambda x: x[1], reverse=True)

        # Optional: enforce non-increasing strictly (avoid a rare bump due to noise)
        fixed = []
        prev = 1.0
        for b, c in proposals:
            c = min(c, prev)
            prev = c
            fixed.append((b, c))
        proposals = fixed

        for bbox_xyxy, conf in proposals:
            bbox_1000 = xyxy_to_1000(bbox_xyxy, img_w, img_h, coord_base)
            detections.append({
                "bbox_2d": bbox_1000,
                "label": label,
                "confidence": conf,
            })

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
        image_info = image_mapping[image_id]
        img_w, img_h = image_info["width"], image_info["height"]
        response = format_detections_response(annotations, class_mapping, img_w, img_h)

        # response = format_detections_response(annotations, class_mapping)
        
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

def resolve_prompt_key(prompts: dict, class_name: str) -> Optional[str]:
    """
    Match prompt_generation.py keys:
      - lowercase
      - spaces -> hyphens
      - punctuation normalization
      - plural / singular mismatch
    """

    # base normalization
    k = class_name.strip().lower()
    k = re.sub(r"\s+", "-", k)              # spaces -> "-"
    k = re.sub(r"[^a-z0-9\-]+", "-", k)     # any weird chars -> "-"
    k = re.sub(r"-+", "-", k)               # collapse ---
    k = k.strip("-")
    k = ALIASES.get(k, k)

    # candidate variants
    candidates = [
        k,
        k + "s",
        k.rstrip("s"),
    ]

    for ck in candidates:
        if ck in prompts:
            return ck

    return None


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

        # IMPORTANT: use os.path.join, not string concat
        image_url = os.path.join(image_url_base, image_info["file_name"]) if image_url_base else image_info["file_name"]

        # For each class present in this image
        for class_id, annotations in class_annotations.items():
            if class_id not in class_mapping:
                continue

            class_name = class_mapping[class_id]
            prompt_key = resolve_prompt_key(prompts, class_name)

            if prompt_key is None:
                continue
            image_info = image_mapping[image_id]
            img_w, img_h = image_info["width"], image_info["height"]
            response = format_detections_response(annotations, class_mapping, img_w, img_h)
            # response = format_detections_response(annotations, class_mapping)

            label_only_conversation = {
                "messages": [
                    {"content": f"<image>{prompts[prompt_key]['label_only']}", "role": "user"},
                    {"content": response, "role": "assistant"},
                ],
                "images": [image_url],
            }
            label_only_data.append(label_only_conversation)

            with_description_conversation = {
                "messages": [
                    {"content": f"<image>{prompts[prompt_key]['with_description']}", "role": "user"},
                    {"content": response, "role": "assistant"},
                ],
                "images": [image_url],
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
    root_dir = "/scratch/siyili/rf20vl-3X"
    output_base_dir = "sharegpt4v_datasets-3X"

    dataset_configs = []
    
    for dataset in os.listdir(root_dir):
        dataset_configs.append({
            'name': dataset,
            'base_data_path': os.path.join(root_dir, dataset),  # Contains train/, valid/, test/ subdirs
            'base_image_path': os.path.join(root_dir, dataset),  # Contains train/, valid/, test/ subdirs
            'prompts_path': os.path.join(root_dir, dataset, f"{dataset}_prompts.json"),  # Prompts file for this dataset
            'splits': ['train/', 'valid/', 'test/']
        })
    
    
    # Output directory for all generated datasets
    
    # Process all datasets
    stats = process_all_datasets(dataset_configs, output_base_dir)
    
    print(f"\nDataset generation complete!")
    print(f"All datasets saved to: {output_base_dir}")

if __name__ == "__main__":
    main()
