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

def _filter_by_classes(bboxes, cats, keep_set):
    if not keep_set:
        return [], []
    fb, fc = [], []
    for b, c in zip(bboxes, cats):
        if c in keep_set:
            fb.append(b)
            fc.append(c)
    return fb, fc

def _scale_xywh_bboxes(bboxes, orig_shape, target_size):
    """Scale COCO xywh bboxes from orig_shape (H,W,*) to target_size (W,H)."""
    oh, ow = orig_shape[:2]
    tw, th = target_size
    sx = tw / ow
    sy = th / oh
    out = []
    for bx, by, bw, bh in bboxes:
        out.append([bx * sx, by * sy, bw * sx, bh * sy])
    return out

def _clip_and_filter_xywh_bboxes(bboxes, cats, img_w, img_h):
    """Clip to image bounds and remove degenerate boxes."""
    vb, vc = [], []
    for (x, y, w, h), c in zip(bboxes, cats):
        x1 = max(0.0, x)
        y1 = max(0.0, y)
        x2 = min(float(img_w), x + w)
        y2 = min(float(img_h), y + h)
        nw = x2 - x1
        nh = y2 - y1
        if nw > 1.0 and nh > 1.0:  # small threshold to avoid near-zero boxes
            vb.append([x1, y1, nw, nh])
            vc.append(c)
    return vb, vc

def apply_mixup(image, bboxes, category_ids, buffer, output_size=(640, 640)):
    """
    Safe MixUp for partial annotation setting:
    Only keep classes that are annotated in BOTH images (class intersection).
    """
    if len(buffer) < 1:
        return image, bboxes, category_ids

    # pick partner
    idx = random.randint(0, len(buffer) - 1)
    mix_img = buffer[idx]['image']
    mix_bboxes = buffer[idx]['bboxes']
    mix_cats = buffer[idx]['category_ids']

    # class intersection rule
    keep = set(category_ids) & set(mix_cats)
    if not keep:
        return image, bboxes, category_ids

    # resize images to target
    tw, th = output_size
    img1 = cv2.resize(image, (tw, th))
    img2 = cv2.resize(mix_img, (tw, th))

    # blend
    alpha = random.uniform(0.5, 0.8)
    mixed_img = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)

    # scale bboxes into target coords
    b1 = _scale_xywh_bboxes(bboxes, image.shape, output_size)
    c1 = list(category_ids)
    b2 = _scale_xywh_bboxes(mix_bboxes, mix_img.shape, output_size)
    c2 = list(mix_cats)

    # keep only intersection classes
    b1, c1 = _filter_by_classes(b1, c1, keep)
    b2, c2 = _filter_by_classes(b2, c2, keep)

    final_bboxes = b1 + b2
    final_cats = c1 + c2

    # clip + remove degenerate
    final_bboxes, final_cats = _clip_and_filter_xywh_bboxes(final_bboxes, final_cats, tw, th)

    return mixed_img, final_bboxes, final_cats

def apply_mosaic(image, bboxes, category_ids, buffer, output_size=(640, 640)):
    """
    Safe Mosaic for partial annotation setting:
    Combine current image with 3 random buffer images.
    Only keep classes annotated in ALL 4 images (class intersection).
    Output is cropped back to (640,640).
    """
    if len(buffer) < 3:
        return image, bboxes, category_ids

    indices = random.sample(range(len(buffer)), 3)
    samples = [{'image': image, 'bboxes': bboxes, 'category_ids': category_ids}] + \
              [{'image': buffer[i]['image'], 'bboxes': buffer[i]['bboxes'], 'category_ids': buffer[i]['category_ids']} for i in indices]

    # class intersection across all 4
    keep = set(samples[0]['category_ids'])
    for s in samples[1:]:
        keep &= set(s['category_ids'])
    if not keep:
        return image, bboxes, category_ids

    w, h = output_size  # NOTE: output_size is (W,H)
    mosaic_img = np.full((h * 2, w * 2, 3), 114, dtype=np.uint8)

    yc = int(random.uniform(0.5 * h, 1.5 * h))
    xc = int(random.uniform(0.5 * w, 1.5 * w))

    new_bboxes = []
    new_cats = []

    for i, s in enumerate(samples):
        img_part = s['image']
        cur_bboxes = s['bboxes']
        cur_cats = s['category_ids']

        # filter classes BEFORE transform (keeps logic clean)
        cur_bboxes, cur_cats = _filter_by_classes(cur_bboxes, cur_cats, keep)

        oh, ow = img_part.shape[:2]
        img_rs = cv2.resize(img_part, (w, h))  # to (W,H)
        scale_x = w / ow
        scale_y = h / oh

        if i == 0:  # top-left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # top-right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, w * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom-left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, h * 2)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(h, y2a - y1a)
        else:  # i == 3 bottom-right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, w * 2), min(yc + h, h * 2)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)

        mosaic_img[y1a:y2a, x1a:x2a] = img_rs[y1b:y2b, x1b:x2b]

        # offsets in mosaic canvas coordinates
        pad_w = x1a - x1b
        pad_h = y1a - y1b

        for bbox, cat in zip(cur_bboxes, cur_cats):
            bx, by, bw, bh = bbox
            sbx = bx * scale_x
            sby = by * scale_y
            sbw = bw * scale_x
            sbh = bh * scale_y
            nbx = sbx + pad_w
            nby = sby + pad_h
            new_bboxes.append([nbx, nby, sbw, sbh])
            new_cats.append(cat)

    # ---- final crop back to (w,h) ----
    # crop window centered around (xc, yc) with boundary clipping
    x0 = int(xc - w / 2)
    y0 = int(yc - h / 2)
    x0 = max(0, min(x0, 2 * w - w))
    y0 = max(0, min(y0, 2 * h - h))

    final_img = mosaic_img[y0:y0 + h, x0:x0 + w].copy()

    # shift boxes by crop origin
    shifted_bboxes = []
    for (x, y, bw, bh) in new_bboxes:
        shifted_bboxes.append([x - x0, y - y0, bw, bh])

    # clip + filter degenerate
    final_bboxes, final_cats = _clip_and_filter_xywh_bboxes(shifted_bboxes, new_cats, w, h)
    return final_img, final_bboxes, final_cats

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

        image_buffer = []
        MAX_BUFFER_SIZE = 20
        
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

                image_buffer.append({'image': image, 'bboxes': bboxes, 'category_ids': category_ids})
                if len(image_buffer) > MAX_BUFFER_SIZE:
                    image_buffer.pop(0)
                    
                final_image = image.copy()
                final_bboxes = copy.deepcopy(bboxes)
                final_cats = copy.deepcopy(category_ids)
                
                if epoch >= no_aug_epochs:
                    # Use buffer excluding the current image (just appended) to avoid self-sampling duplicates
                    candidates = image_buffer[:-1]
                    # mosaic / mixup first
                    if random.random() < -0.8 and len(candidates) >= 3:
                        final_image, final_bboxes, final_cats = apply_mosaic(
                            final_image, final_bboxes, final_cats, candidates, output_size=target_size
                        )
                    elif random.random() < -0.8 and len(candidates) >= 1:
                        final_image, final_bboxes, final_cats = apply_mixup(
                            final_image, final_bboxes, final_cats, candidates, output_size=target_size
                        )
                        
                    final_bboxes, final_cats = _clip_and_filter_xywh_bboxes(
                        final_bboxes,
                        final_cats,
                        final_image.shape[1],
                        final_image.shape[0],
                    )
                    try:
                        transformed = transform_pipeline(
                            image=final_image,
                            bboxes=final_bboxes,
                            category_ids=final_cats
                        )
                    except Exception as e:
                        print("\n=== Albumentations crash ===")
                        print("dataset:", dataset, "epoch:", epoch)
                        print("image_filename:", image_filename)
                        print("image_path:", image_path)
                        print("final_image shape:", final_image.shape)
                        print("num bboxes:", len(final_bboxes))
                        for i, (b, c) in enumerate(zip(final_bboxes, final_cats)):
                            print(f"bbox[{i}]={b} cat={c}")
                        # save debug image with boxes overlaid BEFORE transform
                        dbg = final_image.copy()
                        for (x, y, w, h) in final_bboxes:
                            x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
                            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0,255,0), 2)
                        dbg_path = os.path.join(output_dir, dataset, "train", f"DEBUG_fail_{os.path.splitext(image_filename)[0]}_ep{epoch}.jpg")
                        cv2.imwrite(dbg_path, dbg)
                        print("saved debug image:", dbg_path)
                        raise

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
