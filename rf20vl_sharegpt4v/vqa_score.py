import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from tqdm import tqdm
import gc
import argparse


import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForImageTextToText, Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration
from qwen_vl_utils import process_vision_info

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


from PIL import Image, ImageDraw

import json
import time
import re

import numpy as np

import random
from collections import defaultdict
from pathlib import Path
from datetime import datetime


DEBUG_VQA = False
from transformers import pipeline

def load_qwen_model(model_name_or_path: str, enable_lora: bool = False):
    """
    model_name_or_path can be:
      - "Qwen3-VL-8B-Instruct" (will load from Hub as "Qwen/<name>")
      - "/scratch/.../merged_model_dir" (will load locally)
      - "Qwen/Qwen3-VL-8B-Instruct" (explicit repo id)
    """
    # Resolve model id / path
    if os.path.exists(model_name_or_path):
        model_id = model_name_or_path            # local dir
        processor_id = model_name_or_path
        tag_for_logic = Path(model_name_or_path).name
    else:
        # allow either "Qwen/..." or bare "Qwen3-..."
        model_id = model_name_or_path if "/" in model_name_or_path else f"Qwen/{model_name_or_path}"
        processor_id = model_id
        tag_for_logic = model_name_or_path

    dtype = "auto" if "-FP8" in tag_for_logic else torch.bfloat16
    print(f"Loading vLLM from: {model_id}")
    print(f"dtype: {dtype}")

    enable_expert_parallel = True if (
        tag_for_logic.startswith("Qwen3-VL-235B-A22B-Instruct-FP8")
        or tag_for_logic.startswith("Qwen3-VL-30B-A3B-Instruct")
    ) else False
    print(f"enable_expert_parallel: {enable_expert_parallel}")

    tensor_parallel_size = 4 if "Qwen2.5-VL-7B" in tag_for_logic else torch.cuda.device_count()
    print(f"tensor_parallel_size: {tensor_parallel_size}")

    model = LLM(
        model=model_id,                 # <-- key change: local path or repo id
        dtype=dtype,
        trust_remote_code=True,
        gpu_memory_utilization=0.80,
        enforce_eager=False,
        enable_expert_parallel=enable_expert_parallel,
        tensor_parallel_size=tensor_parallel_size,
        seed=0,
        enable_lora=enable_lora,
    )

    processor = AutoProcessor.from_pretrained(processor_id, trust_remote_code=True)

    # keep your padding fixes
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer

    return model, processor

def find_latest_lora_checkpoint(lora_ds_dir: Path) -> Path | None:
    """
    Expect something like:
      /scratch/.../<dataset>/single_instruction/checkpoint-1000/
    Return the checkpoint dir path or None.
    """
    if not lora_ds_dir.exists():
        return None

    # common patterns: checkpoint-xxx
    ckpts = [p for p in lora_ds_dir.glob("checkpoint-*") if p.is_dir()]
    if not ckpts:
        # sometimes adapter is saved directly in the dir
        # (adapter_config.json / adapter_model.safetensors)
        if (lora_ds_dir / "adapter_config.json").exists():
            return lora_ds_dir
        return None

    def step(p: Path):
        m = re.search(r"checkpoint-(\d+)", p.name)
        return int(m.group(1)) if m else -1

    ckpts = sorted(ckpts, key=step)
    best = ckpts[-1]

    # sanity: must have adapter_config.json (typical PEFT format)
    if not (best / "adapter_config.json").exists():
        return None
    return best

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def xywh_to_xyxy(b):
    x, y, w, h = map(float, b)
    return [x, y, x + w, y + h]

def clamp_xyxy(b, W, H):
    x1, y1, x2, y2 = map(float, b)
    x1 = max(0.0, min(x1, W - 1))
    y1 = max(0.0, min(y1, H - 1))
    x2 = max(0.0, min(x2, W - 1))
    y2 = max(0.0, min(y2, H - 1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return [x1, y1, x2, y2]

def model_generate(messages, model, processor):
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    # inputs = processor(text=[text_input], images=image_inputs, padding=True, return_tensors="pt").to(model.device)
    inputs_org = processor(text=[text_input], images=image_inputs, padding=True, return_tensors="pt")

    with torch.no_grad():
        # generated_ids = model.generate(**inputs, max_new_tokens=512)
        # generated_ids = model.generate(**inputs, max_new_tokens=1024)
        output_text = None
        # if(model_name == "Qwen3-VL-2B-Instruct-FP8" or model_name == "Qwen3-VL-235B-A22B-Instruct-FP8"):
            
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        # if video_inputs is not None:
        #     mm_data['video'] = video_inputs

        inputs =  {
            'prompt': text_input,
            'multi_modal_data': mm_data,
        }
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=2048,
            top_k=-1,
            stop_token_ids=[],
        )
        outputs = model.generate(inputs, sampling_params = sampling_params)
        for i, output in enumerate(outputs):
            output_text = output.outputs[0].text

        
    return output_text, inputs_org



# def model_generate_with_scores(conversations, model, processor, max_new_tokens=2):
def model_generate_with_scores(conversations, model, processor, max_new_tokens=1, lora_request=None):
# def model_generate_with_scores(conversations, model, processor, max_new_tokens=5):
    # conversations is a list of message lists

    # Prepare inputs for the model
    text_input = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(conversations)
    # inputs = processor(text=text_input, images=image_inputs, padding=True, return_tensors="pt").to(model.device)
    # inputs = processor(text=text_input, images=image_inputs, padding=True, return_tensors="pt")
    
    with torch.no_grad():
        # generated_ids = model.generate(**inputs, max_new_tokens=512)
        # generated_ids = model.generate(**inputs, max_new_tokens=1024)
        # output_text = None
        # if(model_name == "Qwen3-VL-2B-Instruct-FP8" or model_name == "Qwen3-VL-235B-A22B-Instruct-FP8"):
            
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        # if video_inputs is not None:
        #     mm_data['video'] = video_inputs

        inputs =  {
            'prompt': text_input,
            # 'prompt': conversations,
            'multi_modal_data': mm_data,
        }
        sampling_params = SamplingParams(
            temperature=0,
            # max_tokens=2048,
            max_tokens=max_new_tokens,
            top_k=-1,
            # logprobs=max_new_tokens,   # get top-5 logprobs per token
            logprobs=10,   # get top-5 logprobs per token
            stop_token_ids=[],
        )
        outputs = model.generate(inputs, sampling_params = sampling_params, lora_request=lora_request)
        # for i, output in enumerate(outputs):
        #     output_text = output.outputs[0].text

    # # Generate outputs
    # with torch.inference_mode():
    #     outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, output_scores=True, return_dict_in_generate=True)
    
    
    return outputs

# VQA utils

import numpy as np

def get_masked_image_vqa_scores_with_instructions(qwen_model, qwen_processor, dataset_instructions_json, prompt_list, pil_images: list, batch_size: int = 8, lora_request=None):
    """
    Scores a batch of images with bounding boxes based on a VQA prompt.
    This function is adapted from GridVQAscores_withSavedSAMProposal_webUI_RefCOCO_officialEval_saveInterimResults_gridWeightedBBox.py
    """
    # if not pil_images: return np.array([])
    
    def getDatasetInstructions(dataset_instructions_json, class_name):
        if class_name in dataset_instructions_json:       
            dataset_instructions = dataset_instructions_json[class_name]
        else:

            # Find the matching key ignoring case
            matched_key = next((key for key in dataset_instructions_json.keys() if key.lower() == class_name.lower()), None)
            if matched_key:
                dataset_instructions = dataset_instructions_json[matched_key]
            else:
                #Throw error
                raise ValueError(f"Class name '{class_name}' not found in dataset instructions JSON keys.")
        
        return dataset_instructions



    # def getPrompt(prompt, dataset_instructions_json):

    #     question = f"""
    #         Given the '{prompt}' class defined as follows: {getDatasetInstructions(dataset_instructions_json, prompt)}

    #         Is the main subject or object being referred to as: '{prompt}' located inside the red bounding box in the image? Please answer Yes or No. Note: The object should be entirely inside the bounding box, with no part outside, and it must be the only object present inside - no other objects should appear within the box.
    #     """

    #     return question

    def getPrompt(prompt, dataset_instructions_json):
        question = f"""
            Given the '{prompt}' class defined as follows:
            {getDatasetInstructions(dataset_instructions_json, prompt)}

            Look at the red bounding box. Question: Does this red box contain at least one instance of '{prompt}'?

            Answer **Yes** if a '{prompt}' is clearly present inside the box and the box mostly covers it (it is okay if parts are slightly outside due to motion/occlusion, and it is okay if other objects/people also appear in the box).
            Answer **No** if there is no '{prompt}' in the box, or if the box is mostly on the wrong object/background.

            Please answer with exactly one word: Yes or No.
        """
        return question

        
    # yes_token_id = qwen_processor.tokenizer.encode("Yes")[0]
    # no_token_id = qwen_processor.tokenizer.encode("No")[0]

    all_scores = []

    # chunk into batches
    for start in range(0, len(pil_images), batch_size):
        batch_imgs = pil_images[start:start + batch_size]
        batch_prompts = prompt_list[start:start + batch_size]

        # build a list of independent requests for vLLM
        inputs_list = []
        for img, prompt in zip(batch_imgs, batch_prompts):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": getPrompt(prompt, dataset_instructions_json)}
                ]
            }]
            text_input = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)

            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs

            inputs_list.append({
                "prompt": text_input,
                "multi_modal_data": mm_data,
            })

        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            top_k=-1,
            logprobs=10,
            stop_token_ids=[],
        )

        outputs = qwen_model.generate(inputs_list, sampling_params=sampling_params, lora_request=lora_request)

        # parse each output
        for out in outputs:
            token_logprobs = out.outputs[0].logprobs[0]  # dict[token_id -> TokenLogprob]

            yes_logprob = None
            no_logprob = None
            for _, info in token_logprobs.items():
                t = info.decoded_token
                if t == "Yes" or (yes_logprob is None and t == "yes"):
                    yes_logprob = info.logprob
                if t == "No" or (no_logprob is None and t == "no"):
                    no_logprob = info.logprob

            if yes_logprob is None and no_logprob is None:
                all_scores.append(-1.0)
                continue
            if yes_logprob is None:
                no_prob = torch.exp(torch.tensor(no_logprob)) if no_logprob is not None else torch.tensor(0.0)
                all_scores.append(float((1 - no_prob).item()))
                continue

            yes_prob = torch.exp(torch.tensor(yes_logprob)) if yes_logprob is not None else torch.tensor(0.0)
            no_prob  = torch.exp(torch.tensor(no_logprob)) if no_logprob is not None else torch.tensor(0.0)
            score = yes_prob / (yes_prob + no_prob + 1e-18)
            all_scores.append(float(score.item()))

    return np.array(all_scores)

def create_img_with_bbox(original_image, bbox_xywh):
    """Draws a single red bounding box on an image."""
    img_with_bbox = original_image.copy()
    draw = ImageDraw.Draw(img_with_bbox)
    x, y, w, h = bbox_xywh
    bbox_xyxy = [x, y, x + w, y + h]
    draw.rectangle(bbox_xyxy, outline='red', width=3)
    return img_with_bbox

# Reuse your existing helpers:
# - load_qwen_model
# - get_masked_image_vqa_scores_with_instructions
# - create_img_with_bbox
# - extract_instruction_block
# - parse_class_name_from_user_msg

def build_unique_image_list_and_instruction_map(sharegpt_json_path: Path):
    """
    Returns:
      unique_images: list[str]  # first-seen unique image paths
      instr_map: dict[str, str] # class_name -> instruction_block
      first_entry_by_image: dict[str, dict] # keep one ShareGPT entry per image for "messages" reuse
      class_list_first_seen: list[str] # class names in first-seen order
    """
    data = json.load(open(sharegpt_json_path, "r"))
    seen = set()
    unique_images = []
    instr_map = {}
    first_entry_by_image = {}
    class_list_first_seen = []

    for entry in data:
        if "images" not in entry or not entry["images"]:
            continue
        img_path = entry["images"][0]

        # first-seen unique image list (COCO-style)
        if img_path not in seen:
            seen.add(img_path)
            unique_images.append(img_path)
            first_entry_by_image[img_path] = entry

        # build class -> instruction_block map from user prompt
        if "messages" in entry and entry["messages"] and entry["messages"][0].get("role") == "user":
            user_msg = entry["messages"][0].get("content", "")
            cls = parse_class_name_from_user_msg(user_msg)
            if cls:
                if cls not in instr_map:
                    instr_map[cls] = extract_instruction_block(user_msg)
                if cls not in class_list_first_seen:
                    class_list_first_seen.append(cls)

    return unique_images, instr_map, first_entry_by_image, class_list_first_seen


def load_coco_category_map(coco_gt_json: Path):
    """
    COCO GT json typically has:
      categories: [{"id":1,"name":"Goalie"}, ...]
    """
    gt = json.load(open(coco_gt_json, "r"))
    if "categories" not in gt:
        return None
    return {int(c["id"]): c["name"] for c in gt["categories"]}


def load_coco_image_map(coco_gt_json: Path, image_root: str = None):
    """
    COCO GT json typically has:
      images: [{"id":0,"file_name":"xxx.jpg"}, ...]
    If file_name is relative and you pass image_root, we join it.
    """
    gt = json.load(open(coco_gt_json, "r"))
    if "images" not in gt:
        return None
    out = {}
    for im in gt["images"]:
        iid = int(im["id"])
        fn = im.get("file_name") or im.get("path") or im.get("coco_url")
        if fn is None:
            continue
        if image_root is not None and not str(fn).startswith("/"):
            fn = str(Path(image_root) / fn)
        out[iid] = fn
    return out
def process_pred_file_write_coco(
    pred_json_path: Path,
    sharegpt_test_json_path: Path,
    model,
    processor,
    out_pred_json: Path,
    vqa_batch_size: int = 8,
    coco_gt_json: Path = None,
    image_root: str = None,
    resume: bool = False,
    max_images: int = None,
    lora_request=None
):
    out_pred_json.parent.mkdir(parents=True, exist_ok=True)

    # ShareGPT: instructions + unique images list
    _, instr_map, _, _ = \
        build_unique_image_list_and_instruction_map(sharegpt_test_json_path)

    # Prefer official COCO maps
    coco_cat_map = load_coco_category_map(coco_gt_json) if coco_gt_json else None
    coco_img_map = load_coco_image_map(coco_gt_json, image_root=image_root) if coco_gt_json else None

    preds = json.load(open(pred_json_path, "r"))

    # If resuming, load existing out file and continue from there
    done = set()
    if resume and out_pred_json.exists():
        preds = json.load(open(out_pred_json, "r"))
        done = set(int(p["image_id"]) for p in preds)

    # Group preds by image_id
    preds_by_image = defaultdict(list)
    for idx, p in enumerate(preds):
        iid = int(p["image_id"])
        preds_by_image[iid].append((idx, p))

    # image_id -> image_path
    if coco_img_map is None:
        raise RuntimeError("coco_gt_json must contain images[]; cannot map image_id -> image_path.")
    image_id_to_path = coco_img_map

    if coco_cat_map is None:
        raise RuntimeError("coco_gt_json must contain categories[]; cannot map category_id -> name.")
    cat_id_to_name = coco_cat_map

    processed_images = 0

    for image_id in sorted(preds_by_image.keys()):
        if resume and image_id in done:
            continue
        if max_images is not None and processed_images >= max_images:
            break

        image_path = image_id_to_path.get(image_id, None)
        if image_path is None:
            continue

        try:
            original_image = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        # Build VQA inputs in EXACT SAME ORDER as detections
        vqa_images = []
        vqa_prompts = []
        det_refs = []  # det dict refs (same order)

        dataset_instructions_json = {}

        for _, det in preds_by_image[image_id]:
            bbox_xywh = det["bbox"]
            cid = int(det["category_id"])
            label = cat_id_to_name.get(cid, str(cid))

            vqa_images.append(create_img_with_bbox(original_image, bbox_xywh))
            vqa_prompts.append(label)

            det_refs.append(det)
            dataset_instructions_json[label] = instr_map.get(
                label, f"(No instructions found for class: {label})"
            )

        if not det_refs:
            continue

        vqa_scores = get_masked_image_vqa_scores_with_instructions(
            model, processor, dataset_instructions_json, vqa_prompts, vqa_images, batch_size=vqa_batch_size, lora_request=lora_request
        )
        vqa_scores_list = vqa_scores.tolist() if hasattr(vqa_scores, "tolist") else list(vqa_scores)

        if len(vqa_scores_list) != len(det_refs):
            raise RuntimeError(
                f"[VQA count mismatch] image_id={image_id}: dets={len(det_refs)} vs vqa={len(vqa_scores_list)}"
            )

        # Overwrite score in-place
        for det, s in zip(det_refs, vqa_scores_list):
            det["score"] = float(s)

        processed_images += 1

        # periodic write for safety
        if processed_images % 50 == 0:
            json.dump(preds, open(out_pred_json, "w"), indent=2)

    json.dump(preds, open(out_pred_json, "w"), indent=2)
    print(f"Done. Calibrated {processed_images} images. Wrote: {out_pred_json}")

def parse_class_name_from_user_msg(user_msg: str):
    """
    Extract class name from:
      'Locate all of the following objects: Attack in the image ...'
    """
    m = re.search(r"Locate all of the following objects:\s*(.*?)\s+in the image", user_msg, flags=re.IGNORECASE)
    return m.group(1).strip() if m else None


def extract_instruction_block(user_msg: str) -> str:
    """
    Extract the part after:
      'Use the following annotator instructions to improve detection accuracy:'
    Fallback to entire user_msg if not found.
    """
    m = re.search(r"Use the following annotator instructions.*?:\s*(.*)\Z", user_msg, flags=re.DOTALL)
    return m.group(1).strip() if m else user_msg.strip()


from typing import List, Optional

def discover_datasets_from_dataset_root(dataset_root: Path) -> List[str]:
    # dataset_root/<dataset>/test/_annotations.coco.json
    out = []
    for p in dataset_root.glob("*/test/_annotations.coco.json"):
        out.append(p.parent.parent.name)  # <dataset>
    return sorted(set(out))

def main_pred():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)

    # roots (only what we truly need)
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="e.g. /scratch/siyili/rf20vl-3X/")
    parser.add_argument("--sharegpt_root", type=str, required=True,
                        help="e.g. /home/siyili/LLaMA-Factory/sharegpt4v_datasets-3X/")
    parser.add_argument("--pred_json_root", type=str, required=True,
                        help="dir containing predictions_<dataset>.json")
    parser.add_argument("--sft_lora", type=str, default=None)

    parser.add_argument("--vqa_batch_size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_images", type=int, default=None)

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    sharegpt_root = Path(args.sharegpt_root)
    pred_json_root = Path(args.pred_json_root)

    # hard-coded output dir: "<pred_json_root>_vqa" (same parent)
    out_dir = pred_json_root.parent / f"{pred_json_root.name}_vqa"
    out_dir.mkdir(parents=True, exist_ok=True)

    # always auto-discover from dataset_root
    datasets = discover_datasets_from_dataset_root(dataset_root)
    print(f"Found {len(datasets)} datasets under dataset_root.")

    model, processor = load_qwen_model(args.model_name, enable_lora=args.sft_lora is not None)
    skipped = 0
    processed = 0

    for ds in datasets:
        coco_gt_json = dataset_root / ds / "test" / "_annotations.coco.json"
        image_root   = dataset_root / ds / "test"
        sharegpt_test_json = sharegpt_root / ds / "test" / "by_class_with_description.json"

        pred_json = pred_json_root / f"predictions_{ds}.json"
        out_pred_json = out_dir / f"predictions_{ds}.json"

        # always skip missing
        if not coco_gt_json.exists():
            skipped += 1
            continue
        if not image_root.exists():
            skipped += 1
            continue
        if not sharegpt_test_json.exists():
            skipped += 1
            continue
        if not pred_json.exists():
            skipped += 1
            continue

        print(f"\n=== {ds} ===")
        print("pred_json:", pred_json)
        print("out_pred_json:", out_pred_json)

        lora_request = None
        if args.sft_lora is not None:
            lora_root = Path(args.sft_lora)
            # adjust if your folder name differs (e.g., "single_instruction")
            lora_ds_dir = lora_root / ds / "single_instruction"
            ckpt_dir = find_latest_lora_checkpoint(lora_ds_dir)

            if ckpt_dir is not None:
                # stable adapter_id; any int is fine as long as unique-ish
                adapter_id = abs(hash(ds)) % 1_000_000_000
                lora_request = LoRARequest(lora_name=ds, lora_int_id=adapter_id, lora_path=str(ckpt_dir))
            else:
                # no LoRA for this dataset -> fall back to base
                lora_request = None

        process_pred_file_write_coco(
            pred_json_path=pred_json,
            sharegpt_test_json_path=sharegpt_test_json,
            model=model,
            processor=processor,
            out_pred_json=out_pred_json,
            vqa_batch_size=args.vqa_batch_size,
            coco_gt_json=coco_gt_json,
            image_root=str(image_root),
            resume=args.resume,
            max_images=args.max_images,
            lora_request=lora_request,
        )
        processed += 1

    print(f"\nDone. processed={processed}, skipped={skipped}, out_dir={out_dir}")

if __name__ == "__main__":
    main_pred()
