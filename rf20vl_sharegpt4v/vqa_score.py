import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

from vllm import LLM, SamplingParams

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


from transformers import pipeline

def load_qwen_model(model_name_or_path: str):
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
        gpu_memory_utilization=0.90,
        enforce_eager=False,
        enable_expert_parallel=enable_expert_parallel,
        tensor_parallel_size=tensor_parallel_size,
        seed=0,
    )

    processor = AutoProcessor.from_pretrained(processor_id, trust_remote_code=True)

    # keep your padding fixes
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer

    return model, processor



def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def load_sigclip_pipeline():
    # ckpt = "google/siglip2-so400m-patch14-384"
    ckpt = "google/siglip2-base-patch16-naflex"
    pipe = pipeline(model=ckpt, task="zero-shot-image-classification")
    return pipe



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
def model_generate_with_scores(conversations, model, processor, max_new_tokens=1):
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
            logprobs=5,   # get top-5 logprobs per token
            stop_token_ids=[],
        )
        outputs = model.generate(inputs, sampling_params = sampling_params)
        # for i, output in enumerate(outputs):
        #     output_text = output.outputs[0].text

    # # Generate outputs
    # with torch.inference_mode():
    #     outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, output_scores=True, return_dict_in_generate=True)
    
    
    return outputs


# Siglip utils

def rescore_with_sigclip(sigclip_pipe, pil_image, candidate_label):
    output = sigclip_pipe(pil_image, candidate_labels=[candidate_label])
    # print(f"SigClip output: {output} for label: {candidate_label}")
    assert len(output) == 1, "Error: SigClip output length is not 1."

    label_score = output[0]['score']

    return label_score


# VQA utils

import numpy as np

def get_masked_image_vqa_scores_with_instructions(qwen_model, qwen_processor, dataset_instructions_json, prompt_list, pil_images: list, batch_size: int = 8):
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



    def getPrompt(prompt, dataset_instructions_json):

        question = f"""
            Given the '{prompt}' class defined as follows: {getDatasetInstructions(dataset_instructions_json, prompt)}

            Is the main subject or object being referred to as: '{prompt}' located inside the red bounding box in the image? Please answer Yes or No. Note: The object should be entirely inside the bounding box, with no part outside, and it must be the only object present inside - no other objects should appear within the box.
        """

        return question

        
    # yes_token_id = qwen_processor.tokenizer.encode("Yes")[0]
    # no_token_id = qwen_processor.tokenizer.encode("No")[0]

    all_final_scores = []
    # Process images in batches
    for i in range(0, len(pil_images)):
        img = pil_images[i]
        prompt = prompt_list[i]
        
        # Create conversations for the batch
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": getPrompt(prompt, dataset_instructions_json)}]}]
        
        # Generate outputs with scores
        outputs = model_generate_with_scores(messages, qwen_model, qwen_processor)

        # # Calculate 'Yes' probability

        assert len(outputs) == 1, "Error: Expected single output for single input."


        token_logprobs = outputs[0].outputs[0].logprobs[0]  # list[dict]
        # print(f"token_logprobs: {token_logprobs}")


        # Initialize scores
        yes_logprob = None
        no_logprob = None

        # Look through generated tokens and their top_logprobs
        for token_id, token_info in token_logprobs.items():
            # top_logprobs = token_info["top_logprobs"]
            logprob = token_info.logprob
            decoded_token = token_info.decoded_token
            # print(f"Decoded token: {decoded_token}: logprob: {logprob}, token_id: {token_id}")

            if "Yes" == decoded_token:
                yes_logprob = logprob
            if yes_logprob is None and "yes" == decoded_token:
                yes_logprob = logprob

            if "No" == decoded_token:
                no_logprob = logprob
            if no_logprob is None and "no" == decoded_token:
                no_logprob = logprob

        # If neither found, skip
        if yes_logprob is None and no_logprob is None:
            all_final_scores.append(-1.0)
            continue
        if yes_logprob is None:
            no_prob = torch.exp(torch.tensor(no_logprob)) if no_logprob is not None else torch.tensor(0.0)
            yes_prob = 1 - no_prob
            score = yes_prob.item()
            all_final_scores.append(score)
            continue

        # Convert from logprobs to probabilities
        yes_prob = torch.exp(torch.tensor(yes_logprob)) if yes_logprob is not None else torch.tensor(0.0)
        no_prob = torch.exp(torch.tensor(no_logprob)) if no_logprob is not None else torch.tensor(0.0)

        # Normalize to get P(Yes)
        score = yes_prob / (yes_prob + no_prob + 1e-18)
        all_final_scores.append(score.item())
    
    return np.array(all_final_scores)


# Drawing utils


def create_img_with_bbox(original_image, bbox_xywh):
    """Draws a single red bounding box on an image."""
    img_with_bbox = original_image.copy()
    draw = ImageDraw.Draw(img_with_bbox)
    x, y, w, h = bbox_xywh
    bbox_xyxy = [x, y, x + w, y + h]
    draw.rectangle(bbox_xyxy, outline='red', width=3)
    return img_with_bbox

import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import json
import re
import gc
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

# ============================================================
# Assumes these functions are already defined above (from Gautam):
#   - load_qwen_model(model_name)
#   - model_generate(messages, model, processor)
#   - get_masked_image_vqa_scores_with_instructions(qwen_model, qwen_processor, dataset_instructions_json, prompt_list, pil_images, batch_size)
#   - create_img_with_bbox(original_image, bbox_xywh)
# ============================================================

def parse_assistant_content(content: str):
    """
    Parses the JSON-style string from the assistant's response.
    Input examples:
      '[{"bbox_2d": [549, 287, 628, 570], "label": "Attack"}]'
      '```json\n[{"bbox_2d":[...],"label":"Attack"}]\n```'
      'Some prose ... [{"bbox_2d":[...],"label":"Attack"}] ...'
    Returns: list[dict]
    """
    try:
        # Grab the first top-level JSON list if present
        m = re.search(r"\[[\s\S]*\]", content)
        if m:
            return json.loads(m.group(0))
        return json.loads(content)
    except Exception as e:
        print(f"[WARN] Failed to parse content: {content[:120]}... Error: {e}")
        return []


def parse_class_name_from_user_msg(user_msg: str) -> str:
    """
    Extract class name from:
      'Locate all of the following objects: Attack in the image ...'
    """
    m = re.search(r"Locate all of the following objects:\s*(.*?)\s+in the image", user_msg, flags=re.IGNORECASE)
    return m.group(1).strip() if m else "object"


def extract_instruction_block(user_msg: str) -> str:
    """
    Extract the part after:
      'Use the following annotator instructions to improve detection accuracy:'
    Fallback to entire user_msg if not found.
    """
    m = re.search(r"Use the following annotator instructions.*?:\s*(.*)\Z", user_msg, flags=re.DOTALL)
    return m.group(1).strip() if m else user_msg.strip()


def xyxy_to_xywh(bbox_xyxy):
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    return [x1, y1, x2 - x1, y2 - y1]


def clamp_xyxy(bbox_xyxy, W, H):
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    x1 = max(0.0, min(x1, W - 1))
    y1 = max(0.0, min(y1, H - 1))
    x2 = max(0.0, min(x2, W - 1))
    y2 = max(0.0, min(y2, H - 1))
    # Ensure proper ordering
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return [x1, y1, x2, y2]


import json
import os
from pathlib import Path

# ... (Include your previously defined functions: load_qwen_model, create_img_with_bbox, etc.)

def parse_assistant_content(content):
    """
    Parses the JSON-style string from the assistant's response.
    Input: '[{"bbox_2d": [549, 287, 628, 570], "label": "Attack"}]'
    """
    try:
        # Some models output prose before JSON, so we use regex to find the bracketed part
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(content)
    except Exception as e:
        print(f"Failed to parse content: {content[:50]}... Error: {e}")
        return []


    #!/usr/bin/env python3
"""
Compute VQAScore for ALL annotated GT boxes in ShareGPT4V-style datasets.

Input dataset format (per dataset):
  /home/siyili/LLaMA-Factory/sharegpt4v_datasets-3X/<dataset_name>/train/by_class_with_description.json

Each entry contains:
  - entry["images"][0] : image path
  - entry["messages"][0]["content"] : user prompt with class + instructions
  - entry["messages"][1]["content"] : assistant JSON list of GT boxes: [{"bbox_2d":[x1,y1,x2,y2],"label":"Class"}]

This script:
  - loads a Qwen-VL model via vLLM (your load_qwen_model)
  - draws each GT box as a red box
  - asks VQA "Yes/No" using your get_masked_image_vqa_scores_with_instructions
  - saves per-dataset JSONL so it's easy to reconstruct ShareGPT later.

Output per dataset (default):
  <output_root>/<dataset_name>/train/vqascore_gt.jsonl

Resumable:
  --resume will skip sample_idx already present in output.
"""

import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from tqdm import tqdm
from PIL import Image

import numpy as np

# =========================
# EXPECTED to exist (from your provided script)
#   - load_qwen_model(model_name) -> (model, processor)
#   - get_masked_image_vqa_scores_with_instructions(qwen_model, qwen_processor, dataset_instructions_json, prompt_list, pil_images, batch_size)
#   - create_img_with_bbox(original_image, bbox_xywh)  # expects xywh
# =========================


def parse_assistant_content(content: str) -> List[Dict[str, Any]]:
    """Parse assistant JSON list, tolerant to prose/code fences."""
    try:
        m = re.search(r"\[[\s\S]*\]", content)
        if m:
            return json.loads(m.group(0))
        return json.loads(content)
    except Exception:
        return []


def parse_class_name_from_user_msg(user_msg: str) -> Optional[str]:
    """Extract class name from 'Locate all of the following objects: X in the image'."""
    m = re.search(
        r"Locate all of the following objects:\s*(.*?)\s+in the image",
        user_msg,
        flags=re.IGNORECASE,
    )
    return m.group(1).strip() if m else None


def extract_instruction_block(user_msg: str) -> str:
    """
    Extract the dataset instruction block used by VQA prompt:
      after 'Use the following annotator instructions to improve detection accuracy:'
    If not found, return full user_msg.
    """
    m = re.search(
        r"Use the following annotator instructions.*?:\s*(.*)\Z",
        user_msg,
        flags=re.DOTALL,
    )
    return m.group(1).strip() if m else user_msg.strip()


def clamp_xyxy(b: List[float], W: int, H: int) -> List[float]:
    x1, y1, x2, y2 = map(float, b)
    x1 = max(0.0, min(x1, W - 1))
    y1 = max(0.0, min(y1, H - 1))
    x2 = max(0.0, min(x2, W - 1))
    y2 = max(0.0, min(y2, H - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def xyxy_to_xywh(b: List[float]) -> List[float]:
    x1, y1, x2, y2 = map(float, b)
    return [x1, y1, x2 - x1, y2 - y1]

def scale_xyxy_from_resized(b_xyxy, W, H, resized_size=1000):
    """Map bbox from resized_size×resized_size coords back to original (W,H)."""
    x1, y1, x2, y2 = map(float, b_xyxy)
    sx = W / float(resized_size)
    sy = H / float(resized_size)
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]

def load_existing_indices(jsonl_path: Path) -> set:
    """Read existing JSONL and collect processed (sample_idx) to enable resume."""
    done = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "sample_idx" in obj:
                    done.add(int(obj["sample_idx"]))
            except Exception:
                continue
    return done


def process_dataset_file(
    ds_json_path: Path,
    model,
    processor,
    output_jsonl: Path,
    vqa_batch_size: int,
    resume: bool,
    max_samples: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Compute VQAScore for all GT boxes in one dataset file and append to JSONL.

    Returns:
      (num_samples_written, num_boxes_scored)
    """
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # resume support
    done_indices = load_existing_indices(output_jsonl) if resume else set()

    with ds_json_path.open("r") as f:
        data = json.load(f)

    written_samples = 0
    scored_boxes_total = 0

    # open output in append mode for streaming writes
    out_f = output_jsonl.open("a")

    try:
        for idx, entry in enumerate(tqdm(data, desc=f"{ds_json_path.parent.parent.name}/train")):
            if max_samples is not None and idx >= max_samples:
                break
            if resume and idx in done_indices:
                continue

            # Basic validation
            if "images" not in entry or not entry["images"]:
                continue
            if "messages" not in entry or len(entry["messages"]) < 2:
                continue

            image_path = entry["images"][0]
            user_msg = entry["messages"][0].get("content", "")
            assistant_msg = entry["messages"][1].get("content", "")

            gt_boxes = parse_assistant_content(assistant_msg)
            if not gt_boxes:
                # still write a record (optional). We'll skip to keep file compact.
                continue

            # Determine instruction map for VQA function
            class_name_from_prompt = parse_class_name_from_user_msg(user_msg)
            instr_block = extract_instruction_block(user_msg)

            # Load image once
            try:
                original_image = Image.open(image_path).convert("RGB")
            except Exception:
                # image missing/corrupt
                continue

            W, H = original_image.size

            # Build per-box images + prompts
            vqa_images = []
            vqa_prompts = []
            cleaned_boxes = []

            for box in gt_boxes:
                if "bbox_2d" not in box:
                    continue
                label = box.get("label", class_name_from_prompt or "object")
                bbox_xyxy = box["bbox_2d"]

                # GT: scale from 1000x1000 coord space back to original pixels
                bbox_xyxy = scale_xyxy_from_resized(bbox_xyxy, W, H, resized_size=1000)

                # Clamp + convert to xywh for your create_img_with_bbox
                bbox_xyxy = clamp_xyxy(bbox_xyxy, W, H)
                bbox_xywh = xyxy_to_xywh(bbox_xyxy)

                try:
                    img_with_box = create_img_with_bbox(original_image, bbox_xywh)
                except Exception:
                    continue

                vqa_images.append(img_with_box)
                vqa_prompts.append(label)
                cleaned_boxes.append(
                    {
                        "bbox_2d": [float(x) for x in bbox_xyxy],
                        "bbox_xywh": [float(x) for x in bbox_xywh],
                        "label": label,
                    }
                )

            if not cleaned_boxes:
                continue

            # dataset_instructions_json expected mapping class->instructions
            # We provide instructions block for ALL labels encountered (safe fallback).
            # If a label differs from the class name in prompt, still map it to the same instruction block;
            # this matches your current VQA prompt design (instructions per class).
            dataset_instructions_json = {cb["label"]: instr_block for cb in cleaned_boxes}
            if class_name_from_prompt is not None:
                dataset_instructions_json[class_name_from_prompt] = instr_block

            # Score (note: your function ignores batch_size internally right now, but we pass it anyway)
            try:
                vqa_scores = get_masked_image_vqa_scores_with_instructions(
                    model,
                    processor,
                    dataset_instructions_json,
                    vqa_prompts,
                    vqa_images,
                    batch_size=vqa_batch_size,
                )
            except Exception:
                continue

            if isinstance(vqa_scores, np.ndarray):
                vqa_scores_list = vqa_scores.tolist()
            else:
                vqa_scores_list = list(vqa_scores)

            # Build output record: keep everything needed to reconstruct ShareGPT later
            record = {
                "dataset": ds_json_path.parent.parent.name,  # <dataset_name>
                "split": "train",
                "sample_idx": idx,
                "images": entry.get("images", []),
                "gt_boxes": cleaned_boxes,
                "vqa_score": [
                    {
                        "bbox_2d": cb["bbox_2d"],
                        "label": cb["label"],
                        "score": float(s),
                    }
                    for cb, s in zip(cleaned_boxes, vqa_scores_list)
                ],
            }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            written_samples += 1
            scored_boxes_total += len(cleaned_boxes)

    finally:
        out_f.close()

    return written_samples, scored_boxes_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--dataset_root", type=str, default="/home/siyili/LLaMA-Factory/sharegpt4v_datasets/")
    parser.add_argument("--output_root", type=str, default="/scratch/siyili/vqascore/")
    parser.add_argument("--split", type=str, default="train", choices=["train","test","valid"])
    parser.add_argument("--dataset_name", type=str, default=None, help="If set, only process this dataset folder name.")
    parser.add_argument("--vqa_batch_size", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="Skip sample_idx already present in output JSONL.")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    # 1) Load Qwen model once
    model, processor = load_qwen_model(args.model_name)

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)

    # 2) Find dataset files
    if args.dataset_name:
        ds_file = dataset_root / args.dataset_name / args.split / "by_class_with_description.json"
        dataset_files = [ds_file] if ds_file.exists() else []
    else:
        dataset_files = list(dataset_root.rglob(f"{args.split}/by_class_with_description.json"))

    if not dataset_files:
        raise FileNotFoundError(f"No dataset files found under {dataset_root} matching */{args.split}/by_class_with_description.json")

    total_written = 0
    total_boxes = 0

    for ds_path in dataset_files:
        dataset_name = ds_path.parent.parent.name
        out_jsonl = output_root / dataset_name / args.split / "vqascore_gt.jsonl"

        print(f"\n=== Processing dataset: {dataset_name}")
        print(f"Input:  {ds_path}")
        print(f"Output: {out_jsonl}")

        written, boxes = process_dataset_file(
            ds_path,
            model,
            processor,
            out_jsonl,
            vqa_batch_size=args.vqa_batch_size,
            resume=args.resume,
            max_samples=args.max_samples,
        )

        print(f"Done {dataset_name}: wrote {written} samples, scored {boxes} boxes.")
        total_written += written
        total_boxes += boxes

    print(f"\nALL DONE. Wrote {total_written} samples total, scored {total_boxes} boxes total.")


if __name__ == "__main__":
    main()
