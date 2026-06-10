import os
import re
import sys
import gc
import json
import glob
import time
import heapq
import random
import logging
import argparse
import math
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_V1_ENABLED"] = "0"

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info, smart_resize

from utils.qwen_eval_utils import *
from utils.shared_eval_utils import *

logger = logging.getLogger(__name__)

DATASET_COST = {}
_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

EVAL_MODE_TO_FILENAME = {
    "by_class_label_only": "by_class_label_only.json",
    "by_class_with_description": "by_class_with_description.json",
    "by_image_label_only": "by_image_label_only.json",
    "by_image_with_description": "by_image_with_description.json",
}

EVAL_MODE_TO_LORA_SUBDIR = {
    "by_class_label_only": "single_class",
    "by_class_with_description": "single_instruction",
    "by_image_label_only": "multi_class",
    "by_image_with_description": "multi_instruction",
}

def set_seed(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_cuda_visible_devices(raw: str):
    if raw is None:
        return None
    raw = str(raw).strip()
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip() != ""]


def _visible_gpu_ids_from_env_or_torch():
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        ids = _parse_cuda_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
        return ids if ids is not None else []
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return [str(i) for i in range(n)]


def _get_worker_rank_and_world():
    try:
        rank = int(os.environ.get("EVAL_WORKER_RANK", "-1"))
    except Exception:
        rank = -1
    try:
        world = int(os.environ.get("EVAL_NUM_WORKERS", "1"))
    except Exception:
        world = 1
    if world <= 0:
        world = 1
    return rank, world


def _is_worker_process():
    rank, _ = _get_worker_rank_and_world()
    return rank >= 0


def _args_without_auto_parallel(argv):
    out = []
    for a in argv:
        if a == "--parallel":
            continue
        out.append(a)
    return out


def _spawn_auto_parallel_workers(script_path: str, child_args, visible_gpu_ids):
    import subprocess

    n = len(visible_gpu_ids)
    if n <= 0:
        raise RuntimeError("No visible GPUs detected; cannot use --parallel.")

    procs = []
    for rank, gpu_id in enumerate(visible_gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["EVAL_WORKER_RANK"] = str(rank)
        env["EVAL_NUM_WORKERS"] = str(n)
        cmd = [sys.executable, script_path] + list(child_args)
        procs.append((rank, gpu_id, subprocess.Popen(cmd, env=env)))

    exit_codes = {}
    for rank, gpu_id, p in procs:
        code = p.wait()
        exit_codes[rank] = code
        if code != 0:
            logger.error(f"[parallel] worker rank={rank} gpu_id={gpu_id} exited with code={code}")

    if any(c != 0 for c in exit_codes.values()):
        raise SystemExit(1)


def split_datasets_by_cost(
    all_dataset_dirs: List[str],
    dataset_cost: Dict[str, int],
    worker_rank: int,
    num_workers: int,
    *,
    default_cost: Optional[int] = None,
    stable_tiebreak: bool = True,
) -> Tuple[List[str], List[Dict]]:
    """
    Greedy LPT bin-packing split across workers.
    """
    assert num_workers > 0
    assert 0 <= worker_rank < num_workers

    if default_cost is None:
        if dataset_cost:
            vals = sorted(dataset_cost.values())
            default_cost = vals[len(vals) // 2]
        else:
            default_cost = 1

    items = []
    for d in all_dataset_dirs:
        name = os.path.basename(d)
        cost = int(dataset_cost.get(name, default_cost))
        items.append((name, d, cost))

    if stable_tiebreak:
        items.sort(key=lambda x: (-x[2], x[0]))
    else:
        items.sort(key=lambda x: -x[2])

    heap = [(0, w) for w in range(num_workers)]
    heapq.heapify(heap)

    assigned_dirs = [[] for _ in range(num_workers)]
    assigned_named = [[] for _ in range(num_workers)]

    for name, d, cost in items:
        total, w = heapq.heappop(heap)
        assigned_dirs[w].append(d)
        assigned_named[w].append((name, cost))
        heapq.heappush(heap, (total + cost, w))

    plan = []
    for w in range(num_workers):
        plan.append({
            "worker": w,
            "total_cost": sum(c for _, c in assigned_named[w]),
            "datasets": assigned_named[w],
        })

    return assigned_dirs[worker_rank], plan


def _uses_smart_resize(model_name: str) -> bool:
    name = str(model_name).lower()
    return ("qwen2_5vl" in name) or ("qwen2.5" in name and "vl" in name) or ("qwen2_5" in name and "vl" in name)


def load_qwen_vllm_model(model_path, tensor_parallel_size=1, max_model_len=16384, enable_lora=False):
    from vllm import LLM

    model = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        enable_lora=enable_lora,
        max_loras=4 if enable_lora else 0,
        max_lora_rank=64 if enable_lora else 0,
        seed=0,
    )

    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer
    return model, processor


def get_sharegpt_test_file(sharegpt_root: str, dataset_name: str, eval_mode: str) -> str:
    if eval_mode not in EVAL_MODE_TO_FILENAME:
        raise ValueError(f"Unknown eval_mode={eval_mode}")

    path = os.path.join(
        os.path.expanduser(sharegpt_root),
        dataset_name,
        "test",
        EVAL_MODE_TO_FILENAME[eval_mode],
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ShareGPT test file not found for dataset={dataset_name}, eval_mode={eval_mode}: {path}"
        )
    return path


def load_sharegpt_eval_samples(sharegpt_root: str, dataset_name: str, eval_mode: str, logger=None):
    path = get_sharegpt_test_file(sharegpt_root, dataset_name, eval_mode)
    with open(path, "r") as f:
        samples = json.load(f)

    if not isinstance(samples, list):
        raise ValueError(f"Expected a list of ShareGPT samples in {path}, got {type(samples)}")

    if logger:
        logger.info(f"Loaded {len(samples)} ShareGPT samples from {path}")

    return samples, path


def extract_user_prompt_text_from_sharegpt(sample: dict) -> str:
    messages = sample.get("messages", None)
    if not isinstance(messages, list):
        raise ValueError("ShareGPT sample missing valid 'messages' list.")

    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            raise ValueError("User message content is not a string.")

        content = re.sub(r"^\s*<image>\s*", "", content, count=1)
        return content.strip()

    raise ValueError("No user message found in ShareGPT sample.")


def extract_image_path_from_sharegpt(sample: dict) -> str:
    images = sample.get("images", None)
    if not isinstance(images, list) or len(images) == 0:
        raise ValueError("ShareGPT sample missing valid 'images' list.")
    image_path = images[0]
    if not isinstance(image_path, str):
        raise ValueError("ShareGPT sample image path is not a string.")
    return image_path


def _normalize_path_for_match(path_str: str) -> str:
    return os.path.normpath(path_str).replace("\\", "/")


def build_image_lookup_from_coco(test_folder: str, images: list):
    by_abs = {}
    by_rel = {}
    by_name = defaultdict(list)

    for img in images:
        file_name = img["file_name"]
        abs_path = os.path.join(test_folder, file_name)

        by_abs[_normalize_path_for_match(abs_path)] = img
        by_rel[_normalize_path_for_match(file_name)] = img
        by_name[os.path.basename(file_name)].append(img)

    return {
        "by_abs": by_abs,
        "by_rel": by_rel,
        "by_name": by_name,
    }


def match_sharegpt_sample_to_coco_image(sample: dict, image_lookup: dict) -> dict:
    raw_path = extract_image_path_from_sharegpt(sample)
    norm_path = _normalize_path_for_match(raw_path)
    basename = os.path.basename(norm_path)

    if norm_path in image_lookup["by_abs"]:
        return image_lookup["by_abs"][norm_path]

    if norm_path in image_lookup["by_rel"]:
        return image_lookup["by_rel"][norm_path]

    candidates = image_lookup["by_name"].get(basename, [])
    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        f"Could not uniquely match ShareGPT sample image path '{raw_path}' to COCO test image."
    )


def group_sharegpt_samples_by_image(samples: list, image_lookup: dict, logger=None):
    grouped = {}

    for sample in samples:
        image_info = match_sharegpt_sample_to_coco_image(sample, image_lookup)
        image_id = image_info["id"]

        if image_id not in grouped:
            grouped[image_id] = {
                "image_info": image_info,
                "samples": [],
            }
        grouped[image_id]["samples"].append(sample)

    if logger:
        num_images = len(grouped)
        num_samples = sum(len(v["samples"]) for v in grouped.values())
        logger.info(f"Grouped {num_samples} ShareGPT samples into {num_images} images")

        if num_images > 0:
            counts = [len(v["samples"]) for v in grouped.values()]
            logger.info(
                f"Prompt instances per image: min={min(counts)}, mean={sum(counts)/len(counts):.2f}, max={max(counts)}"
            )

    return grouped


def build_messages_from_sharegpt_sample(
    sample: dict,
    actual_image_path: str,
    include_system_prompt: bool = True,
):
    user_text = extract_user_prompt_text_from_sharegpt(sample)
    base64_image = encode_image(actual_image_path)

    final_messages = []

    if include_system_prompt and "SYSTEM_PROMPT" in globals() and SYSTEM_PROMPT:
        final_messages.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        })

    final_messages.append({
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "min_pixels": MIN_PIXELS,
                "max_pixels": MAX_PIXELS,
                "image_url": f"data:image/jpeg;base64,{base64_image}"
            },
            {
                "type": "text",
                "text": user_text
            }
        ],
        "temperature": 0.0
    })

    return final_messages


def build_prompt_instances_for_dataset(
    grouped_samples_by_image: dict,
    test_folder: str,
    model_name: str,
    logger=None,
):
    instances = []

    for image_id, bundle in grouped_samples_by_image.items():
        image_info = bundle["image_info"]
        file_name = image_info["file_name"]
        height = image_info["height"]
        width = image_info["width"]

        image_path = os.path.join(test_folder, file_name)
        if not os.path.exists(image_path):
            matches = glob.glob(os.path.join(test_folder, "**", file_name), recursive=True)
            if not matches:
                raise FileNotFoundError(f"Could not locate image file for {file_name} under {test_folder}")
            image_path = matches[0]

        input_height = 1000
        input_width = 1000
        if _uses_smart_resize(model_name):
            input_height, input_width = smart_resize(
                height, width, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
            )

        for sample in bundle["samples"]:
            final_messages = build_messages_from_sharegpt_sample(
                sample=sample,
                actual_image_path=image_path,
                include_system_prompt=True,
            )

            instances.append({
                "image_id": image_id,
                "file_name": file_name,
                "image_path": image_path,
                "orig_width": width,
                "orig_height": height,
                "input_width": input_width,
                "input_height": input_height,
                "messages": final_messages,
                "sample": sample,
            })

    if logger:
        logger.info(f"Built {len(instances)} prompt instances for dataset")

    return instances


def _build_token_char_spans(token_strs: List[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    cur = 0
    for s in token_strs:
        n = len(s)
        spans.append((cur, cur + n))
        cur += n
    return spans


def _find_matching_bracket(text: str, open_bracket_idx: int) -> Optional[int]:
    if open_bracket_idx < 0 or open_bracket_idx >= len(text) or text[open_bracket_idx] != "[":
        return None
    depth = 0
    for i in range(open_bracket_idx, len(text)):
        c = text[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return i
    return None


def _find_next_bbox_2d_span(text: str, start_idx: int) -> Optional[Tuple[int, int, int, int]]:
    m = re.search(r'("?\s*bbox_2d\s*"?\s*:\s*)\[', text[start_idx:], flags=re.IGNORECASE)
    if not m:
        return None
    match_start = start_idx + m.start()
    bracket_open_idx = start_idx + m.end() - 1
    bracket_close_idx = _find_matching_bracket(text, bracket_open_idx)
    if bracket_close_idx is None:
        return None
    return match_start, bracket_close_idx + 1, bracket_open_idx, bracket_close_idx


def _extract_chosen_token_logprobs(token_ids, logprobs_list):
    if token_ids is None or logprobs_list is None:
        return None
    chosen = []
    for tid, entry in zip(token_ids, logprobs_list):
        lp_val = None
        try:
            if entry is None:
                lp_val = None
            elif isinstance(entry, dict):
                v = entry.get(tid, None)
                if v is None:
                    lp_val = None
                elif hasattr(v, "logprob"):
                    lp_val = float(v.logprob)
                else:
                    lp_val = float(v)
            else:
                v = entry[tid] if hasattr(entry, "__getitem__") else None
                if v is not None and hasattr(v, "logprob"):
                    lp_val = float(v.logprob)
        except Exception:
            lp_val = None
        chosen.append(lp_val)
    return chosen


def _collect_logprobs_for_char_span(
    text: str,
    token_spans: List[Tuple[int, int]],
    token_logprobs: List[Optional[float]],
    start: int,
    end: int,
    *,
    char_filter=None,
) -> List[float]:
    used_logps: List[float] = []
    for ti, (ts, te) in enumerate(token_spans):
        if te <= start or ts >= end:
            continue
        if char_filter is not None:
            overlap_s = max(ts, start)
            overlap_e = min(te, end)
            overlap = text[overlap_s:overlap_e]
            if not any(char_filter(c) for c in overlap):
                continue
        lp = token_logprobs[ti]
        if lp is None:
            continue
        try:
            used_logps.append(float(lp))
        except Exception:
            continue
    return used_logps


def _prepare_token_alignment(
    output_text: str,
    token_strs: List[str],
    token_logprobs: List[Optional[float]],
) -> Tuple[str, List[Tuple[int, int]], List[Optional[float]]]:
    n_tok = min(len(token_strs), len(token_logprobs))
    token_strs = [str(s) for s in token_strs[:n_tok]]
    token_logprobs = token_logprobs[:n_tok]
    joined = "".join(token_strs)
    text = joined if joined else (output_text or "")
    token_spans = _build_token_char_spans(token_strs)
    return text, token_spans, token_logprobs


def compute_bbox_2d_confidences_from_tokens(
    *,
    output_text: str,
    parsed_boxes: List[Dict[str, Any]],
    token_strs: List[str],
    token_logprobs: List[Optional[float]],
) -> List[Optional[float]]:
    text, token_spans, token_logprobs = _prepare_token_alignment(
        output_text=output_text,
        token_strs=token_strs,
        token_logprobs=token_logprobs,
    )
    confidences: List[Optional[float]] = [None] * len(parsed_boxes)
    search_idx = 0
    number_chars = set("0123456789-+.eE")

    for box_i, box in enumerate(parsed_boxes):
        if not isinstance(box, dict):
            confidences[box_i] = None
            continue
        bbox_2d = box.get("bbox_2d", None)
        if not (isinstance(bbox_2d, (list, tuple)) and len(bbox_2d) >= 4):
            confidences[box_i] = None
            continue

        hit = _find_next_bbox_2d_span(text, search_idx)
        if hit is None:
            confidences[box_i] = None
            continue

        _, match_end, b_open, b_close = hit
        inner = text[b_open + 1:b_close]
        nums = list(_NUMBER_RE.finditer(inner))
        if len(nums) < 4:
            confidences[box_i] = None
            search_idx = match_end
            continue

        used_logps: List[float] = []
        for j in range(4):
            nm = nums[j]
            s_abs = (b_open + 1) + nm.start()
            e_abs = (b_open + 1) + nm.end()
            used_logps.extend(
                _collect_logprobs_for_char_span(
                    text,
                    token_spans,
                    token_logprobs,
                    s_abs,
                    e_abs,
                    char_filter=lambda c: c in number_chars,
                )
            )

        if not used_logps:
            confidences[box_i] = None
            search_idx = match_end
            continue

        mean_logp = sum(used_logps) / len(used_logps)
        confidences[box_i] = math.exp(mean_logp)
        search_idx = match_end

    return confidences


def compute_bbox_token_confidences_from_tokens(
    *,
    output_text: str,
    parsed_boxes: List[Dict[str, Any]],
    token_strs: List[str],
    token_logprobs: List[Optional[float]],
) -> List[Optional[float]]:
    text, token_spans, token_logprobs = _prepare_token_alignment(
        output_text=output_text,
        token_strs=token_strs,
        token_logprobs=token_logprobs,
    )
    confidences: List[Optional[float]] = [None] * len(parsed_boxes)
    search_idx = 0

    for box_i, box in enumerate(parsed_boxes):
        if not isinstance(box, dict):
            confidences[box_i] = None
            continue
        bbox_2d = box.get("bbox_2d", None)
        if not (isinstance(bbox_2d, (list, tuple)) and len(bbox_2d) >= 4):
            confidences[box_i] = None
            continue

        hit = _find_next_bbox_2d_span(text, search_idx)
        if hit is None:
            confidences[box_i] = None
            continue

        match_start, match_end, b_open, _ = hit
        key_segment = text[match_start:b_open]
        key_hit = re.search(r"bbox", key_segment, flags=re.IGNORECASE)
        if not key_hit:
            confidences[box_i] = None
            search_idx = match_end
            continue

        key_start = match_start + key_hit.start()
        key_end = match_start + key_hit.end()
        used_logps = _collect_logprobs_for_char_span(
            text,
            token_spans,
            token_logprobs,
            key_start,
            key_end,
            char_filter=lambda c: c.isalpha(),
        )
        if not used_logps:
            confidences[box_i] = None
            search_idx = match_end
            continue

        mean_logp = sum(used_logps) / len(used_logps)
        confidences[box_i] = math.exp(mean_logp)
        search_idx = match_end

    return confidences


def inference_vllm_batch(
    messages_list,
    model,
    processor,
    lora_path=None,
    lora_id=None,
    max_tokens=2048,
    *,
    return_token_data=False,
    logprobs_k=1,
):
    from vllm import SamplingParams

    prompts = []
    mm_datas = []

    t_prep0 = time.perf_counter()
    for messages in messages_list:
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        mm = {}
        if image_inputs is not None:
            mm["image"] = image_inputs
        if video_inputs is not None:
            mm["video"] = video_inputs

        prompts.append(text_input)
        mm_datas.append(mm)
    prepare_extra_wall_s = time.perf_counter() - t_prep0

    inputs_list = [
        {"prompt": p, "multi_modal_data": mm}
        for p, mm in zip(prompts, mm_datas)
    ]

    sp_kwargs = dict(
        temperature=0,
        max_tokens=max_tokens,
        top_k=-1,
        stop_token_ids=[],
    )
    if return_token_data:
        sp_kwargs["logprobs"] = int(logprobs_k)
    sampling_params = SamplingParams(**sp_kwargs)

    lora_request = None
    if lora_path and lora_id:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest(str(lora_id), lora_id, lora_path)

    t0 = time.perf_counter()
    outputs = model.generate(inputs_list, sampling_params=sampling_params, lora_request=lora_request)
    decode_wall_s = time.perf_counter() - t0

    texts = []
    token_data_list = []
    for out in outputs:
        seq = out.outputs[0]
        texts.append(seq.text)
        if return_token_data:
            token_ids = getattr(seq, "token_ids", None)
            logprobs_list = getattr(seq, "logprobs", None)
            chosen_logps = _extract_chosen_token_logprobs(token_ids, logprobs_list)
            token_strs = None
            if token_ids is not None:
                try:
                    token_strs = processor.tokenizer.batch_decode(
                        [[tid] for tid in token_ids],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                except Exception:
                    token_strs = [
                        processor.tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                        for tid in token_ids
                    ]
            token_data_list.append(
                {
                    "token_ids": token_ids,
                    "token_strs": token_strs,
                    "token_logprobs": chosen_logps,
                }
            )

    batch_size = len(messages_list)
    avg_out_len_chars = sum(len(t) for t in texts) / batch_size if batch_size > 0 else 0.0

    if return_token_data:
        return texts, token_data_list, decode_wall_s, prepare_extra_wall_s, batch_size, avg_out_len_chars
    return texts, decode_wall_s, prepare_extra_wall_s, batch_size, avg_out_len_chars


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _normalize_label(label):
    if label is None:
        return ""
    return str(label).strip().lower()


def _rounded_box_key(box, decimals=1):
    bbox = box.get("bbox_2d", None)
    label = _normalize_label(box.get("label", ""))
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    vals = []
    for v in bbox:
        fv = _safe_float(v)
        if fv is None:
            return None
        vals.append(round(fv, decimals))

    return (label, vals[0], vals[1], vals[2], vals[3])


def merge_boxes_for_image(pred_box_lists, dedup_decimals=1):
    merged = []
    seen = set()

    for boxes in pred_box_lists:
        if not isinstance(boxes, list):
            continue
        for box in boxes:
            if not isinstance(box, dict):
                continue
            key = _rounded_box_key(box, decimals=dedup_decimals)
            if key is None or key in seen:
                continue
            seen.add(key)
            merged.append(box)

    return merged


def convert_raw_boxes_to_coco_format(
    raw_boxes,
    image_id,
    original_width,
    original_height,
    input_width,
    input_height,
    categories_dict,
):
    coco_annotations = []

    for idx, box in enumerate(raw_boxes):
        if not isinstance(box, dict):
            continue

        bbox_2d = box.get("bbox_2d", None)
        if not bbox_2d or len(bbox_2d) != 4:
            continue

        label = box.get("label", "unknown")
        label_norm = _normalize_label(label)

        category_id = None
        for cat_id, cat_name in categories_dict.items():
            if _normalize_label(cat_name) == label_norm:
                category_id = cat_id
                break

        if category_id is None:
            for cat_id, cat_name in categories_dict.items():
                cat_norm = _normalize_label(cat_name)
                if cat_norm in label_norm or label_norm in cat_norm:
                    category_id = cat_id
                    break

        if category_id is None:
            continue

        x1, y1, x2, y2 = bbox_2d
        x1 = _safe_float(x1)
        y1 = _safe_float(y1)
        x2 = _safe_float(x2)
        y2 = _safe_float(y2)
        if None in (x1, y1, x2, y2):
            continue

        x1 = x1 / float(input_width) * float(original_width)
        y1 = y1 / float(input_height) * float(original_height)
        x2 = x2 / float(input_width) * float(original_width)
        y2 = y2 / float(input_height) * float(original_height)

        x1 = max(0.0, min(float(original_width), x1))
        y1 = max(0.0, min(float(original_height), y1))
        x2 = max(0.0, min(float(original_width), x2))
        y2 = max(0.0, min(float(original_height), y2))

        x_min = min(x1, x2)
        y_min = min(y1, y2)
        w = max(0.0, abs(x2 - x1))
        h = max(0.0, abs(y2 - y1))

        if w <= 0 or h <= 0:
            continue

        score = box.get("confidence", 1.0)
        score = _safe_float(score)
        if score is None:
            score = 1.0

        coco_annotations.append({
            "id": idx,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [int(round(x_min)), int(round(y_min)), int(round(w)), int(round(h))],
            "area": float(w * h),
            "segmentation": [],
            "iscrowd": 0,
            "score": float(score),
        })

    return coco_annotations


def process_dataset_vllm(
    model,
    processor,
    dataset_dir,
    dataset_name,
    eval_mode,
    sharegpt_root,
    results_dir,
    vis_dir,
    model_name,
    batch_size,
    debug,
    score_mode="constant",
    lora_path=None,
    lora_id=None,
):
    t_dataset0 = time.perf_counter()

    test_folder = os.path.join(dataset_dir, "test")
    annotation_files = glob.glob(os.path.join(test_folder, "*_annotations.coco.json"))
    if not annotation_files:
        raise FileNotFoundError(f"No test annotation file found in {test_folder}")
    annotation_file = annotation_files[0]

    with open(annotation_file, "r") as f:
        annotations = json.load(f)

    all_images = annotations.get("images", [])
    debug_image_ids = None
    if debug:
        debug_images = sorted(
            all_images, key=lambda x: (int(x.get("id", 0)), x.get("file_name", ""))
        )[:15]
        debug_image_ids = {img["id"] for img in debug_images}

    categories = annotations.get("categories", [])
    categories_dict = {cat["id"]: cat["name"] for cat in categories}

    samples, sharegpt_path = load_sharegpt_eval_samples(sharegpt_root, dataset_name, eval_mode, logger)
    image_lookup = build_image_lookup_from_coco(test_folder, all_images)
    grouped = group_sharegpt_samples_by_image(samples, image_lookup, logger)
    if debug_image_ids is not None:
        grouped = {image_id: bundle for image_id, bundle in grouped.items() if image_id in debug_image_ids}

    instances = build_prompt_instances_for_dataset(grouped, test_folder, model_name, logger)

    logger.info(f"[DATASET] dataset={dataset_name} eval_mode={eval_mode}")
    logger.info(f"[DATASET] sharegpt_file={sharegpt_path}")

    raw_boxes_by_image = defaultdict(list)

    timing = {"total": 0.0, "prepare": 0.0, "decode": 0.0, "post": 0.0, "vis": 0.0, "io": 0.0}
    batch_sizes = []
    output_lens = []
    n_attempted = 0
    n_failed = 0

    pending_instances = []

    def flush_batch():
        nonlocal pending_instances, n_attempted, n_failed

        if not pending_instances:
            return

        try:
            messages_list = [inst["messages"] for inst in pending_instances]
            if score_mode == "constant":
                texts, decode_wall_s, prepare_extra_wall_s, bs_used, _ = inference_vllm_batch(
                    messages_list,
                    model,
                    processor,
                    lora_path=lora_path,
                    lora_id=lora_id,
                    max_tokens=2048,
                    return_token_data=False,
                )
                token_data_list = [None] * len(texts)
            else:
                texts, token_data_list, decode_wall_s, prepare_extra_wall_s, bs_used, _ = inference_vllm_batch(
                    messages_list,
                    model,
                    processor,
                    lora_path=lora_path,
                    lora_id=lora_id,
                    max_tokens=2048,
                    return_token_data=True,
                    logprobs_k=1,
                )
            timing["prepare"] += float(prepare_extra_wall_s)
            timing["decode"] += float(decode_wall_s)
            batch_sizes.append(int(bs_used))
            n_attempted += len(pending_instances)

            for text, inst, tok in zip(texts, pending_instances, token_data_list):
                if isinstance(text, str):
                    output_lens.append(len(text))

                t_post0 = time.perf_counter()
                parsed_boxes = parse_qwen_response(text, logger)
                if score_mode != "constant":
                    token_strs = (tok or {}).get("token_strs", None)
                    token_logprobs = (tok or {}).get("token_logprobs", None)
                    confidences = None
                    if token_strs is not None and token_logprobs is not None:
                        try:
                            if score_mode == "bbox_coord_avg":
                                confidences = compute_bbox_2d_confidences_from_tokens(
                                    output_text=text,
                                    parsed_boxes=parsed_boxes,
                                    token_strs=token_strs,
                                    token_logprobs=token_logprobs,
                                )
                            elif score_mode == "bbox_token":
                                confidences = compute_bbox_token_confidences_from_tokens(
                                    output_text=text,
                                    parsed_boxes=parsed_boxes,
                                    token_strs=token_strs,
                                    token_logprobs=token_logprobs,
                                )
                        except Exception as e:
                            logger.warning(f"Failed to compute {score_mode} confidence: {e}")
                            confidences = None

                    for box_idx, box in enumerate(parsed_boxes):
                        if not isinstance(box, dict):
                            continue
                        conf = None
                        if confidences is not None and box_idx < len(confidences):
                            conf = confidences[box_idx]
                        box["confidence"] = float(conf) if conf is not None else 0.0

                timing["post"] += time.perf_counter() - t_post0
                raw_boxes_by_image[inst["image_id"]].append(parsed_boxes)

        except Exception as e:
            n_failed += len(pending_instances)
            logger.exception(f"Batch inference failed for dataset={dataset_name} batch_size={len(pending_instances)}: {e}")
        finally:
            pending_instances = []

    for inst in tqdm(instances, desc=f"Processing {dataset_name}", unit="prompt"):
        pending_instances.append(inst)
        if len(pending_instances) >= batch_size:
            flush_batch()
    flush_batch()

    results_map = {}
    merged_raw_boxes_by_image = {}

    for image_id, box_lists in raw_boxes_by_image.items():
        merged_raw_boxes_by_image[image_id] = merge_boxes_for_image(box_lists, dedup_decimals=1)

    for image_id, bundle in grouped.items():
        image_info = bundle["image_info"]
        merged_raw = merged_raw_boxes_by_image.get(image_id, [])

        input_height = 1000
        input_width = 1000
        if _uses_smart_resize(model_name):
            input_height, input_width = smart_resize(
                image_info["height"],
                image_info["width"],
                min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS,
            )

        coco_anns = convert_raw_boxes_to_coco_format(
            raw_boxes=merged_raw,
            image_id=image_id,
            original_width=image_info["width"],
            original_height=image_info["height"],
            input_width=input_width,
            input_height=input_height,
            categories_dict=categories_dict,
        )
        results_map[image_id] = coco_anns

    final_results_list = []
    for _, ann_list in results_map.items():
        if ann_list:
            final_results_list.extend(ann_list)

    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"predictions_{dataset_name}.json")
    t_io0 = time.perf_counter()
    with open(results_file, "w") as f:
        json.dump(final_results_list, f, indent=2)
    timing["io"] += time.perf_counter() - t_io0

    timing["total"] = time.perf_counter() - t_dataset0

    def _safe_mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    logger.info(
        f"[TimingSummary] dataset={dataset_name} attempted_prompts={n_attempted} failed_prompts={n_failed} "
        f"total_wall_s={timing['total']:.4f} prepare_wall_s={timing['prepare']:.4f} "
        f"decode_wall_s={timing['decode']:.4f} post_wall_s={timing['post']:.4f} io_wall_s={timing['io']:.4f}"
    )
    if batch_sizes:
        logger.info(
            f"[TimingSummary] batch_sizes n_flush={len(batch_sizes)} "
            f"min={min(batch_sizes)} mean={_safe_mean(batch_sizes):.2f} max={max(batch_sizes)}"
        )
    if output_lens:
        logger.info(
            f"[TimingSummary] avg_out_len_chars={_safe_mean(output_lens):.1f} n_outputs={len(output_lens)}"
        )

    logger.info(
        f"[RESULT] dataset={dataset_name} images={len(grouped)} prompts={len(instances)} "
        f"final_coco_annotations={len(final_results_list)}"
    )
    logger.info(f"[RESULT] saved to {results_file}")

    return results_file, final_results_list


def main():
    parser = argparse.ArgumentParser(description="vLLM-only Qwen eval using ShareGPT test prompts")

    parser.add_argument(
        "--eval_mode",
        type=str,
        required=True,
        choices=list(EVAL_MODE_TO_FILENAME.keys()),
    )
    parser.add_argument("--sharegpt_root", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--lora_model_path_root", type=str, default=None)
    parser.add_argument("--lora_checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.set_defaults(parallel=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument(
        "--score_mode",
        type=str,
        default="constant",
        choices=["constant", "bbox_coord_avg", "bbox_token"],
        help="Scoring mode for COCO score. constant keeps default 1.0 behavior.",
    )

    args = parser.parse_args()

    if args.parallel and not _is_worker_process():
        visible_gpu_ids = _visible_gpu_ids_from_env_or_torch()
        child_args = _args_without_auto_parallel(sys.argv[1:])
        _spawn_auto_parallel_workers(os.path.abspath(__file__), child_args, visible_gpu_ids)
        return

    if _is_worker_process() and args.vllm_tensor_parallel_size != 1:
        print("[WARN] parallel worker forces vllm_tensor_parallel_size=1")
        args.vllm_tensor_parallel_size = 1

    set_seed()

    model_name = (
        os.path.basename(os.path.normpath(args.lora_model_path_root))
        if args.lora_model_path_root
        else os.path.basename(os.path.normpath(args.base_model_path))
    )

    save_root = os.path.join(args.save_dir, model_name)
    os.makedirs(save_root, exist_ok=True)

    run_name = f"{model_name}_{args.eval_mode}_vllm_{args.score_mode}"
    if args.lora_checkpoint:
        run_name += f"_{args.lora_checkpoint}"
    if args.debug:
        run_name += "_debug"

    worker_rank, num_workers = _get_worker_rank_and_world()
    log_file_name = run_name if worker_rank < 0 else f"{run_name}.rank{worker_rank}"
    log_file = os.path.join(save_root, f"{log_file_name}.log")

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(threadName)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

    logger.info(f"Run name: {run_name}")
    logger.info(f"Eval mode: {args.eval_mode}")
    logger.info(f"ShareGPT root: {args.sharegpt_root}")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Score mode: {args.score_mode}")

    all_dataset_dirs = []
    for d in sorted(glob.glob(os.path.join(args.data_dir, "*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "test")):
            all_dataset_dirs.append(d)

    if worker_rank >= 0:
        my_shard, plan = split_datasets_by_cost(
            all_dataset_dirs,
            DATASET_COST,
            worker_rank,
            num_workers,
            default_cost=None,
            stable_tiebreak=True,
        )
        all_dataset_dirs = my_shard
        logger.info(f"[parallel] rank={worker_rank}/{num_workers} assigned {len(all_dataset_dirs)} datasets")
        logger.info(f"[parallel] shard plan: {plan}")

    eval_lora = args.lora_model_path_root is not None and os.path.isdir(args.lora_model_path_root)

    logger.info(f"Loading vLLM base model from {args.base_model_path} (enable_lora={eval_lora})")
    model, processor = load_qwen_vllm_model(
        args.base_model_path,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        enable_lora=eval_lora,
    )

    output_dir_root = os.path.join(save_root, "results", run_name)
    vis_dir_root = os.path.join(save_root, "visualizations", run_name) if args.vis else None
    os.makedirs(output_dir_root, exist_ok=True)
    if vis_dir_root:
        os.makedirs(vis_dir_root, exist_ok=True)

    for idx, dataset_dir in enumerate(tqdm(all_dataset_dirs, desc="Processing datasets", unit="dataset")):
        dataset_name = os.path.basename(dataset_dir)
        logger.info(f"Starting dataset: {dataset_name}")

        predictions_file = os.path.join(output_dir_root, f"predictions_{dataset_name}.json")
        if os.path.exists(predictions_file):
            logger.info(f"Skipping dataset: {dataset_name} (existing predictions file: {predictions_file})")
            continue

        lora_path = None
        lora_id = None
        if eval_lora:
            lora_subdir = EVAL_MODE_TO_LORA_SUBDIR[args.eval_mode]
            candidate_lora_path = os.path.join(args.lora_model_path_root, dataset_name, lora_subdir)
            if os.path.isdir(candidate_lora_path):
                lora_path = (
                    os.path.join(candidate_lora_path, args.lora_checkpoint.lstrip("/"))
                    if args.lora_checkpoint else candidate_lora_path
                )
                lora_id = idx + 1
                logger.info(f"Using LoRA adapter from {lora_path} with ID {lora_id}")
                if not os.path.isdir(lora_path):
                    logger.warning(f"LoRA path does not exist: {lora_path}; skipping dataset.")
                    continue
            else:
                logger.warning(f"No LoRA adapter found for {candidate_lora_path}; skipping dataset.")
                continue

        process_dataset_vllm(
            model=model,
            processor=processor,
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            eval_mode=args.eval_mode,
            sharegpt_root=args.sharegpt_root,
            results_dir=output_dir_root,
            vis_dir=vis_dir_root,
            model_name=args.base_model_path,
            batch_size=args.batch_size,
            debug=args.debug,
            score_mode=args.score_mode,
            lora_path=lora_path,
            lora_id=lora_id,
        )

        logger.info(f"Finished dataset: {dataset_name}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
