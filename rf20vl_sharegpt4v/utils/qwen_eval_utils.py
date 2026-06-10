import os
import json
import base64
import numpy as np
import cv2
import time
import random
import re
import supervision as sv
from openai import OpenAI
import pickle
from utils.shared_eval_utils import *
import math
from typing import List, Dict, Optional, Tuple, Any

MODEL_ID = "qwen2.5-vl-72b-instruct"
MIN_PIXELS = 4*28*28
MAX_PIXELS = 12800*28*28
SYSTEM_PROMPT = "You are a helpful assistant capable of object detection."
MAX_WORKERS = 4
REQUEST_LIMIT = 100
MAX_RETRIES = 2
RETRY_DELAY_BASE = 8
RETRY_DELAY_MAX = 300

def load_processing_status(status_file, logger):
    """Load the processing status using pickle"""
    if os.path.exists(status_file) and os.path.getsize(status_file) > 0:
        try:
            with open(status_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading status: {e}")
            return {}
    return {}

def load_existing_results(results_file, logger):
    """Load existing results from a file if it exists"""
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse existing results file {results_file}. Creating backup.")
            backup_file = f"{results_file}.bak.{int(time.time())}"
            os.rename(results_file, backup_file)
            return []
    return []

def save_processing_status(status_map, status_file, logger):
    """Save the processing status using pickle"""
    temp_file = status_file + ".tmp"
    try:
        os.makedirs(os.path.dirname(status_file), exist_ok=True)
        with open(temp_file, 'wb') as f:
            pickle.dump(status_map, f)
        os.replace(temp_file, status_file)
        return True
    except Exception as e:
        logger.error(f"Error saving status: {e}")
        print(f"Error saving status: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return False
    
def is_valid_json(json_text):
    """Check if a string is valid JSON"""
    try:
        if not json_text or not json_text.strip().startswith(('[', '{')):
            return False
        json.loads(json_text)
        return True
    except json.JSONDecodeError:
        return False

def encode_image(image_path):
    """Encode image to base64 format"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def is_timeout_error(error_msg):
    """Check if the error is a timeout error"""
    return "RequestTimeOut" in str(error_msg) or "timed out" in str(error_msg).lower()

def parse_qwen_response(response, logger):
    """Parse the JSON response from Qwen model, handling markdown and potential issues"""
    if not response:
        return []
    
    

    json_text = ""
    try:
        print("************************ ",response)
        response = re.sub(r"^.*?</think>\s*", "", response, flags=re.DOTALL) #retains stuff only after </think> token if it exists 
        
        if "```json" in response:
            json_text = response.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in response:
            match = re.search(r"```(?:.*\n)?([\s\S]+?)```", response)
            if match:
                json_text = match.group(1).strip()
            else:
                json_text = response.strip()
        else:
            json_text = response.strip()

        json_text = re.sub(r",\s*(\]|\})", r"\1", json_text)

        if not is_valid_json(json_text):
            logger.warning(f"Invalid JSON structure detected in response: {json_text}...")
            match = re.search(r'(\[.*\]|\{.*\})', json_text, re.DOTALL)
            if match:
                json_text = match.group(1)
                if not is_valid_json(json_text):
                    # logger.warning(f"Could not extract valid JSON even after searching: {json_text}...")
                    # return []
                    object_matches = re.findall(r'\{[^{}]*\}', json_text, re.DOTALL)
                    valid_objects = []
                    for obj in object_matches:
                        parsed = json.loads(obj)
                        valid_objects.append(parsed)

                    json_text = json.dumps(valid_objects)
                    # print("**********",json_text)
                    if(valid_objects == []):
                        logger.warning(f"Could not extract valid JSON even after searching: {json_text}...")
                        return []
            else:
                 logger.warning(f"No JSON list/object found in the response: {json_text}...")
                 return []

        parsed_data = json.loads(json_text)

        if isinstance(parsed_data, list):
            boxes = parsed_data
        elif isinstance(parsed_data, dict):
             potential_keys = ['boxes', 'detections', 'objects', 'annotations']
             found = False
             for key in potential_keys:
                 if key in parsed_data and isinstance(parsed_data[key], list):
                     boxes = parsed_data[key]
                     found = True
                     break
             if not found:
                 logger.warning(f"Parsed JSON is a dict, but no expected list key found: {list(parsed_data.keys())}")
                 return []
        else:
            logger.warning(f"Parsed JSON is not a list or expected dict format: {type(parsed_data)}")
            return []

        return boxes

    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response: {e}")
        logger.debug(f"Problematic JSON text: {json_text}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error parsing response: {e}")
        logger.debug(f"Original Response: {response}")
        return []


def visualize_predictions(image_path, boxes, input_width, input_height, 
                          original_width, original_height, categories_dict, save_path, logger):
    """Visualize prediction boxes on the image using supervision library"""
    frame = cv2.imread(image_path)
    
    xyxy = []
    class_ids = []
    confidences = []
    labels = []
    
    for box in boxes:
        if not isinstance(box, dict):
            continue
            
        bbox_2d = box.get("bbox_2d", None)
        if not bbox_2d or len(bbox_2d) != 4:
            continue
            
        label_text = box.get("label", "unknown")
        
        category_id = None
        for cat_id, cat_name in categories_dict.items():
            if cat_name.lower() in label_text.lower():
                category_id = cat_id
                break
        
        if category_id is None:
            category_id = -1
        
        x1, y1, x2, y2 = bbox_2d
        x1 = int(x1 / input_width * original_width)
        y1 = int(y1 / input_height * original_height)
        x2 = int(x2 / input_width * original_width)
        y2 = int(y2 / input_height * original_height)
        
        xyxy.append([x1, y1, x2, y2])
        class_ids.append(category_id)
        # If upstream attached confidence/score, use it; otherwise default to 1.0.
        conf = box.get("confidence", box.get("score", 1.0))
        try:
            conf = float(conf)
        except Exception:
            conf = 1.0
        confidences.append(conf)
        labels.append(label_text)
    
    if not xyxy:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, frame)
        return
    
    xyxy = np.array(xyxy, dtype=np.float32)
    class_ids = np.array(class_ids, dtype=np.int32)
    confidences = np.array(confidences, dtype=np.float32)
    
    detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidences,
        class_id=class_ids
    )
    
    box_annotator = sv.BoxAnnotator(
        color=sv.Color.ROBOFLOW, 
        thickness=2
    )
    
    label_annotator = sv.LabelAnnotator(
        color=sv.Color.ROBOFLOW,
        text_color=sv.Color.WHITE,
        text_scale=0.5,
        text_thickness=1
    )
    
    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )
    
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, annotated_frame)