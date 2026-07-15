import argparse
import hashlib
import os
import re
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import json
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description="PaliGemma 2 Local Object Detection CLI")
    parser.add_argument("--video", type=str, required=True, help="Path to local video file")
    parser.add_argument("--classes", type=str, required=True, help="Path to json file defining classes")
    parser.add_argument("--output", type=str, required=True, help="Filename to save the bbox to")
    parser.add_argument("--model", type=str, default="google/paligemma2-3b-mix-448", help="Hugging Face model ID")
    return parser.parse_args()

# NB: -mix- models are fine-tuned for multiple tasks, -pt- models need fine-tuning before use

def get_class_color(label_name):
    """
    Generates a deterministic, high-visibility RGB color based on the label text string.
    Ensures the same object class always gets the same color.
    """
    # Create a stable hash of the label text
    hash_object = hashlib.md5(label_name.lower().strip().encode())
    hex_dig = hash_object.hexdigest()
    
    # Extract RGB values using chunks of the hash
    r = int(hex_dig[0:2], 16)
    g = int(hex_dig[2:4], 16)
    b = int(hex_dig[4:6], 16)
    
    # Maximize brightness/saturation for visual clarity against varied image backgrounds
    max_val = max(r, g, b, 1)
    r = int((r / max_val) * 200) + 55
    g = int((g / max_val) * 200) + 55
    b = int((b / max_val) * 200) + 55
    
    return (r, g, b)

def parse_paligemma_boxes(output_text, img_w, img_h):
    """
    Parses PaliGemma <locXXXX> tokens and scales them to original image pixels.
    Format is: <locYMIN><locXMIN><locYMAX><locXMAX> label
    """
    # Regex pattern to extract location tokens and the trailing label
    pattern = r"<loc(\d+)><loc(\d+)><loc(\d+)><loc(\d+)>\s*([^<]+)"
    matches = re.findall(pattern, output_text)
    
    results = []
    for match in matches:
        ymin, xmin, ymax, xmax = map(int, match[:4])
        label = match[4].strip().replace("\n", "").rstrip("; ")
        
        # Denormalize from 1024-grid to actual pixel spaces
        pixel_xmin = (xmin / 1024) * img_w
        pixel_ymin = (ymin / 1024) * img_h
        pixel_xmax = (xmax / 1024) * img_w
        pixel_ymax = (ymax / 1024) * img_h
        
        results.append({
            "label": label,
            "box": [pixel_xmin, pixel_ymin, pixel_xmax, pixel_ymax]
        })
    return results

def normalise_bbox(box, img_w, img_h):
    """Converts [xmin, ymin, xmax, ymax] into normalized [xmin, xmax, ymin, ymax]."""
    xmin, ymin, xmax, ymax = box
    return [ xmin / img_w, xmax / img_w, ymin / img_h, ymax / img_h ]


def read_label_classes(fname):
    """Reads label classes from json file"""
    try:
        with open(fname, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        print(f"Syntax Error (check for trailing commas!): {e}")
    except FileNotFoundError:
        print("The specified file was not found.")


def get_prompt(classes_list):
    """Calculates prompt from classes list"""
    prompt_list = sorted(classes_list, key=lambda x: x["prompt_order"])
    class_str = ""
    for item in prompt_list:
        if len(class_str) > 0:
            class_str += " ; "
        cprompt = item.get("prompt", "Unknown Label")
        class_str += cprompt
    prompt = "<image> detect " + class_str
    return prompt


def main():
    args = parse_args()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("Container Boot Failure: The 'HF_TOKEN' runtime variable is missing.")

    classes_list = read_label_classes(args.classes)

    # 1. Load the model directly into VRAM using bfloat16
    print(f"Loading {args.model} onto GPU...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda",
        token=hf_token
    )
    processor = AutoProcessor.from_pretrained(args.model, token=hf_token)
    
    # 2. Process video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Correct syntax format required by PaliGemma 2 for object detection
    prompt = get_prompt(classes_list)

    bboxes = []
    bboxid = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        fno = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        conv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(conv)
        img_w, img_h = image.size

        inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    
        # 3. Generate spatial bounding box tokens
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    
        # 4. Decode text and isolate model output from input prefix
        decoded = processor.decode(output[0], skip_special_tokens=False)
        # Extract only the generated suffix part
        input_len = inputs.input_ids.shape[1]
        generated_tokens = output[0][input_len:]
        clean_output = processor.decode(generated_tokens, skip_special_tokens=False)
    
        # 5. Parse and Print Detections
        detections = parse_paligemma_boxes(clean_output, img_w, img_h)

        print(f"{fno} {len(detections)}")

        for det in detections:
            box = det["box"] # [xmin, ymin, xmax, ymax]
            label = det["label"]
            
            match = None
            for item in classes_list:
                if item.get("prompt") == label:
                    match = item
                    break
            class_id = match["code"]
            bboxid += 1
            bbox = [bboxid, fno, box, class_id]
            bboxes.append(bbox)

    cap.release()

    with open(args.output, "w", encoding="utf-8") as txt_f:
        for bbox in bboxes:
            bboxid, fno, box, class_id = bbox
            nbox = normalise_bbox(box, img_w, img_h)
            txt_f.write(f"{bboxid} {fno} 0000000000.000000 " + " ".join(map(str, nbox)) + f" {class_id}" + "\n" )
    


if __name__ == "__main__":
    main()
