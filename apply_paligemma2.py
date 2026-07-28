import argparse
import hashlib
import os
import re
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import json
import cv2
import label_classes
import paligemma2_support
import bbox_io as bbio

def parse_args():
    parser = argparse.ArgumentParser(description="PaliGemma 2 Local Object Detection CLI")
    parser.add_argument("--video", type=str, required=True, help="Path to local video file")
    parser.add_argument("--classes", type=str, required=True, help="Path to json file defining classes")
    parser.add_argument("--output", type=str, required=True, help="Filename to save the bbox to")
    parser.add_argument("--model", type=str, default="google/paligemma2-3b-mix-448", help="Hugging Face model ID")
    return parser.parse_args()

# NB: -mix- models are fine-tuned for multiple tasks, -pt- models need fine-tuning before use

def main():
    args = parse_args()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("Container Boot Failure: The 'HF_TOKEN' runtime variable is missing.")

    classes_list = label_classes.read_label_classes(args.classes)

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
    prompt = label_classes.get_prompt(classes_list)

    bboxes = []
    bboxid = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        fno = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ts = 0.0

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
        detections = paligemma2_support.parse_boxes(clean_output, img_w, img_h)

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
            bbox = [bboxid, fno, ts, box, class_id]
            bboxes.append(bbox)

    cap.release()

    bbio.write(args.output, bboxes, img_w, img_h)

if __name__ == "__main__":
    main()
