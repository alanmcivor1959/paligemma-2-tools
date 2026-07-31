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
    parser.add_argument("--batch", type=int, default=16, help="Number of frames per batch")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens in response")
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
    
    frame_buffer = []
    frame_no = []
    frame_ts = []

    # 2. Process video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Correct syntax format required by PaliGemma 2 for object detection
    prompt = label_classes.get_prompt(classes_list)

    bboxes = []
    bboxid = 0
    
    batch_size = args.batch

    max_new_tokens = args.max_new_tokens

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            frame_buffer.append(frame)
            frame_no.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            frame_ts.append(0.0)

        # Process when the buffer is full OR when we hit the end of the video
        if len(frame_buffer) == batch_size or (not ret and len(frame_buffer) > 0):

            # Convert the batch of OpenCV frames to PIL RGB format
            images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frame_buffer]
            prompts = [prompt] * len(frame_buffer)

            img_w, img_h = images[0].size

            # Preprocess the batch (Padding handles matching structural shapes)
            inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to("cuda")
            prompt_length = inputs["input_ids"].shape[-1]

            # Run parallel batch inference
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                generated_ids = generated_ids[:, prompt_length:]

            outputs = processor.batch_decode(generated_ids, skip_special_tokens=False)
    
            # Overlay detections and write out each frame in the batch
            for idx, output_text in enumerate(outputs):
                current_frame = frame_buffer[idx]
                fno = frame_no[idx]
                ts = frame_ts[idx]
                detections = paligemma2_support.parse_boxes(output_text, img_w, img_h)

                print(f"{fno} {len(detections)}")

                for det in detections:
                    box = det["box"] # [xmin, ymin, xmax, ymax]
                    label = det["label"]
            
                    match = label_classes.get_entry_from_prompt(classes_list, label)
                    if match is None:
                        print(f"{fno}: Invalid label '{label}'")
                    if match is not None:
                        class_id = match["code"]
                        bboxid += 1
                        bbox = [bboxid, fno, ts, box, class_id]
                        bboxes.append(bbox)

            # Clear the buffer for the next batch
            frame_buffer.clear()
            frame_no.clear()
            frame_ts.clear()

        if not ret:
            break

    cap.release()

    bbio.write(args.output, bboxes, img_w, img_h)

if __name__ == "__main__":
    main()
