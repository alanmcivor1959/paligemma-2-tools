import argparse
import os
import re
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import label_classes
import paligemma2_support


def parse_args():
    parser = argparse.ArgumentParser(description="PaliGemma 2 Local Object Detection CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to local image file")
    parser.add_argument("--classes", type=str, required=True, help="Path to json file defining classes")
    parser.add_argument(
        "--output", type=str, default="detected_output.jpg", help="Filename to save the visualized image"
    )
    parser.add_argument("--model", type=str, default="google/paligemma2-3b-mix-448", help="Hugging Face model ID")
    return parser.parse_args()


# NB: -mix- models are fine-tuned for multiple tasks, -pt- models need fine-tuning before use


def convert_to_yolo(box, img_w, img_h):
    """Converts [xmin, ymin, xmax, ymax] into YOLO normalized [x_center, y_center, width, height]."""
    xmin, ymin, xmax, ymax = box

    # Calculate box width and height
    box_w = xmax - xmin
    box_h = ymax - ymin

    # Calculate center point coordinates
    x_center = xmin + (box_w / 2)
    y_center = ymin + (box_h / 2)

    # Normalise against total image scale dimensions
    return [round(x_center / img_w, 6), round(y_center / img_h, 6), round(box_w / img_w, 6), round(box_h / img_h, 6)]


def main():
    args = parse_args()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("Container Boot Failure: The 'HF_TOKEN' runtime variable is missing.")

    # Formulate precise dynamic class indexes from user array entries
    classes_list = label_classes.read_label_classes(args.classes)

    # 1. Load the model directly into VRAM using bfloat16
    print(f"Loading {args.model} onto GPU...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", token=hf_token
    )
    processor = AutoProcessor.from_pretrained(args.model, token=hf_token)

    # 2. Process image and format target prefix
    image = Image.open(args.image).convert("RGB")
    img_w, img_h = image.size

    # Correct syntax format required by PaliGemma 2 for object detection
    prompt = label_classes.get_prompt(classes_list)
    print(f"Processing prompt: '{prompt.strip()}'")

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

    #    print(f"\nRaw model response: {clean_output}")

    # 6. Print detections
    print("\n--- Detection Results ---")
    if not detections:
        print("No targets detected.")
        print(f"Raw model response: {clean_output}")
    for det in detections:
        print(f"Label: {det['label']} | Bounding Box [xmin, ymin, xmax, ymax]: {det['box']}")

    # 7. Visualise and Draw Multi-Coloured Boxes
    if detections:
        print(f"\nFound {len(detections)} object(s). Drawing distinct class boundaries...")
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.load_default(size=16)
        except TypeError:
            font = ImageFont.load_default()

        for det in detections:
            box = det["box"]  # [xmin, ymin, xmax, ymax]
            label = det["label"]

            match = None
            for item in classes_list:
                if item.get("prompt") == label:
                    match = item
                    break
            name = match["name"]

            # Fetch the dynamic color for this specific label category
            class_color = label_classes.get_class_colour(name)

            print(f" -> Label: {name} | Box: {box} | RGB: {class_color}")

            # Draw bounding box (3 pixels thick)
            draw.rectangle(box, outline=class_color, width=3)

            # Draw text background banner using the same class color
            text_size = draw.textbbox((box[0], box[1]), name, font=font)
            text_w = text_size[2] - text_size[0]
            text_h = text_size[3] - text_size[1]

            # Keep label boundary inside image dimensions
            text_background = [box[0], max(0, box[1] - text_h - 4), box[0] + text_w + 6, box[1]]
            draw.rectangle(text_background, fill=class_color)

            # Render text in black or white depending on background choice (default black here)
            draw.text((box[0] + 3, max(0, box[1] - text_h - 2)), name, fill=(0, 0, 0), font=font)

        # Save output image
        image.save(args.output)
        print(f"Success! Visualised output saved to: {args.output}")

    # 8. Save data in yolo format
    yolo_lines = []
    if detections:
        for det in detections:
            box = det["box"]  # [xmin, ymin, xmax, ymax]
            label = det["label"]

            match = None
            for item in classes_list:
                if item.get("prompt") == label:
                    match = item
                    break
            class_id = match["code"]
            yolo_box = convert_to_yolo(box, img_w, img_h)

            yolo_lines.append(f"{class_id} " + " ".join(map(str, yolo_box)))

        filename = os.path.basename(args.image)
        base_name, _ = os.path.splitext(filename)

        print(f"{base_name}\n")

        out_txt_path = f"{base_name}.yolo"
        with open(out_txt_path, "w", encoding="utf-8") as txt_f:
            txt_f.write("\n".join(yolo_lines) + "\n")
        print(f"Success! YOLO output saved to: {out_txt_path}")


if __name__ == "__main__":
    main()
