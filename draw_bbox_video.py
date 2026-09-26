import argparse
import collections
import cv2
import time
import bbox_io as bbio
import classes_io as cio
import label_classes

def parse_args():
    parser = argparse.ArgumentParser(description="BBOX overlay drawer")
    parser.add_argument("--video", type=str, required=True, help="Path to local video file")
    parser.add_argument("--bbox", help="Bounding boxes", required=True)
    parser.add_argument("--classes", type=str, required=True, help="Path to json file defining classes")
    parser.add_argument("--output", type=str, required=True, help="Filename to save the video to")
    return parser.parse_args()

def load_bounding_boxes(file_path, img_w, img_h):
    frames_dict = collections.defaultdict(list)

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            fields = line.split()
            try:
                frame_num = int(fields[1])
                # Layout: [x_min, y_min, x_max, y_max]
                box = [
                    float(fields[3]) * img_w,  # x_min
                    float(fields[5]) * img_h,  # y_min
                    float(fields[4]) * img_w,  # x_max
                    float(fields[6]) * img_h,  # y_max
                ]
                label = int(fields[7])  # class_id
            except (IndexError, ValueError):
                continue
            
            frames_dict[frame_num].append({"box": box, "label": label})

    if not frames_dict:
        return []
    max_frame = max(frames_dict.keys())
    return [frames_dict[i] for i in range(max_frame + 1)] # frames_dict[0] is always empty


args = parse_args()

# 1. Open the existing source video
cap = cv2.VideoCapture(args.video)

if not cap.isOpened():
    print(f"Error: Could not open input video {args.video}")
    exit()

# 2. Extract properties from the original video
# We use int() because VideoWriter demands integer dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

bboxes = load_bounding_boxes(args.bbox, frame_width, frame_height)

classes = cio.read_classes(args.classes)

# 3. Initialize the VideoWriter for Linux
# 'mp4v' inside an .mp4 container works out-of-the-box on most Linux distros
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))



# Check if writer opened successfully
if not out.isOpened():
    print("Error: Could not open VideoWriter. Try changing codec to *'XVID' with .avi extension.")
    cap.release()
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video file
        
    frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # -------------------------------------------------------------------------
    # 4. PERFORM ANNOTATIONS
    # Note: On Linux, ensure the colorspace is BGR (OpenCV's default)
    # -------------------------------------------------------------------------
    
    boxes = bboxes[frame_no] if frame_no < len(bboxes) else []

    for bb in boxes:
        box = bb["box"]
        label = bb["label"]
        match = next((item for item in classes if item.get("code") == label), None)
        class_name = match.get("name")
        ccolour = label_classes.get_class_colour(class_name)
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))
        cv2.rectangle(frame, start_point, end_point, ccolour, 1)
        x1,y1 = start_point
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        # 3. Calculate text size to match the background rectangle perfectly
        (text_width, text_height), baseline = cv2.getTextSize(class_name, font, font_scale, thickness)
        # 4. Define background coordinates (positioned right above y1)
        bg_start = (x1, y1 - text_height - baseline - 2)
        bg_end = (x1 + text_width, y1)
        # 5. Draw the background rectangle (filled with ccolour)
        cv2.rectangle(frame, bg_start, bg_end, ccolour, cv2.FILLED)
        # 6. Draw the white text on top of the background
        # (0, 0, 0) is black, (255, 255, 255) is white in BGR format
        text_origin = (x1, y1 - baseline - 1)
        cv2.putText(frame, class_name, text_origin, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    out.write(frame)

# 6. Clean up resources
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Successfully saved annotated video to: {args.output}")
