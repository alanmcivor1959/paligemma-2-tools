import argparse
import collections
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

def convert_class_code_to_index(classes,class_id):
    for i in range(len(classes)):
        if class_id == classes.iloc[i,0]:
            return i
    sys.exit("Invalid class code") 

def load_bounding_boxes(file_path, img_w, img_h):
    frames_dict = collections.defaultdict(list)

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            fields = line.split()
            frame_num = int(fields[1])

            # Layout: [x_min, y_min, x_max, y_max]
            box = [
                float(fields[3])*img_w,  # x_min
                float(fields[5])*img_h,  # y_min
                float(fields[4])*img_w,  # x_max
                float(fields[6])*img_h,  # y_max
            ]
            label = int(fields[7])  # class_id

            frames_dict[frame_num].append({"box": box, "label": label})

    if not frames_dict:
        return []
    max_frame = max(frames_dict.keys())
    return [frames_dict[i] for i in range(max_frame + 1)]


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def compare_boxes_soft_matching(tool_a_frames, tool_b_frames, classes, lam, iou_threshold, symmetric):
    """
    Compares boxes globally using standard optimization matrix: -(iou + lam * X).
    Collects exact classification counts to map out a complete Confusion Matrix.
    """
    total_frames = max(len(tool_a_frames), len(tool_b_frames))
    
    bg_idx = len(classes)
    
    confusion_matrix = np.zeros((bg_idx + 1, bg_idx + 1), dtype=int)

    for frame_idx in range(total_frames):
        boxes_a = tool_a_frames[frame_idx] if frame_idx < len(tool_a_frames) else []
        boxes_b = tool_b_frames[frame_idx] if frame_idx < len(tool_b_frames) else []

        num_a = len(boxes_a)
        num_b = len(boxes_b)

        if not symmetric and num_a == 0:
            continue
        
        if num_a == 0:
            for item_b in boxes_b:
                b_idx = convert_class_code_to_index(classes,item_b["label"])
                confusion_matrix[bg_idx, b_idx] += 1
            continue
            
        if num_b == 0:
            for item_a in boxes_a:
                a_idx = convert_class_code_to_index(classes,item_a["label"])
                confusion_matrix[a_idx, bg_idx] += 1
            continue

        cost_matrix = np.zeros((num_a, num_b))
        iou_matrix = np.zeros((num_a, num_b))
        
        for i, item_a in enumerate(boxes_a):
            for j, item_b in enumerate(boxes_b):
                iou = calculate_iou(item_a["box"], item_b["box"])
                iou_matrix[i, j] = iou
                x = 1.0 if item_a["label"] == item_b["label"] else 0.0
                cost_matrix[i, j] = -(iou + lam * x)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_a = set()
        matched_b = set()

        for r, c in zip(row_ind, col_ind):
            iou = iou_matrix[r, c]
            if iou >= iou_threshold:
                matched_a.add(r)
                matched_b.add(c)
                
                a_idx = convert_class_code_to_index(classes,boxes_a[r]["label"])
                b_idx = convert_class_code_to_index(classes,boxes_b[c]["label"])
                confusion_matrix[a_idx, b_idx] += 1

        # Handle leftover unmatched boxes from Tool A (Missed / False Negatives)
        for i in range(num_a):
            if i not in matched_a:
                a_idx = convert_class_code_to_index(classes,boxes_a[i]["label"])
                confusion_matrix[a_idx, bg_idx] += 1

        # Handle leftover unmatched boxes from Tool B (Phantom detections / False Positives)
        for j in range(num_b):
            if j not in matched_b:
                b_idx = convert_class_code_to_index(classes,boxes_b[j]["label"])
                confusion_matrix[bg_idx, b_idx] += 1

    return confusion_matrix


def calculate_per_class_metrics(matrix, classes):
    """
    Computes Precision, Recall, and Accuracy per class from the confusion matrix.
    Total population (N) for class metrics includes all localized bounding boxes.
    """
    metrics = {}
    total_elements = np.sum(matrix)
    bg_idx = len(classes)

    for idx in range(len(classes)):
        # True Positive: Class correctly predicted as Class
        tp = matrix[idx, idx]
        
        # False Positive: Something else predicted as this class (Column sum minus TP)
        fp = np.sum(matrix[:, idx]) - tp
        
        # False Negative: This class predicted as something else or missed (Row sum minus TP)
        fn = np.sum(matrix[idx, :]) - tp
        
        # True Negative: All other assignments in the entire video evaluation matrix
        tn = total_elements - (tp + fp + fn)

        # Performance Ratios
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / total_elements if total_elements > 0 else 0.0

        metrics[idx] = {
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
        }
        
    return metrics


def print_performance_report(metrics, classes):
    """Outputs a structured summary table with class breakdowns and all three aggregation types."""
    print("=" * 78)
    print("                 PER-CLASS EVALUATION METRICS                    ")
    print("=" * 78)
    print(f"{'Class':<15}{'Precision':>12}{'Recall':>12}{'F1-Score':>12}{'Accuracy':>12}{'Total TP':>12}")
    print("-" * 78)
    
    # Track global sums for Micro-Averaging
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Track metrics for Macro and Weighted Macro
    class_precisions = []
    class_recalls = []
    class_f1s = []
    class_weights = []  # Ground truth instances per class (TP + FN)
    
    labels = []
    for i in range(len(classes)):
        labels.append(classes.iloc[i,1])

    for cls, scores in metrics.items():
        tp = scores['counts']['tp']
        fp = scores['counts']['fp']
        fn = scores['counts']['fn']
        
        p = scores['precision']
        r = scores['recall']
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        
        # Ground Truth Count for this class acts as its weight
        gt_count = tp + fn

        print(f"{labels[cls]:<15}"
              f"{p:>12.4f}"
              f"{r:>12.4f}"
              f"{f1:>12.4f}"
              f"{scores['accuracy']:>12.4f}"
              f"{tp:>12}")
        
        # Accumulate for Micro
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Accumulate for Macro / Weighted Macro
        class_precisions.append(p)
        class_recalls.append(r)
        class_f1s.append(f1)
        class_weights.append(gt_count)
        
    print("-" * 78)
    
    # 1. Micro-Averaged Performance Metrics
    #    Average over all detections independently
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    print(f"{'MICRO AGG':<15}"
          f"{micro_precision:>12.4f}"
          f"{micro_recall:>12.4f}"
          f"{micro_f1:>12.4f}"
          f"{'':>12}"  
          f"{total_tp:>12}")
          
    # 2. Macro-Averaged Performance Metrics
    #    Average of all class statistics
    num_classes = len(metrics) if metrics else 1
    macro_precision = sum(class_precisions) / num_classes
    macro_recall = sum(class_recalls) / num_classes
    macro_f1 = sum(class_f1s) / num_classes
    
    print(f"{'MACRO AGG':<15}"
          f"{macro_precision:>12.4f}"
          f"{macro_recall:>12.4f}"
          f"{macro_f1:>12.4f}"
          f"{'':>12}"  
          f"{'-':>12}")
          
    # 3. Weighted Macro-Averaged Performance Metrics
    #    Average of all class statistics weighted by numbers in baseline set
    total_weight = sum(class_weights)
    if total_weight > 0:
        weighted_precision = sum(p * w for p, w in zip(class_precisions, class_weights)) / total_weight
        weighted_recall = sum(r * w for r, w in zip(class_recalls, class_weights)) / total_weight
        weighted_f1 = sum(f1 * w for f1, w in zip(class_f1s, class_weights)) / total_weight
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    print(f"{'WEIGHTED MACRO':<15}"
          f"{weighted_precision:>12.4f}"
          f"{weighted_recall:>12.4f}"
          f"{weighted_f1:>12.4f}"
          f"{'':>12}"  
          f"{'-':>12}")
          
    print("=" * 78)

def print_confusion_matrix_report(matrix, classes):
    """Prints a structured, readable confusion matrix grid."""
    labels = []
    for i in range(len(classes)):
        labels.append(classes.iloc[i,1])
    # use the label background instead of missing/extra, as assuming all objects
    # of interest in a frame are labelled, a missing detection is an object identified
    # as background and an extra detection is part of the background identified as
    # an object (hallucination)
    labels.append("background")
    print("=" * 65)
    print("                 CONFUSION MATRIX SUMMARY                        ")
    print("=" * 65)
    print("Rows: Baseline  | Columns: Candidate  \n")
    
    # Header format string
    header_str = f"{'':<15}" + "".join([f"{lbl:>12}" for lbl in labels])
    print(header_str)
    print("-" * len(header_str))
    
    for i, row_label in enumerate(labels):
        row_str = f"{row_label:<15}" + "".join([f"{matrix[i, j]:>12}" for j in range(len(labels))])
        print(row_str)
    print("=" * 65)

# this can't yet handle a custom dataset that uses standard object classes
def load_classes(fname):
    df = pd.DataFrame()
    if fname is None:
        data = { 'category' : [1, 2, 3, 4, 5 ], 'name' : ["person", "trolley", "group", "bicycle", "scooter"] }
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(fname, sep="\\s+", header=None, names=['category', 'name'])
    return df


def image_size(value):
    """Validates and parses a positive integer pair in WxH format."""
    try:
        width, height = map(int, value.lower().split('x'))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid format: '{value}'. Must be WidthxHeight using integers (e.g., 800x600)."
        )
    
    # Enforce strictly positive values
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(
            f"Invalid dimensions: {width}x{height}. Both width and height must be greater than 0."
        )
        
    return width, height


def parse_args():
    parser = argparse.ArgumentParser(description="Compare candidate bounding boxes to baseline")
    parser.add_argument("-t", "--baseline", help="Baseline bounding boxes", required=True)
    parser.add_argument("-o", "--candidate", help="Candidate bounding boxes", required=True)
    parser.add_argument("-u", "--classes", type=str, help="use these object classes (standard set: person/bicycle/scooter)")
    parser.add_argument("--size", type=image_size, default=(1024,576), help="Frame dimensions formatted as WidthxHeight (1024x576)")
    parser.add_argument("--iou", type=float, default=0.5, help="IOU threshold (0.5)")
    parser.add_argument("--lam", type=float, default=0.2, help="Weight to apply to class difference (0.2)")
    parser.add_argument("--symmetric", action='store_true', help="Treat sets symmetrically instead of a baseline/candidate pair")
    return parser.parse_args()


def main():
    args = parse_args()

    class_df = load_classes(args.classes)
    width, height = args.size

    baseline_b = load_bounding_boxes(args.baseline, width, height)
    candidate_b = load_bounding_boxes(args.candidate, width, height)

    lam = args.lam
    iou_threshold = args.iou
    symmetric = args.symmetric
    matrix = compare_boxes_soft_matching(baseline_b, candidate_b, class_df, lam, iou_threshold, symmetric)

    metrics_summary = calculate_per_class_metrics(matrix, class_df)

    print("");
    if not symmetric:
        print_performance_report(metrics_summary, class_df)
        print("");
    print_confusion_matrix_report(matrix, class_df)
    print("");


if __name__ == "__main__":
    main()

