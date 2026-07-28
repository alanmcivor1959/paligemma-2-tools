def normalise_bbox(box, img_w, img_h):
    """Converts [xmin, ymin, xmax, ymax] into normalized [xmin, xmax, ymin, ymax]."""
    xmin, ymin, xmax, ymax = box
    return [ xmin / img_w, xmax / img_w, ymin / img_h, ymax / img_h ]


def write(fname, bboxes, img_w, img_h):
    """Write bboxes in std format to fname"""
    with open(fname, "w", encoding="utf-8") as txt_f:
        for bbox in bboxes:
            bboxid, fno, ts, box, class_id = bbox
            nbox = normalise_bbox(box, img_w, img_h)
            tv_sec = int(ts);
            tv_usec = round((ts - tv_sec) * 1_000_000)
            ts_str = f"{tv_sec:10d}.{tv_usec:06d}"
            txt_f.write(f"{bboxid} {fno} {ts_str} " + " ".join(map(str, nbox)) + f" {class_id}" + "\n" )

def read(fname, img_w, img_h):
    """Read bboxes from fname and return as array"""
    bboxes = []
    with open("input.txt", "r") as infile:
        for line in infile:
            fields = line.split()
            if not fields:
                continue  # Skip empty lines
            bboxid = int(fields[0])
            fno = int(fields[1])
            ts = float(fields[2])
            x1 = float(fields[3]) * img_w
            x2 = float(fields[4]) * img_w
            y1 = float(fields[5]) * img_h
            y2 = float(fields[6]) * img_h
            box = [x1, y1, x2, y2]
            class_id = int(fields[7])
            bbox = [bboxid, fno, ts, box, class_id]
            bboxes.append(bbox)
    return bboxes

