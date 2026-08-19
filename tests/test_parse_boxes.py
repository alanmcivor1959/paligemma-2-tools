import math

from paligemma2_support import parse_boxes

# Helper for rounding expected values

def calc_box(img_w, img_h, numbers):
    xmin = (numbers[1] / 1024) * img_w
    ymin = (numbers[0] / 1024) * img_h
    xmax = (numbers[3] / 1024) * img_w
    ymax = (numbers[2] / 1024) * img_h
    return [xmin, ymin, xmax, ymax]


def test_single_detection():
    input_str = "<loc0120><loc0345><loc0678><loc0987> cat"
    boxes = parse_boxes(input_str, 200, 100)
    assert len(boxes) == 1
    expected_box = calc_box(200, 100, [120, 345, 678, 987])
    # Allow small floating point error
    for val, exp in zip(boxes[0]["box"], expected_box):
        assert math.isclose(val, exp, rel_tol=1e-6)
    assert boxes[0]["label"] == "cat"


def test_multiple_detections():
    input_str = (
        "<loc0000><loc0000><loc1024><loc1024> dog "
        + "<loc0102><loc0203><loc0304><loc0405> mouse"
    )
    boxes = parse_boxes(input_str, 512, 256)
    assert len(boxes) == 2
    # first detection should span full image
    expected_box0 = calc_box(512, 256, [0, 0, 1024, 1024])
    for val, exp in zip(boxes[0]["box"], expected_box0):
        assert math.isclose(val, exp, rel_tol=1e-6)
    assert boxes[0]["label"] == "dog"

    # second detection: coordinates 102/1024 *512 and etc.
    expected_box1 = calc_box(512, 256, [102, 203, 304, 405])
    for val, exp in zip(boxes[1]["box"], expected_box1):
        assert math.isclose(val, exp, rel_tol=1e-6)
    assert boxes[1]["label"] == "mouse"


def test_no_detections():
    input_str = "no detections here"
    boxes = parse_boxes(input_str, 800, 600)
    assert boxes == []


def test_invalid_format_ignored():
    # Missing last <loc> - should ignore and return no result
    input_str = "<loc0123><loc0456><loc0789> missing"
    boxes = parse_boxes(input_str, 1000, 1000)
    assert boxes == []
