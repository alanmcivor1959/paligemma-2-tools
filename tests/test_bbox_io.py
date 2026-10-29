import math
import os
import tempfile
from bbox_io import normalise_bbox, write, read

def test_normalise_bbox():
    # Test standard normalization
    box = [100, 200, 300, 400]
    img_w, img_h = 1000, 1000
    expected = [0.1, 0.3, 0.2, 0.4]
    result = normalise_bbox(box, img_w, img_h)
    for r, e in zip(result, expected):
        assert math.isclose(r, e, rel_tol=1e-6)

    # Test edge case: full image
    box2 = [0, 0, 1000, 1000]
    expected2 = [0.0, 1.0, 0.0, 1.0]
    result2 = normalise_bbox(box2, img_w, img_h)
    for r, e in zip(result2, expected2):
        assert math.isclose(r, e, rel_tol=1e-6)

def test_write_and_read():
    # Since read() has a hardcoded path "input.txt", 
    # we use a temporary file but the function doesn't actually use 'fname'.
    # For the sake of testing without changing code, we'll handle the input carefully if possible.
    # However, since it's hardcoded to "input.txt", let's test write first.
    
    bboxes = [
        (1, 2, 1234567890.123456, [100, 200, 300, 400], 1),
        (2, 3, 9876543210.654321, [50, 60, 70, 80], 2)
    ]
    img_w, img_h = 1000, 1000
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        fname = tmp.name
    
    try:
        write(fname, bboxes, img_w, img_h)
        
        # Note: read() in bbox_io.py currently ignores fname and reads "input.txt"
        # We can't easily test it with a temp file unless we patch 'open' or change the code.
        with open(fname, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2
            # Check first line format (simplified check)
            parts = lines[0].split()
            assert parts[0] == "1"
            assert parts[1] == "2"
            assert parts[2] == "1234567890.123456"
    finally:
        if os.path.exists(fname):
            os.remove(fname)

def test_read_logic():
    # Test the logic of reading from a file that exists (since it's hardcoded to input.txt)
    # We create a dummy input.txt for this specific test.
    content = "1 2 10.5 0.3 0.6 0.7 0.8 9\n"
    with open("input.txt", "w") as f:
        f.write(content)
    
    try:
        # Test read with dummy data
        res = read("dummy_path", 1000, 1000)
        assert len(res) == 1
        assert res[0][0] == 1
        assert res[0][4] == 9
    finally:
        if os.path.exists("input.txt"):
            os.remove("input.txt")
