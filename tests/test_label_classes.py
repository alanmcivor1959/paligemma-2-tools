import os
import json
import tempfile
from label_classes import (
    read_label_classes,
    get_prompt,
    get_class_colour,
    get_entry_from_prompt
)

def test_read_label_classes():
    # Create a temporary JSON file for testing
    content = [{"name": "object1", "prompt": "a_thing"}]
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as tmp:
        tmp.write(json.dumps(content))
        temp_path = tmp.name

    try:
        data = read_label_classes(temp_path)
        assert len(data) == 1
        assert data[0]["name"] == "object1"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_get_prompt():
    classes = [
        {"name": "door", "prompt": "a door", "prompt_order": 1},
        {"name": "window", "prompt": "a window", "prompt_order": 2}
    ]
    expected = "<image> detect a door ; a window\n"
    assert get_prompt(classes) == expected

def test_get_class_colour():
    # Test consistency: same label should yield same color
    color1 = get_class_colour("door")
    color2 = get_class_colour("door")
    assert color1 == color2
    assert len(color1) == 3
    assert all(0 <= c <= 255 for c in color1)

    # Test different labels produce different colors (likely)
    color3 = get_class_colour("window")
    assert color1 != color3

def test_get_entry_from_prompt():
    classes = [
        {"name": "door", "prompt": "a door"},
        {"name": "window", "prompt": "a window"}
    ]
    # Valid prompt
    res1 = get_entry_from_prompt(classes, "a door")
    assert res1 is not None
    assert res1["name"] == "door"

    # Invalid prompt
    res2 = get_entry_from_prompt(classes, "a car")
    assert res2 is None
