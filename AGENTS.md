# AGENTS

## Environment
- Set `HF_TOKEN` in the environment or `.env`. Without it, model download will fail.

## CLI Tools
- **Image detection** – `detect_cli.py`
  ```bash
  python detect_cli.py \ \
    --image path/to/img.jpg \ \
    --classes path/to/classes.json \ \
    [--output detected_output.jpg] \ \
    [--model google/paligemma2-3b-mix-448]
  ```
  The `--classes` file must be a JSON array where each entry contains at least:
  * `name`: human‑readable label
  * `prompt`: token used in the prompt
  * `code`: integer ID for YOLO export.

- **Video detection** – `apply_paligemma2.py`
  ```bash
  python apply_paligemma2.py \ \
    --video path/to/video.mp4 \ \
    --classes path/to/classes.json \ \
    --output output.txt \ \
    [--model google/paligemma2-3b-mix-448] \ \
    [--batch 16] \ \
    [--max_new_tokens 100]
  ```
  Results are written to the `--output` file as a `.txt` of bounding boxes.

## Notes
- GPU is required: the scripts set `device_map="cuda"`. If no CUDA device, script will crash.
- The first run downloads the model from Hugging Face; subsequent runs use cache.
- All outputs are deterministic when `HF_TOKEN` and the same classes file are used.

## Useful Resources
- **Model documentation**: https://ai.google.dev/gemma/docs/paligemma
- **Hugging Face caching**: https://huggingface.co/blog/paligemma2/
