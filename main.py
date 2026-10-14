"""
Colored line-art style: line art + flat fill, one color per body part sampled from the video.
Entry point. Run: python main.py -> pick an image/video -> the result is saved to OUTPUT_DIR.

pip install torch torchvision transformers opencv-python numpy timm
"""
import os

import torch

from config import CFG, STYLE_NAME, IMAGE_EXTS, VIDEO_EXTS
from utils import get_device
from segmenter import Parser
from pipeline import make_frame_fn
from colors import reset_color_memory
from media_io import process_image, process_video, pick_input_file, make_output_path


def main():
    in_path = pick_input_file()
    if not in_path:
        print("[INFO] No file selected -> exiting.")
        return
    ext = os.path.splitext(in_path)[1].lower()
    if ext in IMAGE_EXTS:
        kind = "image"
    elif ext in VIDEO_EXTS:
        kind = "video"
    else:
        raise ValueError(f"Unrecognized file type '{ext}'")
    out_path = make_output_path(in_path, kind)
    print(f"[INFO] {STYLE_NAME} | {in_path} ({kind}) -> {out_path}")

    device = get_device()
    dtype = torch.float16 if (CFG["USE_FP16"] and device.type == "cuda") else torch.float32
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[INFO] Device: {device} | dtype: {dtype}")

    parser = Parser(CFG["PARSE_MODEL_NAME"], device, dtype)
    frame_fn = make_frame_fn(parser, CFG)
    reset_color_memory()  # reset the color memory for a fresh run

    if kind == "image":
        process_image(frame_fn, in_path, out_path)
    else:
        process_video(frame_fn, in_path, out_path, CFG["MAX_FRAMES"])


if __name__ == "__main__":
    main()
