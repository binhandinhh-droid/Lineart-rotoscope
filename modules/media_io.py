"""Image / video input and output, file picker and output path helpers."""
import os
import re

import numpy as np
import cv2

from config import OUTPUT_DIR, DIALOG_START_DIR, IMAGE_EXTS, VIDEO_EXTS, STYLE_NAME


def process_image(frame_fn, in_path, out_path):
    """Run the frame function on a single image and save the result."""
    bgr = cv2.imdecode(np.fromfile(in_path, np.uint8), cv2.IMREAD_COLOR)  # safe with Unicode paths
    if bgr is None:
        raise RuntimeError(f"Could not read image: {in_path}")
    result = frame_fn(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    ok, buf = cv2.imencode(os.path.splitext(out_path)[1], cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Could not encode the result image")
    buf.tofile(out_path)
    print(f"[OK] Image saved: {out_path}")


def process_video(frame_fn, in_path, out_path, max_frames=None):
    """Run the frame function on every frame of a video and write an mp4."""
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {in_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    limit = max_frames or float("inf")
    n = 0
    try:
        while n < limit:
            ok, bgr = cap.read()
            if not ok:
                break
            out = frame_fn(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            writer.write(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            n += 1
            if n % 20 == 0:
                print(f"  ... {n}/{total} frames")
    finally:
        cap.release()
        writer.release()
    print(f"[OK] Video saved: {out_path} ({n} frames)")


def pick_input_file():
    """Open a file dialog and return the chosen image/video path (or None if cancelled)."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    exts = " ".join("*" + e for e in IMAGE_EXTS + VIDEO_EXTS)
    path = filedialog.askopenfilename(
        title="Select an input image or video",
        initialdir=DIALOG_START_DIR if os.path.isdir(DIALOG_START_DIR) else None,
        filetypes=[("Images + Videos", exts), ("All files", "*.*")])
    root.destroy()
    return path or None


def make_output_path(input_path, kind):
    """Build OUTPUT_DIR/<original name>_<style>.<ext> without overwriting existing files."""
    stem = os.path.splitext(os.path.basename(input_path))[0]
    stem = re.sub(r"[^\w\-]+", "_", stem).strip("_")[:60]
    stem = stem.encode("ascii", "ignore").decode() or "output"
    ext = ".png" if kind == "image" else ".mp4"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path, n = os.path.join(OUTPUT_DIR, f"{stem}_{STYLE_NAME}{ext}"), 2
    while os.path.exists(path):
        path = os.path.join(OUTPUT_DIR, f"{stem}_{STYLE_NAME}_{n}{ext}")
        n += 1
    return path
