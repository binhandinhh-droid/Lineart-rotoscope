"""
Global settings for the colored line-art style.
Edit the paths and the CFG dictionary here; every other module imports from this file.
"""

# =====================================================================
# SETTINGS
# =====================================================================
STYLE_NAME = "lineart_color"
OUTPUT_DIR = r"D:\project\Rotoscope_animation\outputs"
DIALOG_START_DIR = r"D:\project\Rotoscope_animation\input_videos"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".flv")

CFG = {
    "PARSE_MODEL_NAME": "mattmdjaga/segformer_b2_clothes",
    "USE_FP16": True,            # only takes effect when CUDA is available
    "MAX_FRAMES": None,          # None = process the whole video

    # line work
    "LINE_MAX_SIDE": 960,                 # max side (px) of the low-res working image
    "SEG_SMOOTH_K": 9,                    # median-blur kernel applied to the label map
    "LINE_THICKNESS_FRAC": 0.003,         # main outline thickness (fraction of the long side)
    "DETAIL_THICKNESS_FRAC": 0.002,       # detail stroke thickness (fraction of the long side)
    "LINE_COLOR": (25, 25, 25),
    "DETAIL_LINES": True,                 # draw inner detail strokes (clothing folds, ...)
    "DETAIL_BLUR_FRAC": 0.002,            # pre-Canny blur sigma (fraction of the long side)
    "CANNY_LOW": 25,
    "CANNY_HIGH": 70,
    "MIN_STROKE_LEN": 22,                 # drop detail strokes shorter than this (pixels)
    "DETAIL_EDGE_MARGIN_FRAC": 0.012,     # keep detail strokes away from the silhouette / label borders
    "FACE_LABELS": (11,),                 # face label ids (11 = Face) -> no detail strokes here
    "FACE_DETAIL_MARGIN_FRAC": 0.006,     # grow the face zone a little further

    # paper background
    "PAPER_COLOR": (250, 249, 245),
    "PAPER_NOISE": 2.5,

    # representative color for each body part
    "COLOR_SAMPLE_ERODE_FRAC": 0.006,     # erosion used to sample only the "core" of each label
    "COLOR_MIN_PIXELS": 40,               # fall back to the whole label if the core is smaller
    "COLOR_MAX_SAMPLES": 20000,           # sub-sample large labels to this many pixels
    "COLOR_TEMPORAL_ALPHA": 0.25,         # 1.0 = disable temporal color smoothing
    "FILL_SATURATION": 1.15,
    "FILL_BRIGHTNESS": 1.0,
}
