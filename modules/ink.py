"""Ink layer: main outlines from label boundaries plus optional inner detail strokes."""
import numpy as np
import cv2

from utils import ell


def filter_short(binary, min_len):
    """Remove connected components smaller than `min_len` pixels from a boolean mask."""
    if min_len <= 1 or not binary.any():
        return binary
    _, lab, stats, _ = cv2.connectedComponentsWithStats(binary.view(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_len
    keep[0] = False  # component 0 is the background
    return keep[lab]


def build_ink(rgb_s, seg_s, full_hw, cfg):
    """Build a float32 ink map in [0, 1] at full resolution (drawn at low resolution, then upscaled)."""
    H, W = full_hw
    sh, sw = seg_s.shape
    L = max(sh, sw)

    # (1) Main outline: boundaries between labels
    k3 = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(seg_s, k3) != cv2.erode(seg_s, k3)
    d_main = int(round(max(1.0, cfg["LINE_THICKNESS_FRAC"] * L))) - 1
    ink = edges.astype(np.uint8)
    if d_main > 1:
        ink = cv2.dilate(ink, ell(d_main))

    # (2) Inner detail strokes on the person (clothing folds, ...)
    if cfg["DETAIL_LINES"]:
        margin = max(1, int(round(cfg["DETAIL_EDGE_MARGIN_FRAC"] * L)))
        inner = cv2.erode((seg_s != 0).astype(np.uint8), ell(2 * margin + 1)).astype(bool)
        near_label = cv2.dilate(edges.astype(np.uint8), ell(2 * margin + 1)).astype(bool)
        gray = cv2.cvtColor(rgb_s, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (0, 0), max(0.8, cfg["DETAIL_BLUR_FRAC"] * L))
        stroke = (cv2.Canny(gray, cfg["CANNY_LOW"], cfg["CANNY_HIGH"]) > 0) & inner & ~near_label
        # Drop detail strokes inside the face region (label-boundary outlines are kept)
        face = np.isin(seg_s, cfg["FACE_LABELS"])
        if face.any():
            fm = max(1, int(round(cfg["FACE_DETAIL_MARGIN_FRAC"] * L)))
            face_zone = cv2.dilate(face.astype(np.uint8), ell(2 * fm + 1)).astype(bool)
            stroke &= ~face_zone
        stroke = filter_short(stroke, int(cfg["MIN_STROKE_LEN"]))
        st = stroke.astype(np.uint8)
        t_det = int(round(max(1.0, cfg["DETAIL_THICKNESS_FRAC"] * L)))
        if t_det > 1:
            st = cv2.dilate(st, ell(t_det))
        ink = np.maximum(ink, st)

    # (3) Smooth, upscale, then apply smoothstep for soft but crisp edges
    ink = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), 0.9)
    if (sh, sw) != (H, W):
        ink = cv2.resize(ink, (W, H), interpolation=cv2.INTER_LINEAR)
    x = np.clip((ink - 0.2) / 0.6, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)
