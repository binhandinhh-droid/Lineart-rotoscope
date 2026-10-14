"""Representative flat fill color for each body part, smoothed over time."""
import numpy as np
import cv2

from utils import ell

# label id -> raw color, temporally smoothed across frames
_COLORS = {}


def reset_color_memory():
    """Forget the temporal color history (call before starting a new image/video)."""
    _COLORS.clear()


def adjust_color(rgb, sat, val):
    """Scale saturation and brightness of a single RGB color."""
    c = np.clip(rgb, 0, 255).astype(np.uint8).reshape(1, 1, 3)
    hsv = cv2.cvtColor(c, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * val, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)[0, 0]


def part_color_table(rgb_s, seg_s, cfg):
    """Build a (256, 3) LUT: LUT[label] = median color of that part; LUT[0] = paper color."""
    r = max(1, int(round(cfg["COLOR_SAMPLE_ERODE_FRAC"] * max(seg_s.shape))))
    ker = ell(2 * r + 1)
    # "Core" pixels of each label (away from its borders)
    interior = cv2.erode(seg_s, ker) == cv2.dilate(seg_s, ker)
    a, cap = cfg["COLOR_TEMPORAL_ALPHA"], int(cfg["COLOR_MAX_SAMPLES"])

    lut = np.empty((256, 3), np.uint8)
    lut[:] = np.asarray(cfg["PAPER_COLOR"], np.uint8)
    for lab in np.unique(seg_s):
        if lab == 0:
            continue
        m = seg_s == lab
        sel = m & interior
        if int(sel.sum()) < cfg["COLOR_MIN_PIXELS"]:
            sel = m  # core too small -> use the whole label
        px = rgb_s[sel]
        if px.shape[0] == 0:
            continue
        if px.shape[0] > cap:
            px = px[:: int(np.ceil(px.shape[0] / cap))]  # sub-sample for speed
        raw = np.median(px, axis=0).astype(np.float32)
        # Exponential moving average over time to avoid color flicker
        old = _COLORS.get(int(lab))
        if old is not None and a < 1.0:
            raw = old * (1.0 - a) + raw * a
        _COLORS[int(lab)] = raw
        lut[lab] = adjust_color(raw, cfg["FILL_SATURATION"], cfg["FILL_BRIGHTNESS"])
    return lut
