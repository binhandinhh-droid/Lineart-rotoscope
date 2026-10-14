"""Prepare the low-resolution working copies of the label map and the image."""
import cv2


def lowres_seg_and_rgb(orig_rgb, seg_map, cfg):
    """Return the label map and RGB image at LINE_MAX_SIDE resolution, with the label map median-filtered."""
    H, W = seg_map.shape
    s = min(1.0, cfg["LINE_MAX_SIDE"] / max(H, W))
    if s < 1.0:
        wh = (max(1, int(round(W * s))), max(1, int(round(H * s))))
        seg_s = cv2.resize(seg_map, wh, interpolation=cv2.INTER_NEAREST)
        rgb_s = cv2.resize(orig_rgb, wh, interpolation=cv2.INTER_AREA)
    else:
        seg_s, rgb_s = seg_map, orig_rgb

    # Median blur removes small speckles in the label map
    k = int(cfg["SEG_SMOOTH_K"])
    if k > 1:
        seg_s = cv2.medianBlur(seg_s, k | 1)
    return seg_s, rgb_s
