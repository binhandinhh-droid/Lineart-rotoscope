"""Compositing: paper background -> flat part colors -> ink strokes on top."""
from functools import lru_cache

import numpy as np
import cv2

from config import CFG


@lru_cache(maxsize=1)
def get_paper(h, w):
    """Paper background with subtle low-frequency noise (cached per resolution)."""
    base = np.empty((h, w, 3), np.float32)
    base[:] = np.asarray(CFG["PAPER_COLOR"], np.float32)
    if CFG["PAPER_NOISE"] > 0:
        n = np.random.default_rng(0).standard_normal((max(2, h // 2), max(2, w // 2))).astype(np.float32)
        n = cv2.resize(cv2.GaussianBlur(n, (0, 0), 0.8), (w, h), interpolation=cv2.INTER_LINEAR)
        base += n[..., None] * float(CFG["PAPER_NOISE"])
    return np.clip(base, 0, 255).astype(np.uint8)


def compose_color_lineart(ink, seg_s, lut, cfg):
    """Paper -> flat color for each body part -> ink strokes drawn over it."""
    H, W = ink.shape
    sh, sw = seg_s.shape
    out = get_paper(H, W).copy()

    fg = seg_s != 0
    ys, xs = np.flatnonzero(fg.any(axis=1)), np.flatnonzero(fg.any(axis=0))
    if ys.size and xs.size:
        # Only process the person's bounding box (low-res coords -> full-res coords)
        y0, y1 = max(0, int(ys[0]) - 2), min(sh, int(ys[-1]) + 3)
        x0, x1 = max(0, int(xs[0]) - 2), min(sw, int(xs[-1]) + 3)
        fy0, fy1 = int(round(y0 * H / sh)), min(H, int(round(y1 * H / sh)))
        fx0, fx1 = int(round(x0 * W / sw)), min(W, int(round(x1 * W / sw)))
        size = (fx1 - fx0, fy1 - fy0)
        fill = cv2.resize(lut[seg_s[y0:y1, x0:x1]], size, interpolation=cv2.INTER_LINEAR)
        m = cv2.resize(fg[y0:y1, x0:x1].astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
        m = np.clip((m - 0.25) / 0.5, 0.0, 1.0)  # soft edge; this sits underneath the ink
        region = out[fy0:fy1, fx0:fx1]
        sel = m > 0.004
        if sel.any():
            al = m[sel][:, None]
            region[sel] = np.clip(region[sel].astype(np.float32) * (1.0 - al)
                                  + fill[sel].astype(np.float32) * al, 0, 255).astype(np.uint8)

    # Blend the ink over everything
    mk = ink > 0.004
    if mk.any():
        al = ink[mk][:, None]
        col = np.asarray(cfg["LINE_COLOR"], np.float32)
        out[mk] = np.clip(out[mk].astype(np.float32) * (1.0 - al) + col * al, 0, 255).astype(np.uint8)
    return out
