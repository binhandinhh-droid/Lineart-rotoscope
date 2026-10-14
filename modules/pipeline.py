"""Single-frame pipeline: segmentation -> ink -> part colors -> composition."""
import torch

from preprocess import lowres_seg_and_rgb
from ink import build_ink
from colors import part_color_table
from compose import compose_color_lineart


def make_frame_fn(parser, cfg):
    """Return a function that converts one RGB frame (HxWx3 uint8) into a colored line-art frame."""
    def frame_fn(orig):
        seg_map = parser.segment(torch.from_numpy(orig).to(parser.device))
        seg_s, rgb_s = lowres_seg_and_rgb(orig, seg_map, cfg)
        ink = build_ink(rgb_s, seg_s, seg_map.shape, cfg)
        lut = part_color_table(rgb_s, seg_s, cfg)
        return compose_color_lineart(ink, seg_s, lut, cfg)
    return frame_fn
