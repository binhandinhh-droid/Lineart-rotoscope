"""Small shared helpers: device selection, tensor conversion, morphology kernels."""
import cv2
import torch


def get_device():
    """Pick the best available torch device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def size_hw(size, default):
    """Convert a HuggingFace processor `size` field into an (height, width) tuple."""
    if isinstance(size, dict):
        if "height" in size and "width" in size:
            return int(size["height"]), int(size["width"])
        if "shortest_edge" in size:
            return int(size["shortest_edge"]), int(size["shortest_edge"])
    if isinstance(size, int):
        return size, size
    return default


def antialias_ok(device):
    """Antialiased interpolation is only supported on CUDA and CPU."""
    return device.type in ("cuda", "cpu")


def to_nchw_float(rgb_gpu):
    """HWC uint8 tensor -> 1xCxHxW float tensor in [0, 1]."""
    return rgb_gpu.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)


def ell(d):
    """Elliptical structuring element with an odd diameter of at least 1."""
    d = max(1, int(d)) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
