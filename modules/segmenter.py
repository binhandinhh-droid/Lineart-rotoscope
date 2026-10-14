"""Human parsing model (SegFormer) that produces a per-pixel body-part label map."""
import torch
import torch.nn.functional as F
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

from utils import size_hw, antialias_ok, to_nchw_float


class Parser:
    def __init__(self, model_name, device, dtype):
        self.device, self.dtype = device, dtype
        proc = SegformerImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(device, dtype=dtype).eval()
        self.in_h, self.in_w = size_hw(getattr(proc, "size", None), (512, 512))
        self.mean = torch.tensor(proc.image_mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(proc.image_std, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    @torch.inference_mode()
    def segment(self, rgb_gpu):
        """Return an (H, W) uint8 label map for an HWC uint8 RGB tensor."""
        H, W = rgb_gpu.shape[:2]
        x = F.interpolate(to_nchw_float(rgb_gpu), size=(self.in_h, self.in_w), mode="bilinear",
                          align_corners=False, antialias=antialias_ok(self.device))
        x = ((x - self.mean) / self.std).to(self.dtype)
        # Upsample logits back to the original resolution before the argmax
        logits = F.interpolate(self.model(pixel_values=x).logits, size=(H, W),
                               mode="bilinear", align_corners=False)
        return logits.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
