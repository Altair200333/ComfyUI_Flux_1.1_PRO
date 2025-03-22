import io
import numpy as np
import base64
import torch
from PIL import Image


def tensor_to_pil(tensor, mode="RGB"):
    arr = np.clip(255.0 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode=mode)


def pil_to_base64(pil_img, fmt="PNG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def mask_to_base64(mask_tensor, expected_size):
    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]
    pil_mask = Image.fromarray(mask_np, mode="L")
    if pil_mask.size != expected_size:
        print("[FLUX INPAINT] Resizing mask to match image dimensions...")
        pil_mask = pil_mask.resize(expected_size, Image.Resampling.NEAREST)
    return pil_to_base64(pil_mask)


def pil_to_tensor(pil_img):
    final_np = np.array(pil_img, dtype=np.float32) / 255.0
    return torch.from_numpy(final_np)[None, ...]
