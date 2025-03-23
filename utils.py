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


def tensor_to_base64(tensor, mode="RGB", fmt="PNG"):
    pil_img = tensor_to_pil(tensor, mode)
    return pil_to_base64(pil_img, fmt), pil_img.size


def mask_to_base64(mask_tensor, expected_size):
    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]
    pil_mask = Image.fromarray(mask_np, mode="L")
    if pil_mask.size != expected_size:
        raise ValueError(
            f"Error: Mask dimensions {pil_mask.size} do not match expected image dimensions {expected_size}."
        )
    return pil_to_base64(pil_mask)


def pil_to_tensor(pil_img):
    final_np = np.array(pil_img, dtype=np.float32) / 255.0
    return torch.from_numpy(final_np)[None, ...]


def sanitize_response(data):
    if isinstance(data, dict):
        sanitized = {}
        for k, v in data.items():
            if isinstance(v, str) and len(v) > 300:
                sanitized[k] = "[LARGE DATA REDACTED]"
            else:
                sanitized[k] = sanitize_response(v)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_response(item) for item in data]
    else:
        return data
