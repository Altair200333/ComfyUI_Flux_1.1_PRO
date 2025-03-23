import requests
from PIL import Image
import io
import os
import configparser
import time
from enum import Enum
from .utils import *
from .constants import *
import random

VERSION_ID = f"{random.randint(1,9)}.{random.randint(0,99)}"

TOOLTIP_DEFINITIONS = {
    "image": "A Base64-encoded string representing the image you wish to modify. Can contain alpha mask if desired.",
    "prompt": "The description of the changes you want to make. This text guides the inpainting process, specifying features, styles, or modifications for the masked area.",
    "steps": "Number of steps for the image generation process. (min: 15, max: 50, default: 50)",
    "guidance": "Guidance strength for the image generation process. (min: 1.5, max: 100.0, default: 60)",
    "safety_tolerance": "Tolerance level for input and output moderation. Between 0 (most strict) and 6 (no moderation) (default: 6)",
    "output_format": "Output format for the generated image. Can be 'jpeg' or 'png' (default: jpeg)",
    "mask": "A Base64-encoded string representing a mask for the areas you want to modify in the image. The mask should be the same dimensions as the image and in black and white. Black areas (0%) indicate no modification, while white areas (100%) specify areas for inpainting. Optional if you provide an alpha mask in the original image.",
    "seed": "Optional seed for reproducibility.",
    "prompt_upsampling": "Whether to perform upsampling on the prompt. If active, automatically modifies the prompt for more creative generation. (default: false)",
    "model": "Select the generation model: 'pro', 'ultra', or 'ultra_raw'.",
    "aspect_ratio": "Aspect ratio of the generated image, e.g., '16:9'.",
    "image_prompt": "Base64 encoded image to remix (if any); leave empty for no image prompt.",
    "image_prompt_strength": "Blend strength between the prompt and image prompt.",
}


class Status(Enum):
    PENDING = "Pending"
    READY = "Ready"
    ERROR = "Error"


class FluxModel(Enum):
    PRO = "pro"
    ULTRA = "ultra"
    ULTRA_RAW = "ultra_raw"


class ConfigLoader:
    def __init__(self):
        curr = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(curr, "config.ini")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Config file not found at {cfg_path}")
        self.config = configparser.ConfigParser()
        self.config.read(cfg_path)
        self.set_api_key()
        self.set_base_url()

    def _get_required_config_value(self, section, option):
        if not self.config.has_section(section) or not self.config.has_option(
            section, option
        ):
            raise KeyError(f"{option} not found in {section}")
        value = self.config[section][option]
        if not value:
            raise KeyError(f"{option} cannot be empty")
        return value

    def set_api_key(self):
        os.environ["API_KEY"] = self._get_required_config_value("API", "API_KEY")

    def set_base_url(self):
        os.environ["BASE_URL"] = self._get_required_config_value("API", "BASE_URL")


class FluxApiClient:
    def __init__(self, api_key, base_url="https://api.us1.bfl.ai"):
        self.api_key = api_key
        self.base_url = base_url

    def _make_headers(self):
        return {"Content-Type": "application/json", "X-Key": self.api_key}

    def send_request(self, method, endpoint, payload=None, timeout=30):
        url = f"{self.base_url}{endpoint}"
        try:
            return requests.request(
                method,
                url,
                json=payload,
                headers=self._make_headers(),
                timeout=timeout,
            )
        except requests.exceptions.RequestException as e:
            error_info = sanitize_response(e)
            raise RuntimeError(f"Request error {method} {url}: {error_info}")

    def submit_job(self, endpoint, payload):
        resp = self.send_request("POST", endpoint, payload)
        if resp.status_code != 200:
            try:
                error_data = resp.json()
            except Exception:
                error_data = {"error": resp.text}
            sanitized_error = sanitize_response(error_data)
            raise RuntimeError(
                f"Job submission error {resp.status_code}: {sanitized_error}"
            )
        response_json = resp.json()
        task_id = response_json.get("id")
        if not task_id:
            sanitized_error = sanitize_response(response_json)
            raise RuntimeError(f"No task id in response: {sanitized_error}")
        return task_id

    def poll_result(self, task_id, output_format="jpeg", max_attempts=MAX_ATTEMPTS):
        """
        Poll the Flux API for the result of the submitted job.
        Uses exponential backoff for attempts 1-4 (wait times: 1s, 2s, 4s, 8s)
        and then cycles through wait times of 2s, 4s, and 8s for subsequent attempts.
        """
        for attempt in range(1, max_attempts + 1):
            if attempt <= 4:
                wait_time = 2 ** (attempt - 1)
            else:
                wait_time = 2 ** (((attempt - 5) % 3) + 1)

            time.sleep(wait_time)
            response = self.send_request("GET", f"/v1/get_result?id={task_id}")
            if response.status_code != 200:
                continue
            data = response.json()
            if data.get("status") != Status.READY.value:
                continue
            sample_url = data.get("result", {}).get("sample")
            if not sample_url:
                raise RuntimeError(f"No sample URL for task {task_id}")
            img_resp = requests.get(sample_url, timeout=30)
            if img_resp.status_code != 200:
                raise RuntimeError(f"Error fetching image: {img_resp.status_code}")
            return Image.open(io.BytesIO(img_resp.content)).convert("RGB")
        raise RuntimeError(f"Max attempts reached for task {task_id}")

    def call_api_job(
        self, endpoint, payload, output_format="jpeg", max_attempts=MAX_ATTEMPTS
    ):
        tid = self.submit_job(endpoint, payload)
        return self.poll_result(tid, output_format, max_attempts)

    def inpaint(
        self,
        image,
        prompt,
        steps,
        guidance,
        safety_tolerance,
        output_format,
        seed=None,
        mask=None,
        prompt_upsampling=False,
        max_attempts=MAX_ATTEMPTS,
    ):
        if isinstance(prompt, list):
            prompt = " ".join(map(str, prompt))
        img_b64, img_size = tensor_to_base64(image, mode="RGB")
        mask_b64 = mask_to_base64(mask[0], img_size) if mask is not None else None
        payload = {
            "image": img_b64,
            "prompt": prompt,
            "steps": steps,
            "guidance": guidance,
            "safety_tolerance": safety_tolerance,
            "output_format": output_format,
            "prompt_upsampling": prompt_upsampling,
        }
        if mask_b64:
            payload["mask"] = mask_b64
        if seed is not None:
            payload["seed"] = seed
        res_img = self.call_api_job(
            endpoint="/v1/flux-pro-1.0-fill",
            payload=payload,
            output_format=output_format,
            max_attempts=max_attempts,
        )
        return pil_to_tensor(res_img)

    def flux_generate(
        self,
        model,
        prompt,
        image_prompt=None,
        prompt_upsampling=False,
        seed=None,
        safety_tolerance=6,
        output_format="png",
        aspect_ratio="16:9",
        image_prompt_strength=0.1,
        max_attempts=MAX_ATTEMPTS,
    ):
        model_str = model.value if hasattr(model, "value") else model
        if model_str == "pro":
            width, height = get_dimensions_from_ratio(aspect_ratio)
            payload = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "prompt_upsampling": prompt_upsampling,
                "safety_tolerance": safety_tolerance,
                "output_format": output_format,
            }

            if image_prompt is not None:
                img_b64, _ = tensor_to_base64(image_prompt, mode="RGB")
                payload["image_prompt"] = img_b64

            endpoint = "/v1/flux-pro-1.1"
        elif model_str in ("ultra", "ultra_raw"):
            is_raw = model_str == "ultra_raw"
            payload = {
                "prompt": prompt,
                "prompt_upsampling": prompt_upsampling,
                "aspect_ratio": aspect_ratio,
                "safety_tolerance": safety_tolerance,
                "output_format": output_format,
                "raw": is_raw,
            }

            if image_prompt is not None:
                img_b64, _ = tensor_to_base64(image_prompt, mode="RGB")
                payload["image_prompt"] = img_b64
                payload["image_prompt_strength"] = image_prompt_strength

            endpoint = "/v1/flux-pro-1.1-ultra"
        else:
            raise ValueError(
                'Unsupported model type: choose one of "pro", "ultra", "ultra_raw"'
            )

        if seed is not None:
            payload["seed"] = seed

        res_img = self.call_api_job(
            endpoint=endpoint,
            payload=payload,
            output_format=output_format,
            max_attempts=max_attempts,
        )
        return pil_to_tensor(res_img)


class FluxProInpaint:
    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "process"
    CATEGORY = "BFL"

    def __init__(self):
        self.config_loader = ConfigLoader()
        self.api_key = os.environ.get("API_KEY")
        self.base_url = os.environ.get("BASE_URL", "https://api.us1.bfl.ai")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": TOOLTIP_DEFINITIONS["image"]}),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": TOOLTIP_DEFINITIONS["prompt"],
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 50,
                        "min": 15,
                        "max": 50,
                        "tooltip": TOOLTIP_DEFINITIONS["steps"],
                    },
                ),
                "guidance": (
                    "FLOAT",
                    {
                        "default": 60.0,
                        "min": 1.5,
                        "max": 100.0,
                        "tooltip": TOOLTIP_DEFINITIONS["guidance"],
                    },
                ),
                "safety_tolerance": (
                    "INT",
                    {
                        "default": 6,
                        "min": 0,
                        "max": 6,
                        "tooltip": TOOLTIP_DEFINITIONS["safety_tolerance"],
                    },
                ),
                "output_format": (
                    ("jpeg", "png"),
                    {
                        "default": "jpeg",
                        "tooltip": TOOLTIP_DEFINITIONS["output_format"],
                    },
                ),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": TOOLTIP_DEFINITIONS["mask"]}),
                "seed": (
                    "INT",
                    {"default": -1, "tooltip": TOOLTIP_DEFINITIONS["seed"]},
                ),
                "prompt_upsampling": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": TOOLTIP_DEFINITIONS["prompt_upsampling"],
                    },
                ),
            },
        }

    def process(
        self,
        image,
        prompt,
        steps,
        guidance,
        safety_tolerance,
        output_format,
        seed=-1,
        mask=None,
        prompt_upsampling=False,
    ):
        client = FluxApiClient(api_key=self.api_key, base_url=self.base_url)
        seed_val = None if seed == -1 else seed
        res = client.inpaint(
            image=image,
            prompt=prompt,
            steps=steps,
            guidance=guidance,
            safety_tolerance=safety_tolerance,
            output_format=output_format,
            seed=seed_val,
            mask=mask,
            prompt_upsampling=prompt_upsampling,
        )
        return (res, VERSION_ID)


class FluxGenerate:
    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "process"
    CATEGORY = "BFL"

    def __init__(self):
        self.config_loader = ConfigLoader()
        self.api_key = os.environ.get("API_KEY")
        self.base_url = os.environ.get("BASE_URL", "https://api.us1.bfl.ai")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": TOOLTIP_DEFINITIONS["prompt"],
                    },
                ),
                "model": (
                    ("pro", "ultra", "ultra_raw"),
                    {
                        "default": "pro",
                        "tooltip": TOOLTIP_DEFINITIONS["model"],
                    },
                ),
                "safety_tolerance": (
                    "INT",
                    {
                        "default": 6,
                        "min": 0,
                        "max": 6,
                        "tooltip": TOOLTIP_DEFINITIONS["safety_tolerance"],
                    },
                ),
                "output_format": (
                    ("jpeg", "png"),
                    {
                        "default": "png",
                        "tooltip": TOOLTIP_DEFINITIONS["output_format"],
                    },
                ),
            },
            "optional": {
                "prompt_upsampling": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": TOOLTIP_DEFINITIONS["prompt_upsampling"],
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "tooltip": TOOLTIP_DEFINITIONS["seed"],
                    },
                ),
                "aspect_ratio": (
                    ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "9:21"],
                    {
                        "default": "16:9",
                        "tooltip": TOOLTIP_DEFINITIONS["aspect_ratio"],
                    },
                ),
                "image_prompt": (
                    "IMAGE",
                    {
                        "tooltip": TOOLTIP_DEFINITIONS["image_prompt"],
                    },
                ),
                "image_prompt_strength": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": TOOLTIP_DEFINITIONS["image_prompt_strength"],
                    },
                ),
            },
        }

    def process(
        self,
        prompt,
        model,
        safety_tolerance,
        output_format,
        prompt_upsampling=False,
        seed=-1,
        aspect_ratio="16:9",
        image_prompt=None,
        image_prompt_strength=0.1,
    ):
        client = FluxApiClient(api_key=self.api_key, base_url=self.base_url)
        seed_val = None if seed == -1 else seed
        res = client.flux_generate(
            model=model,
            prompt=prompt,
            image_prompt=image_prompt,
            prompt_upsampling=prompt_upsampling,
            seed=seed_val,
            safety_tolerance=safety_tolerance,
            output_format=output_format,
            aspect_ratio=aspect_ratio,
            image_prompt_strength=image_prompt_strength,
        )
        return (res, VERSION_ID)
