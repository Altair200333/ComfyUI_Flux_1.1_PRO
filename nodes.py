import requests
from PIL import Image
import io
import os
import configparser
import time
from enum import Enum
from .utils import *


class Status(Enum):
    PENDING = "Pending"
    READY = "Ready"
    ERROR = "Error"


class ConfigLoader:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.ini")

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found at {config_path}. Please ensure config.ini exists in the same directory as the script."
            )

        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.set_api_key()
        self.set_base_url()

    def _get_required_config_value(self, section, option):
        if not self.config.has_section(section):
            raise KeyError(f"Section '{section}' not found in config file")
        if not self.config.has_option(section, option):
            raise KeyError(f"{option} not found in {section} section")
        value = self.config[section][option]
        if not value:
            raise KeyError(f"{option} cannot be empty")
        return value

    def set_api_key(self):
        try:
            api_key = self._get_required_config_value("API", "API_KEY")
            os.environ["API_KEY"] = api_key
        except KeyError as e:
            print(f"[FLUX INPAINT] Error setting API_KEY: {str(e)}")
            print(
                "[FLUX INPAINT] Please ensure config.ini contains a valid API_KEY under the [API] section"
            )
            raise

    def set_base_url(self):
        try:
            base_url = self._get_required_config_value("API", "BASE_URL")
            os.environ["BASE_URL"] = base_url
        except KeyError as e:
            print(f"[FLUX INPAINT] Error setting BASE_URL: {str(e)}")
            print(
                "[FLUX INPAINT] Please ensure your config.ini contains a valid BASE_URL under the [API] section"
            )
            raise


class FluxProInpaint:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "BFL"

    def __init__(self):
        try:
            self.config_loader = ConfigLoader()
            self.api_key = os.environ.get("API_KEY")
            self.base_url = os.environ.get("BASE_URL", "https://api.us1.bfl.ai")
            if not self.api_key:
                raise ValueError(
                    "API_KEY not found in environment variables after loading config"
                )

            print(f"[FLUX INPAINT] Initialized with BASE_URL: {self.base_url}")
            print(
                f"[FLUX INPAINT] API_KEY is set: {self.api_key[:5]}...{self.api_key[-5:]}"
            )
        except Exception as e:
            print(f"[FLUX INPAINT] Initialization Error: {str(e)}")
            print(
                "[FLUX INPAINT] Please ensure config.ini is properly set up with API credentials"
            )
            raise

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "A Base64-encoded string representing the image you wish to modify. Can contain alpha mask if desired."
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "The description of the changes you want to make. This text guides the inpainting process, specifying features, styles, or modifications for the masked area.",
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 50,
                        "min": 15,
                        "max": 50,
                        "tooltip": "Number of steps for the image generation process. (min: 15, max: 50, default: 50)",
                    },
                ),
                "guidance": (
                    "FLOAT",
                    {
                        "default": 60.0,
                        "min": 1.5,
                        "max": 100.0,
                        "tooltip": "Guidance strength for the image generation process. (min: 1.5, max: 100, default: 60)",
                    },
                ),
                "safety_tolerance": (
                    "INT",
                    {
                        "default": 6,
                        "min": 0,
                        "max": 6,
                        "tooltip": "Tolerance level for input and output moderation. Between 0 (most strict) and 6 (least strict) (default: 2)",
                    },
                ),
                "output_format": (
                    ["jpeg", "png"],
                    {
                        "default": "jpeg",
                        "tooltip": "Output format for the generated image. Can be 'jpeg' or 'png' (default: jpeg)",
                    },
                ),
            },
            "optional": {
                "mask": (
                    "MASK",
                    {
                        "tooltip": "A Base64-encoded string representing a mask for the areas you want to modify in the image. The mask should be the same dimensions as the image and in black and white. Black areas (0%) indicate no modification, while white areas (100%) specify areas for inpainting. Optional if you provide an alpha mask in the original image."
                    },
                ),
                "seed": (
                    "INT",
                    {"default": -1, "tooltip": "Optional seed for reproducibility."},
                ),
                "prompt_upsampling": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Whether to perform upsampling on the prompt. If active, automatically modifies the prompt for more creative generation. (default: false)",
                    },
                ),
            },
        }

    def _build_payload(
        self,
        img_base64,
        prompt,
        steps,
        guidance,
        safety_tolerance,
        output_format,
        prompt_upsampling,
        mask_base64=None,
        seed=-1,
    ):
        payload = {
            "image": img_base64,
            "prompt": prompt,
            "steps": steps,
            "guidance": guidance,
            "output_format": output_format,
            "safety_tolerance": safety_tolerance,
            "prompt_upsampling": prompt_upsampling,
        }
        if mask_base64:
            payload["mask"] = mask_base64
        if seed != -1:
            payload["seed"] = seed
        return payload

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
        """
        In ComfyUI, an IMAGE tensor is usually shape [B, H, W, C].
        We'll handle only the first batch here (if multiple images are fed in, only the first will be inpainted).
        """
        preserve_alpha = image.shape[-1] == 4

        if isinstance(prompt, list):
            prompt = " ".join(str(p) for p in prompt)

        print(f"[FLUX INPAINT] preserve_alpha set to: {preserve_alpha}")
        try:
            print(f"[FLUX INPAINT] Processing image with prompt: '{prompt}'")
            pil_image = tensor_to_pil(image, mode="RGBA" if preserve_alpha else "RGB")
            img_base64 = pil_to_base64(pil_image)
            print("[FLUX INPAINT] Converted input image to base64.")

            mask_base64 = None
            if mask is not None:
                mask_base64 = mask_to_base64(mask[0], pil_image.size)
                print("[FLUX INPAINT] Processed mask to base64.")

            url = f"{self.base_url}/v1/flux-pro-1.0-fill"
            headers = {"Content-Type": "application/json", "X-Key": self.api_key}
            payload = self._build_payload(
                img_base64,
                prompt,
                steps,
                guidance,
                safety_tolerance,
                output_format,
                prompt_upsampling,
                mask_base64,
                seed,
            )

            print(f"[FLUX INPAINT] Sending request to: {url}")
            try:
                resp = requests.request(
                    "POST", url, json=payload, headers=headers, timeout=30
                )
            except requests.exceptions.RequestException as e:
                raise RuntimeError(
                    f"[FLUX INPAINT] Network error during POST request: {e}"
                )

            print(f"[FLUX INPAINT] Response Status: {resp.status_code}")
            if resp.status_code == 200:
                resp_json = resp.json()
                task_id = resp_json.get("id")
                if not task_id:
                    raise RuntimeError(
                        f"[FLUX INPAINT] Error: No 'id' in server response. Full response: {resp_json}"
                    )

                print(f"[FLUX INPAINT] Task ID received: {task_id}")
                return self.get_result(task_id, output_format, preserve_alpha)
            else:
                try:
                    err_details = resp.json()
                    print(f"[FLUX INPAINT] Server Error JSON: {err_details}")
                except:
                    print("[FLUX INPAINT] Error reading JSON from response:")
                    print(resp.text)
                raise RuntimeError(
                    f"[FLUX INPAINT] Non-200 response ({resp.status_code}). See logs above for details."
                )

        except Exception as e:
            print(f"[FLUX INPAINT] Unexpected Error: {e}")
            import traceback

            traceback.print_exc()
            raise

    def get_result(self, task_id, output_format, preserve_alpha=False, max_attempts=15):
        attempt = 1
        while attempt <= max_attempts:
            wait_time = min((2**attempt + 1) * 0.5, 15)
            print(
                f"[FLUX INPAINT] Waiting {wait_time} seconds before polling attempt {attempt}…"
            )
            time.sleep(wait_time)

            get_url = f"{self.base_url}/v1/get_result?id={task_id}"
            headers = {"X-Key": self.api_key}

            try:
                resp = requests.request("GET", get_url, headers=headers, timeout=30)
            except requests.exceptions.RequestException as e:
                print(f"[FLUX INPAINT] Network error during GET request: {e}")
                attempt += 1
                continue

            print(
                f"[FLUX INPAINT] Poll attempt {attempt}, status code: {resp.status_code}"
            )
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status")
                print(f"[FLUX INPAINT] Task status: {status}")

                if status == Status.READY.value:
                    sample_url = data.get("result", {}).get("sample")
                    if not sample_url:
                        raise RuntimeError(
                            f"[FLUX INPAINT] No sample URL in 'result' for task {task_id}."
                        )
                    try:
                        img_resp = requests.request("GET", sample_url, timeout=30)
                        if img_resp.status_code != 200:
                            raise RuntimeError(
                                f"[FLUX INPAINT] Error fetching final image: {img_resp.status_code}"
                            )
                        pil_img = Image.open(io.BytesIO(img_resp.content))
                        if preserve_alpha and output_format.lower() == "png":
                            pil_img = pil_img.convert("RGBA")
                        else:
                            pil_img = pil_img.convert("RGB")
                        final_tensor = pil_to_tensor(pil_img)
                    except Exception as e:
                        print(f"[FLUX INPAINT] Error processing final image: {e}")
                        import traceback

                        traceback.print_exc()
                        raise RuntimeError(
                            "[FLUX INPAINT] Error processing final image."
                        )

                    print(f"[FLUX INPAINT] Retrieved final image for task {task_id}")
                    return (final_tensor,)

                elif status == Status.PENDING.value:
                    print("[FLUX INPAINT] Task still Pending; will retry.")
                else:
                    raise RuntimeError(f"[FLUX INPAINT] Unexpected status '{status}'.")
            else:
                print(f"[FLUX INPAINT] Non-200 while polling: {resp.status_code}")
            attempt += 1
        raise RuntimeError(
            f"[FLUX INPAINT] Max attempts reached for task_id {task_id}. Aborting."
        )

    def sanitize_response(self, data):
        """
        Optional helper if you want to log responses but remove big base64 data.
        Not currently called, but you can use it if desired.
        """
        if isinstance(data, dict):
            sanitized = {}
            for k, v in data.items():
                if isinstance(v, str) and len(v) > 100:
                    sanitized[k] = "[BASE64 DATA REDACTED]"
                else:
                    sanitized[k] = self.sanitize_response(v)
            return sanitized
        elif isinstance(data, list):
            return [self.sanitize_response(item) for item in data]
        else:
            return data
