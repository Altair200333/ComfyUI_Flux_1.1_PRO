# ComfyUI Flux Pro 1.0 Fill [Inpaint] Extension

This extension adds inpainting capabilities to ComfyUI using the FLUX.1 Fill [pro] model. The node allows you to provide an input image and mask to perform inpainting on specific areas of the image.

## Installation

1. Clone this repository into your ComfyUI's `custom_nodes` directory:
```bash
cd /path/to/ComfyUI/custom_nodes/
git clone https://github.com/yourusername/ComfyUI_Flux_1.1_INPAINT.git
```

2. Edit the `config.ini` file in the extension directory to add your API key:
```ini
[API]
API_KEY=your_api_key_here
BASE_URL=https://api.us1.bfl.ai
```

3. Restart ComfyUI

## Usage

After installation, you'll find a new node called "Flux Pro 1.0 Fill [Inpaint]" in the "BFL" category. This node allows you to:

1. Provide an input image to be inpainted
2. Supply a mask (white areas will be inpainted, black areas preserved)
3. Specify a text prompt to guide the inpainting process
4. Configure various parameters like steps, guidance strength, etc.

### Node Inputs

- **image**: The input image to be inpainted
- **prompt**: Text description of what you want the inpainted areas to contain
- **steps**: Number of steps for image generation (15-50)
- **guidance**: Guidance strength for generation (1.5-100)
- **safety_tolerance**: Moderation level (0-6, lower is stricter)
- **output_format**: Output image format ("jpeg" or "png")
- **mask** (optional): The mask specifying areas to inpaint (white=inpaint, black=keep)
- **seed** (optional): Seed for reproducibility
- **prompt_upsampling** (optional): Whether to upsample the prompt

## API Information

This node uses the FLUX.1 Fill [pro] API endpoint:
```
https://api.us1.bfl.ai/v1/flux-pro-1.0-fill
```

For more information about the API, please refer to the official documentation.
