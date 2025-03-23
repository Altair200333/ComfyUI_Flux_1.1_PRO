from .nodes import FluxProInpaint, FluxGenerate

NODE_CLASS_MAPPINGS = {
    "FluxProInpaint": FluxProInpaint,
    "FluxGenerate": FluxGenerate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxProInpaint": "Flux Pro 1.0 Fill [Inpaint]",
    "FluxGenerate": "Flux Generate 1.1 [Pro, Ultra, Raw]",
}
