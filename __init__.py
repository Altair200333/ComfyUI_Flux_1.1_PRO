from .nodes import FluxProInpaint, FluxGenerate, FluxProOutpaint

NODE_CLASS_MAPPINGS = {
    "FluxProInpaint": FluxProInpaint,
    "FluxGenerate": FluxGenerate,
    "FluxProOutpaint": FluxProOutpaint,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxGenerate": "Flux Generate 1.1 [Pro, Ultra, Raw]",
    "FluxProInpaint": "Flux Pro 1.0 Fill [Inpaint]",
    "FluxProOutpaint": "Flux Pro 1.0 Extend [Outpaint]",
}
