
from .image.image_crop_alpha_node import ImageCropByAlpha
from .image.canvas_fusion_node import CanvasFusionNode
NODE_CLASS_MAPPINGS = {
    "ImageCropByAlpha": ImageCropByAlpha,
    "CanvasFusionNode": CanvasFusionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCropByAlpha": "ImageCropAlphaNode (Image)",
    "CanvasFusionNode": "CanvasFusionNode (Image)"
}