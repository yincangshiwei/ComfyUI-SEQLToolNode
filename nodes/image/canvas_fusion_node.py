import torch
import numpy as np
from PIL import Image, ImageDraw, ImageColor

# Helper to check Pillow version for resampling method
_HAS_RESAMPLING_ENUM = hasattr(Image, 'Resampling')

class CanvasFusionNode:
    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image",)
    FUNCTION = "fuse_images"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "foreground_image": ("IMAGE",),
                "position": (["center", "left", "right", "top", "bottom",
                              "top_left", "top_right", "bottom_left", "bottom_right"],
                             {"default": "center"}),
                "padding_x": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}), # Allow negative padding for fine-tuning
                "padding_y": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}), # Allow negative padding
                # New scaling options
                "scale_foreground_if_larger": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no"}),
                "fit_within_canvas_ratio": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 2.0, "step": 0.01}), # 1.0 = 100% of canvas

                # Canvas creation params (if no canvas_image)
                "canvas_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "canvas_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "canvas_color": ("STRING", {"default": "#FFFFFFFF"}),  # Default white opaque (RGBA)
            },
            "optional": {
                "canvas_image": ("IMAGE",),
            }
        }

    def _tensor_to_pil(self, tensor_image: torch.Tensor) -> Image.Image:
        img_np = tensor_image[0].cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        if img_np.shape[-1] == 3:
            return Image.fromarray(img_np, 'RGB')
        elif img_np.shape[-1] == 4:
            return Image.fromarray(img_np, 'RGBA')
        elif img_np.shape[-1] == 1:
            return Image.fromarray(img_np.squeeze(axis=-1), 'L')
        else:
            raise ValueError(f"Unsupported number of channels: {img_np.shape[-1]}")

    def _pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        # Ensure output is RGBA for consistency
        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')

        img_np = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)

    def get_resampling_method(self):
        if _HAS_RESAMPLING_ENUM:
            return Image.Resampling.LANCZOS
        else:
            return Image.LANCZOS # For older Pillow versions

    def fuse_images(self, foreground_image: torch.Tensor, position: str,
                    padding_x: int, padding_y: int,
                    scale_foreground_if_larger: bool, fit_within_canvas_ratio: float,
                    canvas_width: int, canvas_height: int, canvas_color: str,
                    canvas_image: torch.Tensor = None):

        fg_pil_initial = self._tensor_to_pil(foreground_image)
        # Work on a copy of the foreground image
        work_fg_pil = fg_pil_initial.copy()
        current_fg_w, current_fg_h = work_fg_pil.size

        # Determine canvas
        final_canvas_pil = None
        if canvas_image is not None:
            base_canvas_pil = self._tensor_to_pil(canvas_image)
            if base_canvas_pil.mode != 'RGBA':
                final_canvas_pil = base_canvas_pil.convert('RGBA')
            else:
                final_canvas_pil = base_canvas_pil.copy()
            c_w, c_h = final_canvas_pil.size
        else:
            c_w, c_h = canvas_width, canvas_height
            try:
                parsed_color_rgba = ImageColor.getcolor(canvas_color, 'RGBA')
            except ValueError:
                print(f"Warning: Invalid canvas_color '{canvas_color}'. Using default #FFFFFFFF (white opaque).")
                parsed_color_rgba = ImageColor.getcolor("#FFFFFFFF", 'RGBA')
            final_canvas_pil = Image.new('RGBA', (c_w, c_h), parsed_color_rgba)

        # --- SCALING LOGIC ---
        if scale_foreground_if_larger:
            # Target dimensions for the foreground to fit within, based on canvas size and ratio
            target_fit_w = int(c_w * fit_within_canvas_ratio)
            target_fit_h = int(c_h * fit_within_canvas_ratio)

            # Ensure target dimensions are at least 1x1
            target_fit_w = max(1, target_fit_w)
            target_fit_h = max(1, target_fit_h)

            # Only scale if foreground is larger than the target fitting area in either dimension
            if current_fg_w > target_fit_w or current_fg_h > target_fit_h:
                # Calculate scaling ratio to fit within target_fit_w and target_fit_h, preserving aspect ratio
                ratio_w = target_fit_w / current_fg_w
                ratio_h = target_fit_h / current_fg_h
                scale_ratio = min(ratio_w, ratio_h)

                new_fg_w = int(current_fg_w * scale_ratio)
                new_fg_h = int(current_fg_h * scale_ratio)

                # Ensure new dimensions are at least 1x1
                new_fg_w = max(1, new_fg_w)
                new_fg_h = max(1, new_fg_h)

                if (new_fg_w != current_fg_w) or (new_fg_h != current_fg_h) : # Only resize if dimensions actually change
                    print(f"Scaling foreground from {current_fg_w}x{current_fg_h} to {new_fg_w}x{new_fg_h} to fit {fit_within_canvas_ratio*100}% of canvas.")
                    resampling_method = self.get_resampling_method()
                    work_fg_pil = work_fg_pil.resize((new_fg_w, new_fg_h), resample=resampling_method)
                    current_fg_w, current_fg_h = work_fg_pil.size # Update dimensions after scaling
        # --- END SCALING LOGIC ---

        # Ensure the foreground image (work_fg_pil) is RGBA for pasting
        if work_fg_pil.mode != 'RGBA':
            paste_fg_pil = work_fg_pil.convert('RGBA')
        else:
            paste_fg_pil = work_fg_pil # It's already RGBA (or became RGBA after convert)

        # Use dimensions of the (potentially scaled) RGBA foreground for positioning
        fg_w_for_paste, fg_h_for_paste = paste_fg_pil.size

        # Calculate paste position
        paste_x, paste_y = 0, 0
        if position == "center":
            paste_x = (c_w - fg_w_for_paste) // 2 + padding_x
            paste_y = (c_h - fg_h_for_paste) // 2 + padding_y
        elif position == "left":
            paste_x = padding_x
            paste_y = (c_h - fg_h_for_paste) // 2 + padding_y
        elif position == "right":
            paste_x = c_w - fg_w_for_paste - padding_x
            paste_y = (c_h - fg_h_for_paste) // 2 + padding_y
        elif position == "top":
            paste_x = (c_w - fg_w_for_paste) // 2 + padding_x
            paste_y = padding_y
        elif position == "bottom":
            paste_x = (c_w - fg_w_for_paste) // 2 + padding_x
            paste_y = c_h - fg_h_for_paste - padding_y
        elif position == "top_left":
            paste_x = padding_x
            paste_y = padding_y
        elif position == "top_right":
            paste_x = c_w - fg_w_for_paste - padding_x
            paste_y = padding_y
        elif position == "bottom_left":
            paste_x = padding_x
            paste_y = c_h - fg_h_for_paste - padding_y
        elif position == "bottom_right":
            paste_x = c_w - fg_w_for_paste - padding_x
            paste_y = c_h - fg_h_for_paste - padding_y

        # Create a new image for the output
        output_pil = final_canvas_pil.copy()
        # Paste the (potentially scaled and RGBA converted) foreground onto the output canvas
        # The .paste method uses the alpha channel of paste_fg_pil as a mask automatically.
        output_pil.paste(paste_fg_pil, (paste_x, paste_y), paste_fg_pil)

        return (self._pil_to_tensor(output_pil),)

NODE_CLASS_MAPPINGS = {
    "CanvasFusionNode": CanvasFusionNode # Changed class name slightly to avoid conflicts if old one is loaded
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CanvasFusionNode": "Canvas Fusion"
}

print("加载 CanvasFusionNode 节点 - 带缩放功能")