# File: ComfyUI/custom_nodes/image_crop_alpha_node.py
# (or ComfyUI/custom_nodes/some_subfolder/image_crop_alpha_node.py)

from PIL import Image, ImageOps, ImageDraw
import numpy as np
import torch
import os  # Only needed for the original script's output naming, not critical for node return


# Tensor to PIL
def tensor_to_pil(tensor):
    if tensor.ndim == 4:  # Batch of images
        pil_images = []
        for i in range(tensor.shape[0]):
            img_np = tensor[i].cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            if img_np.shape[-1] == 1:  # Grayscale
                pil_images.append(Image.fromarray(img_np.squeeze(-1), mode='L'))
            elif img_np.shape[-1] == 3:  # RGB
                pil_images.append(Image.fromarray(img_np, mode='RGB'))
            elif img_np.shape[-1] == 4:  # RGBA
                pil_images.append(Image.fromarray(img_np, mode='RGBA'))
            else:
                raise ValueError(f"Unsupported number of channels: {img_np.shape[-1]}")
        return pil_images
    elif tensor.ndim == 3:  # Single image HWC
        img_np = tensor.cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        if img_np.shape[-1] == 1:  # Grayscale
            return Image.fromarray(img_np.squeeze(-1), mode='L')
        elif img_np.shape[-1] == 3:  # RGB
            return Image.fromarray(img_np, mode='RGB')
        elif img_np.shape[-1] == 4:  # RGBA
            return Image.fromarray(img_np, mode='RGBA')
        else:
            raise ValueError(f"Unsupported number of channels: {img_np.shape[-1]}")
    else:
        raise ValueError("Input tensor must be 3D or 4D")


# PIL to Tensor
def pil_to_tensor(pil_images):
    if not isinstance(pil_images, list):
        pil_images = [pil_images]

    tensors = []
    for pil_image in pil_images:
        img_np = np.array(pil_image).astype(np.float32) / 255.0
        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, axis=2)  # H, W, C (C=1)

        tensor = torch.from_numpy(img_np)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(2)
        tensors.append(tensor)

    return torch.stack(tensors)


class ImageCropByAlphaAdvanced:
    """
    A ComfyUI node to crop an image based on its alpha channel content.
    Removes surrounding transparent or near-transparent areas.
    Supports global or individual L/R/T/B padding.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "alpha_threshold": ("INT", {
                    "default": 10, "min": 0, "max": 255, "step": 1, "display": "number"
                }),
                "padding": ("INT", {  # Global padding
                    "default": 0, "min": -1024, "max": 2048, "step": 1, "display": "number"  # Allow negative padding
                }),
                "individual_padding": ("BOOLEAN", {"default": False, "label_on": "Use Individual Padding",
                                                   "label_off": "Use Global Padding"}),
                "left_padding": ("INT", {
                    "default": 0, "min": -1024, "max": 2048, "step": 1, "display": "number"
                }),
                "right_padding": ("INT", {
                    "default": 0, "min": -1024, "max": 2048, "step": 1, "display": "number"
                }),
                "top_padding": ("INT", {
                    "default": 0, "min": -1024, "max": 2048, "step": 1, "display": "number"
                }),
                "bottom_padding": ("INT", {
                    "default": 0, "min": -1024, "max": 2048, "step": 1, "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("cropped_image", "bbox_mask")
    FUNCTION = "crop_image"
    CATEGORY = "image/transform"

    def _crop_pil_image(self, pil_img, alpha_threshold,
                        global_padding, use_individual_padding,
                        p_left, p_right, p_top, p_bottom):
        img_rgba = pil_img.convert("RGBA")
        width, height = img_rgba.size

        thresholded_alpha_mask_pil = None

        if alpha_threshold > 0:
            alpha_channel = img_rgba.split()[-1]
            thresholded_alpha_mask_pil = alpha_channel.point(lambda p: 255 if p >= alpha_threshold else 0)
            bbox = thresholded_alpha_mask_pil.getbbox()
        else:
            if img_rgba.mode == "RGBA":
                bbox = img_rgba.getbbox()
            else:
                bbox = pil_img.getbbox()
            thresholded_alpha_mask_pil = Image.new("L", img_rgba.size, 255)  # Full mask if no threshold

        padding_info = ""
        if bbox:
            if use_individual_padding:
                pad_l, pad_r, pad_t, pad_b = p_left, p_right, p_top, p_bottom
                padding_info = f"Individual Padding (L:{pad_l}, R:{pad_r}, T:{pad_t}, B:{pad_b})"
            else:
                pad_l = pad_r = pad_t = pad_b = global_padding
                padding_info = f"Global Padding: {global_padding}"

            # Apply padding: subtract from left/top, add to right/bottom of original bbox
            left = bbox[0] - pad_l
            upper = bbox[1] - pad_t
            right = bbox[2] + pad_r
            lower = bbox[3] + pad_b

            # Ensure padding does not make coordinates negative or exceed image bounds
            # Max with 0 for left/upper, min with width/height for right/lower
            # This behavior means negative padding can shrink INTO the content area
            # and positive padding expands OUT from the content area.

            # The crop box itself needs to be clamped to image dimensions
            crop_left = max(0, left)
            crop_upper = max(0, upper)
            crop_right = min(width, right)
            crop_lower = min(height, lower)

            if crop_left < crop_right and crop_upper < crop_lower:
                cropped_img = img_rgba.crop((crop_left, crop_upper, crop_right, crop_lower))

                final_bbox_mask = Image.new("L", img_rgba.size, 0)
                # Draw the effective bounding box (bbox + padding, before clamping to image edges)
                # This represents the "intended" crop area even if parts are outside original
                # For the MASK, we'll draw the *original computed bbox* based on alpha.
                # The crop itself uses the padded box.
                if thresholded_alpha_mask_pil:
                    final_bbox_mask.paste(thresholded_alpha_mask_pil.crop(bbox), bbox)
                elif bbox:  # Fallback for alpha_threshold=0 case
                    draw = ImageDraw.Draw(final_bbox_mask)
                    draw.rectangle(bbox, fill=255)

                print(f"Original size: {img_rgba.size}, New size: {cropped_img.size}")
                print(f"Computed bbox (no padding): {bbox}")
                print(padding_info)
                print(f"Padded box (before clamp): L:{left}, U:{upper}, R:{right}, B:{lower}")
                print(f"Final crop box (clamped): L:{crop_left}, U:{crop_upper}, R:{crop_right}, B:{crop_lower}")
                return cropped_img, final_bbox_mask
            else:
                print(
                    f"Warning: Invalid crop dimensions after padding. Original bbox: {bbox}, {padding_info}. Padded box (before clamp): L:{left} U:{upper} R:{right} B:{lower}. Clamped: L:{crop_left} U:{crop_upper} R:{crop_right} B:{crop_lower}. Returning original.")
                return pil_img, Image.new("L", img_rgba.size, 0)
        else:
            print(f"Warning: No content found with alpha_threshold {alpha_threshold}. Returning original.")
            return pil_img, Image.new("L", img_rgba.size, 0)

    def crop_image(self, image: torch.Tensor, alpha_threshold: int,
                   padding: int, individual_padding: bool,
                   left_padding: int, right_padding: int, top_padding: int, bottom_padding: int):

        pil_images = tensor_to_pil(image)
        cropped_pil_images = []
        bbox_masks_pil = []

        for pil_img in pil_images:
            cropped_img, bbox_mask = self._crop_pil_image(
                pil_img, alpha_threshold,
                padding, individual_padding,
                left_padding, right_padding, top_padding, bottom_padding
            )
            cropped_pil_images.append(cropped_img)
            bbox_masks_pil.append(bbox_mask)

        cropped_tensor = pil_to_tensor(cropped_pil_images)
        bbox_mask_tensor = pil_to_tensor(bbox_masks_pil)

        if bbox_mask_tensor.ndim == 4 and bbox_mask_tensor.shape[-1] == 1:
            bbox_mask_tensor = bbox_mask_tensor.squeeze(-1)

        return (cropped_tensor, bbox_mask_tensor)


NODE_CLASS_MAPPINGS = {
    "ImageCropByAlphaAdvanced": ImageCropByAlphaAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCropByAlphaAdvanced": "Crop Image by Alpha (Adv Padding)"
}

if __name__ == "__main__":
    # Create a dummy RGBA image with transparent borders for testing
    test_img_pil = Image.new("RGBA", (200, 150), (0, 0, 0, 0))
    draw = ImageDraw.Draw(test_img_pil)
    # Content from (50,30) to (150,120)
    draw.rectangle((50, 30, 150, 120), fill=(255, 0, 0, 100))
    draw.rectangle((60, 40, 140, 110), fill=(0, 255, 0, 255))

    test_tensor = pil_to_tensor([test_img_pil])

    cropper = ImageCropByAlphaAdvanced()

    print("\n--- Test 1: Global Padding 10, Threshold 50 ---")
    cropped_tensor_g, mask_tensor_g = cropper.crop_image(
        test_tensor, alpha_threshold=50,
        padding=10, individual_padding=False,
        left_padding=0, right_padding=0, top_padding=0, bottom_padding=0
    )
    res_g = tensor_to_pil(cropped_tensor_g)[0]
    res_g.save("test_crop_global_padding.png")
    print(f"Global padding: Saved test_crop_global_padding.png, size: {res_g.size}")
    # Expected bbox for threshold 50 should be around (60,40,140,110)
    # Global_pad=10: crop box (50,30,150,120), size (100,90)

    print("\n--- Test 2: Individual Padding, Threshold 50 ---")
    # L:5, R:15, T:2, B:8
    cropped_tensor_i, mask_tensor_i = cropper.crop_image(
        test_tensor, alpha_threshold=50,
        padding=100,  # This global padding should be ignored
        individual_padding=True,
        left_padding=5, right_padding=15, top_padding=2, bottom_padding=8
    )
    res_i = tensor_to_pil(cropped_tensor_i)[0]
    res_i.save("test_crop_individual_padding.png")
    print(f"Individual padding: Saved test_crop_individual_padding.png, size: {res_i.size}")
    # Expected bbox for threshold 50 for inner green (60,40,140,110)
    # Padded L: 60-5=55, T: 40-2=38
    # Padded R: 140+15=155, B: 110+8=118
    # Crop box: (55,38,155,118), size (100,80)

    print("\n--- Test 3: Negative Global Padding -5, Threshold 50 ---")
    cropped_tensor_n, _ = cropper.crop_image(
        test_tensor, alpha_threshold=50,
        padding=-5, individual_padding=False,
        left_padding=0, right_padding=0, top_padding=0, bottom_padding=0
    )
    res_n = tensor_to_pil(cropped_tensor_n)[0]
    res_n.save("test_crop_neg_global_padding.png")
    print(f"Negative global padding: Saved test_crop_neg_global_padding.png, size: {res_n.size}")
    # Expected bbox for threshold 50 for inner green (60,40,140,110)
    # Global_pad=-5: crop box L:60-(-5)=65, T:40-(-5)=45, R:140+(-5)=135, B:110+(-5)=105
    # Crop box: (65,45,135,105), size (70,60)

    print("\n--- Test 4: Individual Negative Padding, Threshold 50 ---")
    cropped_tensor_ni, _ = cropper.crop_image(
        test_tensor, alpha_threshold=50,
        padding=0, individual_padding=True,
        left_padding=-3, right_padding=-8, top_padding=-2, bottom_padding=-5
    )  # Shrinking from inside
    res_ni = tensor_to_pil(cropped_tensor_ni)[0]
    res_ni.save("test_crop_neg_indiv_padding.png")
    print(f"Negative individual padding: Saved test_crop_neg_indiv_padding.png, size: {res_ni.size}")
    # Expected bbox (60,40,140,110)
    # Padded L: 60-(-3)=63, T: 40-(-2)=42, R: 140+(-8)=132, B: 110+(-5)=105
    # Crop box: (63,42,132,105), size (69,63)