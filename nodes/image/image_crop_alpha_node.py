# File: ComfyUI/custom_nodes/image_crop_alpha_node.py
# (or ComfyUI/custom_nodes/some_subfolder/image_crop_alpha_node.py)

from PIL import Image, ImageOps, ImageDraw
import numpy as np
import torch
import os  # Only needed for the original script's output naming, not critical for node return

# Try to import skimage for Otsu thresholding
try:
    from skimage.filters import threshold_otsu

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not installed. Automatic alpha thresholding will not be available.")
    print("Please install it with: pip install scikit-image")


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
        if img_np.ndim == 2:  # Grayscale mask
            img_np = np.expand_dims(img_np, axis=2)  # HWC (H, W, 1)
        tensor = torch.from_numpy(img_np)
        # ComfyUI expects images as (H, W, C) and masks as (H, W) or (H,W,1) typically.
        # If it's a mask (H,W,1), it will be squeezed later in the wrapper if needed.
        tensors.append(tensor)

    if not tensors:  # Should not happen if pil_images is not empty
        return torch.empty(0)

    try:
        return torch.stack(tensors)
    except RuntimeError as e:
        # This can happen if images in the batch have different dimensions after cropping.
        # ComfyUI expects batch tensors to have consistent dimensions.
        print(f"Error stacking tensors: {e}. This might be due to images in a batch being cropped to different sizes.")
        print("Returning the first processed image/mask only to avoid crashing ComfyUI.")
        if tensors:
            # For IMAGE, ensure it's 4D (B, H, W, C)
            if tensors[0].ndim == 3:
                return tensors[0].unsqueeze(0)
            return tensors[0]  # Should already be B,H,W,C if it was a batch of 1 originally
        raise e  # Re-raise if tensors list was empty or other unexpected issue


class ImageCropByAlpha:
    """
    A ComfyUI node to crop an image based on its alpha channel content.
    Removes surrounding transparent or near-transparent areas.
    The output bbox_mask is cropped to the same dimensions as the cropped_image.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "auto_threshold": ("BOOLEAN", {"default": True}),
                "alpha_threshold": ("INT", {
                    "default": 10, "min": 0, "max": 255, "step": 1, "display": "number"
                }),
                "padding": ("INT", {
                    "default": 0, "min": 0, "max": 1024, "step": 1, "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("cropped_image", "cropped_bbox_mask", "used_alpha_threshold")  # Renamed mask output
    FUNCTION = "crop_image_wrapper"
    CATEGORY = "image/transform"

    def _crop_pil_image(self, pil_img: Image.Image, alpha_threshold: int, padding: int):
        """
        Internal method to crop a single PIL Image.
        Returns the cropped PIL Image, the cropped thresholded alpha mask, and the used alpha threshold.
        The returned mask will have the same dimensions as the cropped image.
        """
        original_mode = pil_img.mode
        img_rgba = pil_img.convert("RGBA")  # Ensure image is RGBA for alpha processing
        width, height = img_rgba.size

        thresholded_alpha_mask_pil = None  # This will be the basis for the output mask

        if alpha_threshold > 0:
            alpha_channel = img_rgba.getchannel('A')
            # This mask is at original dimensions
            thresholded_alpha_mask_pil = alpha_channel.point(lambda p: 255 if p >= alpha_threshold else 0)
            bbox = thresholded_alpha_mask_pil.getbbox()
        else:
            # If alpha_threshold is 0, use the original alpha channel for bbox,
            # or the content for RGB images.
            if original_mode == "RGBA":
                bbox = img_rgba.getbbox()  # Uses original alpha
                if bbox:  # If content found, use original alpha as the mask basis
                    thresholded_alpha_mask_pil = img_rgba.getchannel('A')
                else:  # Fully transparent RGBA
                    thresholded_alpha_mask_pil = Image.new("L", img_rgba.size, 0)
            elif original_mode == "RGB":  # For RGB images, effectively a content crop
                # getbbox for RGB considers any non-black pixel.
                # We need a mask representing this "content".
                # A simple way is to convert to L and find bbox.
                # Or just use PIL's getbbox on RGB directly.
                temp_L_img = pil_img.convert("L")
                bbox = temp_L_img.getbbox()  # getbbox on luminance
                if bbox:
                    # Create a mask that's white where content was (within the bbox)
                    # and black elsewhere, at original dimensions.
                    thresholded_alpha_mask_pil = Image.new("L", img_rgba.size, 0)
                    draw = ImageDraw.Draw(thresholded_alpha_mask_pil)
                    draw.rectangle(bbox, fill=255)  # Fill the content area
                else:  # Completely black RGB
                    thresholded_alpha_mask_pil = Image.new("L", img_rgba.size, 0)

            else:  # For L or other modes, convert to RGBA and use its bbox
                # This case implies alpha_threshold == 0 for non-RGB/RGBA images.
                # Treat as content crop.
                bbox = img_rgba.getbbox()  # Uses alpha of converted RGBA
                if bbox:
                    thresholded_alpha_mask_pil = img_rgba.getchannel('A')  # Take the derived alpha
                else:
                    thresholded_alpha_mask_pil = Image.new("L", img_rgba.size, 0)

            # Ensure thresholded_alpha_mask_pil is not None
            if thresholded_alpha_mask_pil is None:
                thresholded_alpha_mask_pil = Image.new("L", img_rgba.size, 0)

        if bbox:
            # Calculate crop coordinates with padding
            left = max(0, bbox[0] - padding)
            upper = max(0, bbox[1] - padding)
            right = min(width, bbox[2] + padding)
            lower = min(height, bbox[3] + padding)

            if left < right and upper < lower:
                # Crop the image (convert back to original mode if it wasn't RGBA implicitly)
                cropped_pil_img = img_rgba.crop((left, upper, right, lower))
                if original_mode == "RGB" and cropped_pil_img.mode == "RGBA":
                    cropped_pil_img = cropped_pil_img.convert("RGB")
                elif original_mode == "L" and cropped_pil_img.mode != "L":
                    cropped_pil_img = cropped_pil_img.convert("L")
                # Add other mode conversions if necessary

                # --- MODIFICATION START ---
                # Crop the thresholded_alpha_mask_pil using the *same* final coordinates
                cropped_bbox_mask_pil = thresholded_alpha_mask_pil.crop((left, upper, right, lower))
                # --- MODIFICATION END ---

                print(
                    f"Original size: {pil_img.size}, New size: {cropped_pil_img.size}, Mode: {original_mode} -> {cropped_pil_img.mode}")
                print(f"Using alpha_threshold: {alpha_threshold}")
                print(f"Computed bbox (no padding): {bbox}")
                print(f"Final crop box (L,U,R,L): {(left, upper, right, lower)}")
                return cropped_pil_img, cropped_bbox_mask_pil
            else:
                print(
                    f"Warning: Invalid crop dimensions after padding. Left:{left} Right:{right} Upper:{upper} Lower:{lower}. Returning original image and an empty mask of original size.")
                # Return original image and an empty mask of original size
                return pil_img, Image.new("L", img_rgba.size, 0)
        else:
            print(
                f"Warning: No content found with alpha_threshold {alpha_threshold} (or by getbbox). Returning original image and an empty mask of original size.")
            # Return original image and empty mask of original size
            return pil_img, Image.new("L", img_rgba.size, 0)

    def crop_image_wrapper(self, image: torch.Tensor, alpha_threshold: int, padding: int, auto_threshold: bool):
        pil_images = tensor_to_pil(image)

        cropped_pil_images_list = []
        cropped_bbox_masks_pil_list = []
        used_thresholds_list = []

        for i, pil_img in enumerate(pil_images):
            current_alpha_threshold = alpha_threshold

            img_for_analysis = pil_img.convert("RGBA")  # Always analyze alpha from RGBA
            alpha_channel_np = np.array(img_for_analysis.getchannel('A'))

            actual_alpha_threshold_for_image = alpha_threshold  # Start with manual or input

            if auto_threshold:
                if not SKIMAGE_AVAILABLE:
                    print(
                        "Warning: 'Auto Threshold' is True, but scikit-image is not installed. Falling back to manual threshold. Please install scikit-image.")
                elif alpha_channel_np.min() == alpha_channel_np.max():
                    # Handles fully transparent, fully opaque, or flat alpha
                    actual_alpha_threshold_for_image = 0 if alpha_channel_np.min() == 0 else 1
                    print(
                        f"Image {i}: Alpha channel is flat (min=max={alpha_channel_np.min()}). Auto threshold set to {actual_alpha_threshold_for_image}.")
                else:
                    try:
                        otsu_val = threshold_otsu(alpha_channel_np)
                        actual_alpha_threshold_for_image = int(otsu_val)
                        print(
                            f"Image {i}: Auto-calculated Otsu threshold: {actual_alpha_threshold_for_image} (Manual was: {alpha_threshold})")
                    except Exception as e:
                        print(
                            f"Warning: Otsu threshold calculation failed for image {i}: {e}. Falling back to manual threshold '{alpha_threshold}'.")
                        actual_alpha_threshold_for_image = alpha_threshold  # Fallback

            cropped_img_pil, cropped_mask_pil = self._crop_pil_image(pil_img, actual_alpha_threshold_for_image, padding)

            cropped_pil_images_list.append(cropped_img_pil)
            cropped_bbox_masks_pil_list.append(cropped_mask_pil)
            used_thresholds_list.append(actual_alpha_threshold_for_image)

        # Important: If images in the batch are cropped to different sizes,
        # pil_to_tensor (specifically torch.stack) will fail.
        # This is a limitation of batch processing in ComfyUI if output dimensions vary.
        # The error handling in pil_to_tensor will try to return the first item if stacking fails.
        cropped_tensor = pil_to_tensor(cropped_pil_images_list)
        bbox_mask_tensor = pil_to_tensor(cropped_bbox_masks_pil_list)

        # MASK tensor for ComfyUI is usually (B, H, W) or (H,W) for single mask.
        # pil_to_tensor returns (B, H, W, 1) for grayscale. Squeeze the last dim.
        if bbox_mask_tensor.ndim == 4 and bbox_mask_tensor.shape[-1] == 1:
            bbox_mask_tensor = bbox_mask_tensor.squeeze(-1)
        elif bbox_mask_tensor.ndim == 3 and image.ndim == 3:  # Single image input, mask is (H,W,1)
            bbox_mask_tensor = bbox_mask_tensor.squeeze(-1)

        final_used_threshold = used_thresholds_list[0] if used_thresholds_list else alpha_threshold

        return (cropped_tensor, bbox_mask_tensor, final_used_threshold)


NODE_CLASS_MAPPINGS = {
    "ImageCropByAlpha": ImageCropByAlpha  # Renamed class to avoid conflicts during testing
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCropByAlpha": "Crop Image by Alpha (Match Mask Size)"  # Renamed display name
}