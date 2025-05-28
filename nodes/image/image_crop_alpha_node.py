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
        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, axis=2)
        tensor = torch.from_numpy(img_np)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(2)
        tensors.append(tensor)

    return torch.stack(tensors)


class ImageCropByAlpha:
    """
    A ComfyUI node to crop an image based on its alpha channel content.
    Removes surrounding transparent or near-transparent areas.
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
    RETURN_NAMES = ("cropped_image", "bbox_mask", "used_alpha_threshold")
    FUNCTION = "crop_image_wrapper"  # Renamed to avoid conflict if old FUNCTION was crop_image
    CATEGORY = "image/transform"

    def _crop_pil_image(self, pil_img: Image.Image, alpha_threshold: int, padding: int):
        """
        Internal method to crop a single PIL Image.
        Returns the cropped PIL Image, the thresholded alpha mask, and the used alpha threshold.
        """
        img_rgba = pil_img.convert("RGBA")  # Ensure image is RGBA
        width, height = img_rgba.size

        thresholded_alpha_mask_pil = None

        if alpha_threshold > 0:
            alpha_channel = img_rgba.getchannel('A')
            thresholded_alpha_mask_pil = alpha_channel.point(lambda p: 255 if p >= alpha_threshold else 0)
            bbox = thresholded_alpha_mask_pil.getbbox()
        else:
            # If alpha_threshold is 0, use the original alpha channel for bbox,
            # or the content for RGB images.
            if pil_img.mode == "RGBA":  # Use original image if RGBA
                # getbbox on RGBA image's original alpha channel (any non-zero alpha)
                bbox = img_rgba.getbbox()
            elif pil_img.mode == "RGB":  # For RGB images, effectively a content crop
                bbox = pil_img.getbbox()  # Use original image if not RGBA
            else:  # For L or other modes, convert to RGBA and use its bbox
                bbox = img_rgba.getbbox()

            # Create a dummy full mask if no explicit thresholding was done for consistent return structure
            # This mask represents "all content considered by getbbox()"
            if bbox:
                # Create a mask that's white within the found bbox
                thresholded_alpha_mask_pil = Image.new("L", img_rgba.size, 0)
                draw = ImageDraw.Draw(thresholded_alpha_mask_pil)
                draw.rectangle(bbox, fill=255)
            else:  # No bbox found (e.g. completely black RGB or fully transparent RGBA)
                thresholded_alpha_mask_pil = Image.new("L", img_rgba.size, 0)

        if bbox:
            left = max(0, bbox[0] - padding)
            upper = max(0, bbox[1] - padding)
            right = min(width, bbox[2] + padding)
            lower = min(height, bbox[3] + padding)

            if left < right and upper < lower:
                cropped_img = img_rgba.crop((left, upper, right, lower))

                # Create a final_bbox_mask on the original image dimensions
                final_bbox_mask = Image.new("L", img_rgba.size, 0)
                # Paste the relevant part of THE THRESHOLDED mask (or the bbox rect if threshold was 0)
                # into the final_bbox_mask at the bbox location.
                if thresholded_alpha_mask_pil:  # This should always be true due to the logic above
                    # If thresholded_alpha_mask_pil was based on point(), use its actual shape
                    # If it was a drawn rectangle for threshold=0, use that.
                    # We want the mask to represent what defined the bbox.
                    mask_content_to_paste = thresholded_alpha_mask_pil.crop(bbox)
                    final_bbox_mask.paste(mask_content_to_paste, bbox)

                print(f"Original size: {pil_img.size}, New size: {cropped_img.size}, Mode: {pil_img.mode}")
                print(f"Using alpha_threshold: {alpha_threshold}")
                print(f"Computed bbox (no padding): {bbox}")
                print(f"Final crop box (L,U,R,L): {(left, upper, right, lower)}")
                return cropped_img, final_bbox_mask
            else:
                print(
                    f"Warning: Invalid crop dimensions after padding. Left:{left} Right:{right} Upper:{upper} Lower:{lower}. Returning original.")
                # Return original image if crop is invalid, and an empty mask
                return pil_img, Image.new("L", img_rgba.size, 0)
        else:
            print(
                f"Warning: No content found with alpha_threshold {alpha_threshold} (or by getbbox). Returning original.")
            # Return original image and empty mask
            return pil_img, Image.new("L", img_rgba.size, 0)

    def crop_image_wrapper(self, image: torch.Tensor, alpha_threshold: int, padding: int, auto_threshold: bool):
        pil_images = tensor_to_pil(image)

        cropped_pil_images = []
        bbox_masks_pil = []
        used_thresholds = []  # To store the threshold used for each image

        for i, pil_img in enumerate(pil_images):
            current_alpha_threshold = alpha_threshold  # Start with manual

            # Convert to RGBA for consistent alpha channel analysis,
            # even if original is RGB (Otsu would then work on a fully opaque alpha)
            img_for_analysis = pil_img.convert("RGBA")
            alpha_channel_np = np.array(img_for_analysis.getchannel('A'))

            if auto_threshold:
                if not SKIMAGE_AVAILABLE:
                    print(
                        "Warning: 'Auto Threshold' is True, but scikit-image is not installed. Falling back to manual threshold. Please install scikit-image.")
                elif alpha_channel_np.min() == alpha_channel_np.max():
                    # Alpha channel is flat (e.g., all 0, all 255, or all 128)
                    # If it's fully transparent (all 0), Otsu might not be helpful or might error.
                    # If it's fully opaque (all 255), Otsu will likely give 254 or 255.
                    # For flat alpha, if > 0, we want to consider all of it content.
                    # If 0, effectively no alpha content.
                    current_alpha_threshold = 0 if alpha_channel_np.min() == 0 else 1
                    print(
                        f"Image {i}: Alpha channel is flat (min=max={alpha_channel_np.min()}). Auto threshold set to {current_alpha_threshold}.")
                else:
                    try:
                        otsu_val = threshold_otsu(alpha_channel_np)
                        current_alpha_threshold = int(otsu_val)
                        print(
                            f"Image {i}: Auto-calculated Otsu threshold: {current_alpha_threshold} (Manual was: {alpha_threshold})")
                    except Exception as e:
                        print(
                            f"Warning: Otsu threshold calculation failed for image {i}: {e}. Falling back to manual threshold '{alpha_threshold}'.")
                        current_alpha_threshold = alpha_threshold  # Fallback to manual

            cropped_img, bbox_mask = self._crop_pil_image(pil_img, current_alpha_threshold, padding)
            cropped_pil_images.append(cropped_img)
            bbox_masks_pil.append(bbox_mask)
            used_thresholds.append(current_alpha_threshold)

        cropped_tensor = pil_to_tensor(cropped_pil_images)

        # For bbox_mask_tensor, ensure all masks have the original image's dimensions before stacking
        # This is important if some images in batch weren't cropped (returned original size).
        # However, _crop_pil_image already returns masks of original dimensions.
        bbox_mask_tensor = pil_to_tensor(bbox_masks_pil)

        if bbox_mask_tensor.ndim == 4 and bbox_mask_tensor.shape[-1] == 1:
            bbox_mask_tensor = bbox_mask_tensor.squeeze(-1)

        # For a batch, returning a single int for "used_alpha_threshold" might be ambiguous.
        # ComfyUI typically expects single values for simple types. Let's return the first one.
        # A more advanced node might return a list or string of thresholds.
        final_used_threshold = used_thresholds[0] if used_thresholds else alpha_threshold

        return (cropped_tensor, bbox_mask_tensor, final_used_threshold)


NODE_CLASS_MAPPINGS = {
    "ImageCropByAlphaAdvanced": ImageCropByAlpha  # Renamed class in mapping
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCropByAlphaAdvanced": "Crop Image by Alpha (Adv)"  # Renamed display name
}

# Test execution (optional, good for direct script testing)
if __name__ == "__main__":
    cropper = ImageCropByAlpha()

    # Test 1: RGBA image with semi-transparent and opaque areas
    print("\n--- Test 1: RGBA image, Auto Threshold, Padding 10 ---")
    test_img_pil_rgba = Image.new("RGBA", (200, 150), (0, 0, 0, 0))
    draw_rgba = ImageDraw.Draw(test_img_pil_rgba)
    draw_rgba.rectangle((50, 30, 150, 120), fill=(255, 0, 0, 100))  # Semi-transparent red box
    draw_rgba.rectangle((60, 40, 140, 110), fill=(0, 255, 0, 255))  # Opaque green box
    test_tensor_rgba = pil_to_tensor([test_img_pil_rgba])

    cropped_image_tensor, bbox_mask_tensor, used_thresh = cropper.crop_image_wrapper(
        test_tensor_rgba, alpha_threshold=50, padding=10, auto_threshold=True
    )
    print(f"Test 1 used threshold: {used_thresh}")
    result_pil_batch = tensor_to_pil(cropped_image_tensor)
    mask_pil_batch = tensor_to_pil(bbox_mask_tensor.unsqueeze(-1))
    if result_pil_batch: result_pil_batch[0].save("test_auto_cropped_output.png")
    if mask_pil_batch: mask_pil_batch[0].convert("L").save("test_auto_bbox_mask_output.png")
    print(f"Cropped image_rgba tensor shape: {cropped_image_tensor.shape}")
    print(f"BBox mask_rgba tensor shape: {bbox_mask_tensor.shape}")

    # Test 2: RGB image (Otsu on added opaque alpha)
    print("\n--- Test 2: RGB image, Auto Threshold (threshold=0), Padding 0 ---")
    rgb_img_pil = Image.new("RGB", (120, 100), (255, 255, 255))  # White background
    draw_rgb = ImageDraw.Draw(rgb_img_pil)
    draw_rgb.rectangle((20, 20, 80, 80), fill=(0, 0, 0))  # Black square
    rgb_tensor = pil_to_tensor([rgb_img_pil])

    # For RGB without actual alpha, auto_threshold would likely result in 1 (if content found)
    # or 0 if alpha_threshold=0 path is taken.
    # The logic for `alpha_threshold == 0` should lead to content crop.
    cropped_rgb_tensor, _, used_thresh_rgb = cropper.crop_image_wrapper(
        rgb_tensor, alpha_threshold=0, padding=0, auto_threshold=True
    )
    print(f"Test 2 used threshold: {used_thresh_rgb}")  # Expect 1 if auto on opaque alpha
    result_rgb_pil = tensor_to_pil(cropped_rgb_tensor)
    if result_rgb_pil: result_rgb_pil[0].save("test_rgb_auto_cropped_output.png")

    # Test 3: Fully transparent image with Auto Threshold
    print("\n--- Test 3: Fully transparent image, Auto Threshold ---")
    transparent_img_pil = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    transparent_tensor = pil_to_tensor([transparent_img_pil])
    cropped_trans_tensor, _, used_thresh_trans = cropper.crop_image_wrapper(
        transparent_tensor, alpha_threshold=10, padding=5, auto_threshold=True
    )
    print(f"Test 3 used threshold: {used_thresh_trans}")  # Expect 0
    result_trans_pil = tensor_to_pil(cropped_trans_tensor)
    if result_trans_pil:
        print(
            f"Transparent image result size: {result_trans_pil[0].size} (should be original size or small error crop)")

    # Test 4: Image like the user provided (assuming it's RGB on white background)
    print("\n--- Test 4: Simulating user image (RGB on white border), Auto Threshold ---")
    # Create an image with a central object and significant "empty" border
    # This image will be RGB, so convert("RGBA") adds an opaque alpha channel.
    user_sim_pil = Image.new("RGB", (600, 800), (240, 240, 240))  # Light gray background
    draw_user_sim = ImageDraw.Draw(user_sim_pil)
    # Draw a "subject" in the middle
    draw_user_sim.ellipse((150, 200, 450, 600), fill=(50, 50, 150))  # Dark blueish ellipse
    user_sim_tensor = pil_to_tensor([user_sim_pil])

    cropped_user_sim_tensor, _, used_thresh_user = cropper.crop_image_wrapper(
        user_sim_tensor, alpha_threshold=1, padding=0, auto_threshold=True
    )
    # For an RGB image converted to RGBA (opaque alpha), Otsu on the alpha will result in
    # current_alpha_threshold = 1 (because alpha is flat and > 0).
    # Then _crop_pil_image with threshold=1 on an opaque alpha will use full image bbox.
    # BUT if the original image input to _crop_pil_image was RGB, and threshold=0 it uses pil_img.getbbox()
    # If threshold > 0 for an RGB image, it uses getbbox on the thresholded *added* alpha, which is full.
    # The critical part is that if `auto_threshold` is true, and the image is RGB,
    # `img_for_analysis` becomes RGBA (opaque). `alpha_channel_np.min() == alpha_channel_np.max()` is true (255).
    # `current_alpha_threshold` becomes 1.
    # Then `_crop_pil_image(pil_img, 1, padding)` is called.
    # Inside `_crop_pil_image`, `pil_img` is still the original RGB. `img_rgba = pil_img.convert("RGBA")`.
    # `alpha_channel = img_rgba.getchannel('A')` (all 255s).
    # `thresholded_alpha_mask_pil = alpha_channel.point(lambda p: 255 if p >= 1 else 0)` (all 255s).
    # `bbox = thresholded_alpha_mask_pil.getbbox()` -> `(0,0,width,height)`.
    # This is not what we want for an RGB image for content crop *if* we want to truly mimic the `alpha_threshold=0` path.
    # The `alpha_threshold=0` path has specific `pil_img.mode == "RGB"` handling where `pil_img.getbbox()` is used.
    # If auto-threshold on an RGB results in `1`, it will not take that specific RGB content crop path.
    #
    # The desired behavior for an RGB image with "auto_threshold" would be:
    # 1. It doesn't have alpha, so "alpha thresholding" is not directly applicable.
    # 2. A sensible "auto" behavior is to perform a content crop, equivalent to `alpha_threshold = 0`.
    # So, if `auto_threshold` is True_ and `pil_img.mode` is not RGBA (or LA), then `current_alpha_threshold` should be set to 0.
    # Let's adjust the auto_threshold block:

    print(f"Test 4 used threshold: {used_thresh_user} (expect 0 if logic adjusted for RGB auto-crop)")
    result_user_sim_pil = tensor_to_pil(cropped_user_sim_tensor)
    if result_user_sim_pil:
        result_user_sim_pil[0].save("test_user_sim_auto_cropped_output.png")
        print("Saved test_user_sim_auto_cropped_output.png")

    # --- REFINED LOGIC FOR crop_image_wrapper's auto_threshold section ---
    # (This is a conceptual refinement, the code above already has a version)
    # if auto_threshold:
    #     original_mode = pil_img.mode
    #     img_for_analysis = pil_img.convert("RGBA") # For consistent alpha channel access
    #     alpha_channel_np = np.array(img_for_analysis.getchannel('A'))

    #     if original_mode not in ["RGBA", "LA", "PA"]: # If no original alpha
    #         current_alpha_threshold = 0 # Default to content crop for non-alpha images
    #         print(f"Image {i}: Original mode {original_mode} (no alpha). Auto threshold set to 0 for content crop.")
    #     elif not SKIMAGE_AVAILABLE:
    #         # ... (as before)
    #     elif alpha_channel_np.min() == alpha_channel_np.max():
    #         # ... (as before, threshold 0 if all transparent, 1 if all opaque/same value)
    #     else:
    #         # ... (Otsu, as before)
    # My implementation of `auto_threshold` block is:
    # 1. Convert to RGBA for analysis.
    # 2. If `alpha_channel_np.min() == alpha_channel_np.max()`: (this covers RGB images which get opaque alpha)
    #    - `current_alpha_threshold = 0` if (synthetic) alpha is all 0s (original was fully transparent RGBA)
    #    - `current_alpha_threshold = 1` if (synthetic) alpha is all 255s (original was RGB or opaque RGBA)
    # This means for an RGB image, `current_alpha_threshold` becomes 1.
    # Fed into `_crop_pil_image(pil_img, 1, padding)`:
    # If `pil_img` is RGB, `img_rgba = pil_img.convert("RGBA")`. `alpha_channel` from this is all 255s.
    # `thresholded_alpha_mask_pil` becomes all 255s. `bbox` is `(0,0,width,height)`.
    # It effectively crops the whole image, not content crop.

    # To fix this for RGB images under auto-threshold:
    # If `auto_threshold` is true, AND the original image is RGB (or L),
    # the `effective_alpha_threshold` should be 0 to trigger the `pil_img.getbbox()` logic.

    # Corrected logic sketch for `crop_image_wrapper` loop:
    # current_alpha_threshold = alpha_threshold
    # if auto_threshold:
    #     if pil_img.mode in ("RGB", "L"): # For images without native alpha
    #         current_alpha_threshold = 0 # Force content crop behavior
    #         print(f"Image {i}: Mode {pil_img.mode}. Auto-threshold forces content crop (threshold 0).")
    #     elif SKIMAGE_AVAILABLE:
    #         # (Otsu logic for images with actual alpha: RGBA, LA)
    #         # ... ensure img_for_analysis is pil_img itself if it has alpha, or converted if not
    #         # ... handle flat alpha (min==max) potentially setting threshold to 1 or 0
    #         # ... run Otsu
    # This change make auto-threshold on RGB images more intuitive (content crop via threshold 0).
    # The provided code has a good attempt, the main test case is how it handles an RGB image with auto_threshold.
    # The `if alpha_channel_np.min() == alpha_channel_np.max():` clause with `current_alpha_threshold = 1` for opaque alpha
    # will make it crop the entire image, not content-crop the RGB.
    # To make auto-threshold on RGB perform content crop, `current_alpha_threshold` must become 0.
    # One way:
    #   if auto_threshold:
    #       if pil_img.mode in ("RGB", "L"):
    #           current_alpha_threshold = 0
    #       elif SKIMAGE_AVAILABLE: ...
    # This is the change I'll incorporate into the main code block.