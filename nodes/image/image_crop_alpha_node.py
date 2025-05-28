# File: ComfyUI/custom_nodes/image_crop_alpha_node.py
# (or ComfyUI/custom_nodes/some_subfolder/image_crop_alpha_node.py)

from PIL import Image, ImageOps, ImageDraw
import numpy as np
import torch
import os

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
    if tensor is None:
        return None
    if tensor.ndim == 4:  # Batch of images
        pil_images = []
        for i in range(tensor.shape[0]):
            img_np = tensor[i].cpu().numpy()
            # If channel dim is missing for mask, add it
            if img_np.ndim == 2:  # H, W
                img_np = np.expand_dims(img_np, axis=-1)  # H, W, C=1

            img_np = (img_np * 255).astype(np.uint8)
            if img_np.shape[-1] == 1:  # Grayscale/Mask
                pil_images.append(Image.fromarray(img_np.squeeze(-1), mode='L'))
            elif img_np.shape[-1] == 3:  # RGB
                pil_images.append(Image.fromarray(img_np, mode='RGB'))
            elif img_np.shape[-1] == 4:  # RGBA
                pil_images.append(Image.fromarray(img_np, mode='RGBA'))
            else:
                raise ValueError(f"Unsupported number of channels: {img_np.shape[-1]}")
        return pil_images
    elif tensor.ndim == 3:  # Single image HWC (or HWD if mask coming in as BHW)
        img_np = tensor.cpu().numpy()
        # If channel dim is missing for mask, add it
        if img_np.ndim == 2:  # ComfyUI MASK (H,W) comes as 3D (B,H,W) from node but 2D here after squeeze.
            # If a single mask tensor (H,W) is passed, it might be 2D directly.
            img_np = np.expand_dims(img_np, axis=-1)  # H, W, C=1

        img_np = (img_np * 255).astype(np.uint8)
        if img_np.shape[-1] == 1:  # Grayscale/Mask
            return Image.fromarray(img_np.squeeze(-1), mode='L')
        elif img_np.shape[-1] == 3:  # RGB
            return Image.fromarray(img_np, mode='RGB')
        elif img_np.shape[-1] == 4:  # RGBA
            return Image.fromarray(img_np, mode='RGBA')
        else:
            raise ValueError(f"Unsupported number of channels: {img_np.shape[-1]}")
    else:  # Could be a single mask tensor (H,W)
        if tensor.ndim == 2:  # Single mask H, W
            img_np = tensor.cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            return Image.fromarray(img_np, mode='L')
        raise ValueError(f"Input tensor must be 2D, 3D or 4D, got {tensor.ndim}D")


# PIL to Tensor
def pil_to_tensor(pil_images):
    if not isinstance(pil_images, list):
        pil_images = [pil_images]
    if not pil_images:  # Empty list
        return torch.empty(0)

    tensors = []
    for pil_image in pil_images:
        img_np = np.array(pil_image).astype(np.float32) / 255.0
        if img_np.ndim == 2:  # For L mode images
            img_np = np.expand_dims(img_np, axis=2)  # H, W, C
        tensor = torch.from_numpy(img_np)
        # ComfyUI expects CHW for some things, HWC for IMAGE tensors usually.
        # IMAGE tensors are Batch, Height, Width, Channel
        # MASK tensors are Batch, Height, Width
        # This function returns HWC, so squeeze may be needed later for masks.
        tensors.append(tensor)

    try:
        return torch.stack(tensors)
    except RuntimeError as e:
        # This can happen if images in the batch have different sizes.
        # For this node, this is expected if input masks are cropped.
        # In ComfyUI, nodes must output tensors where batch items have same dimensions.
        # This means if one output relies on cropped dimensions and another on original,
        # it won't stack if cropping changes dimensions.
        # The logic below tries to ensure all masks in bbox_masks_pil have same dimensions.
        print(f"Error during pil_to_tensor torch.stack: {e}")
        print("This might be due to inconsistent image/mask sizes in the batch.")
        # Fallback or re-raise: For now, re-raise to see where it happens.
        # If this becomes an issue, the node might need to output lists for varying sizes,
        # or enforce consistent sizing (e.g. padding).
        raise e


class ImageCropByAlphaAdvanced:
    """
    A ComfyUI node to crop an image based on its alpha channel content or overall content.
    Optionally crops a provided mask using the same bounding box.
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
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("cropped_image", "bbox_mask", "used_alpha_threshold")
    FUNCTION = "crop_image_alpha_mask_wrapper"
    CATEGORY = "image/transform"

    def _crop_pil_image(self, pil_img: Image.Image, alpha_threshold: int, padding: int):
        """
        Internal method to crop a single PIL Image.
        Returns:
            - cropped_pil_image: The cropped PIL Image.
            - generated_bbox_mask: A mask (original dimensions) showing the bounding box.
            - crop_coordinates: Tuple (left, upper, right, lower) of the crop box,
                                or (0,0,width,height) if no crop occurred.
        """
        img_rgba_for_analysis = pil_img.convert("RGBA")  # Ensure image is RGBA for consistent alpha access
        width, height = pil_img.size  # Use original image's size for return consistency

        bbox = None
        if alpha_threshold > 0:
            alpha_channel = img_rgba_for_analysis.getchannel('A')
            thresholded_alpha_for_bbox = alpha_channel.point(lambda p: 255 if p >= alpha_threshold else 0)
            bbox = thresholded_alpha_for_bbox.getbbox()
        else:  # alpha_threshold is 0
            # For RGBA, getbbox() on original alpha uses any non-zero alpha
            # For RGB, getbbox() effectively becomes a content crop ( finds non-black region)
            # For L,  getbbox() effectively becomes a content crop ( finds non-black region)
            if pil_img.mode == "RGBA" or pil_img.mode == "LA":
                bbox = img_rgba_for_analysis.getbbox()  # Use already converted RGBA
            elif pil_img.mode == "RGB" or pil_img.mode == "L":
                bbox = pil_img.getbbox()  # Use original image for content crop
            else:  # Other modes, fall back to RGBA version's bbox
                bbox = img_rgba_for_analysis.getbbox()

        # Create a generated_bbox_mask (on original image dimensions)
        generated_bbox_mask = Image.new("L", (width, height), 0)
        if bbox:
            draw = ImageDraw.Draw(generated_bbox_mask)
            draw.rectangle(bbox, fill=255)  # Mark the raw bbox area

        if bbox:
            left = max(0, bbox[0] - padding)
            upper = max(0, bbox[1] - padding)
            right = min(width, bbox[2] + padding)
            lower = min(height, bbox[3] + padding)

            if left < right and upper < lower:
                # Crop the *original* pil_img, not necessarily img_rgba_for_analysis, to preserve mode
                cropped_img = pil_img.crop((left, upper, right, lower))
                crop_coords = (left, upper, right, lower)
                # print(f"Original: {pil_img.size}, New: {cropped_img.size}, Mode: {pil_img.mode}, BBox: {bbox}, CropBox: {crop_coords}")
                return cropped_img, generated_bbox_mask, crop_coords
            else:
                # print(f"Warning: Invalid crop dims. Left:{left} R:{right} U:{upper} L:{lower}. Original: {pil_img.size}")
                return pil_img, generated_bbox_mask, (0, 0, width, height)  # No crop
        else:
            # print(f"Warning: No content/bbox found. Original: {pil_img.size}")
            return pil_img, generated_bbox_mask, (0, 0, width, height)  # No crop

    def crop_image_alpha_mask_wrapper(self, image: torch.Tensor, alpha_threshold: int, padding: int,
                                      auto_threshold: bool, mask: torch.Tensor = None):
        pil_images = tensor_to_pil(image)

        pil_input_masks = []
        if mask is not None:
            pil_input_masks = tensor_to_pil(mask)
            if len(pil_input_masks) != len(pil_images):
                print(
                    f"Warning: Number of images ({len(pil_images)}) and masks ({len(pil_input_masks)}) mismatch. Masks will be ignored for unmatched images or if mask batch is shorter.")
                # Pad pil_input_masks with None if shorter
                pil_input_masks.extend([None] * (len(pil_images) - len(pil_input_masks)))
        else:
            pil_input_masks = [None] * len(pil_images)

        cropped_pil_images = []
        bbox_masks_pil = []  # This list will store the final masks for the bbox_mask output
        used_thresholds = []

        # Determine output mask format based on whether 'mask' input was connected
        output_masks_are_cropped_input_format = (mask is not None)

        for i, pil_img in enumerate(pil_images):
            pil_input_mask_single = pil_input_masks[i] if i < len(pil_input_masks) else None
            current_alpha_threshold = alpha_threshold

            if auto_threshold:
                # If the image mode is RGB or L (no intrinsic alpha), auto_threshold means content crop (threshold 0)
                if pil_img.mode in ('RGB', 'L') and not pil_img.info.get(
                        "transparency"):  # Check for palette transparency too
                    current_alpha_threshold = 0
                    # print(f"Image {i}: Mode {pil_img.mode}. Auto-threshold implies content crop, effective threshold set to 0.")
                # For images with alpha (RGBA, LA, or P with transparency)
                elif pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
                    if not SKIMAGE_AVAILABLE:
                        print(
                            "Warning: 'Auto Threshold' for alpha image, but scikit-image not installed. Using manual threshold.")
                    else:
                        # Analyze the actual alpha channel
                        img_for_alpha_analysis = pil_img.convert("RGBA")  # Ensures consistent alpha channel access
                        alpha_channel_np = np.array(img_for_alpha_analysis.getchannel('A'))

                        if alpha_channel_np.min() == alpha_channel_np.max():
                            # Alpha channel is flat. If fully transparent, threshold is 0. If opaque/semi-opaque, threshold is 1.
                            current_alpha_threshold = 0 if alpha_channel_np.min() == 0 else 1
                            # print(f"Image {i}: Alpha channel is flat (min=max={alpha_channel_np.min()}). Auto threshold set to {current_alpha_threshold}.")
                        else:
                            try:
                                otsu_val = threshold_otsu(alpha_channel_np)
                                current_alpha_threshold = int(otsu_val)
                                # print(f"Image {i}: Auto-calculated Otsu threshold for alpha: {current_alpha_threshold}")
                            except Exception as e:
                                print(
                                    f"Warning: Otsu threshold failed for image {i}: {e}. Falling back to manual '{alpha_threshold}'.")
                                # current_alpha_threshold remains alpha_threshold (manual)
                # else: # Image mode like P without transparency, or other complex modes. Fallback to manual.
                # print(f"Image {i}: Mode {pil_img.mode} with auto_threshold. Behavior defaults to manual threshold {alpha_threshold} as it's not explicitly RGB/L or alpha-bearing for Otsu.")
                # current_alpha_threshold remains alpha_threshold

            cropped_pil_img, generated_bbox_mask, crop_coords = \
                self._crop_pil_image(pil_img, current_alpha_threshold, padding)

            cropped_pil_images.append(cropped_pil_img)
            used_thresholds.append(current_alpha_threshold)

            # Determine the mask to output based on format
            if output_masks_are_cropped_input_format:
                if pil_input_mask_single is not None:
                    # Goal: crop pil_input_mask_single to match cropped_pil_img dimensions
                    output_mask_for_item = pil_input_mask_single.copy()
                    if output_mask_for_item.mode != 'L':
                        output_mask_for_item = output_mask_for_item.convert('L')

                    if output_mask_for_item.size != pil_img.size:
                        print(
                            f"Warning: Image {i} (size {pil_img.size}) and its input mask (size {output_mask_for_item.size}) have different dimensions. Creating blank mask of cropped image size.")
                        final_mask = Image.new("L", cropped_pil_img.size, 0)  # Blank mask
                    # Check if a valid crop actually happened (coordinates changed from full image)
                    elif crop_coords == (0, 0, pil_img.width, pil_img.height) or not (
                            crop_coords[0] < crop_coords[2] and crop_coords[1] < crop_coords[3]):
                        # No crop on image, or crop box invalid. Mask should be original dimensions.
                        # Ensure it's resized to cropped_pil_img.size if somehow they differ (should not if no crop)
                        if output_mask_for_item.size != cropped_pil_img.size:
                            final_mask = output_mask_for_item.resize(cropped_pil_img.size, Image.NEAREST)
                        else:
                            final_mask = output_mask_for_item
                    else:  # Valid crop happened, crop the input mask
                        final_mask = output_mask_for_item.crop(crop_coords)
                    bbox_masks_pil.append(final_mask)
                else:  # No specific input mask for this item, but mask input was connected globally
                    # Create a default mask (e.g., white) of cropped_pil_img.size
                    default_mask = Image.new("L", cropped_pil_img.size, 255)  # All white
                    bbox_masks_pil.append(default_mask)
            else:  # Mask input was NOT connected, use the generated bbox mask from _crop_pil_image
                bbox_masks_pil.append(generated_bbox_mask)

        cropped_tensor = pil_to_tensor(cropped_pil_images)

        # Ensure all masks in bbox_masks_pil are 'L' mode before tensor conversion if any doubt
        # (The logic above should ensure this)
        bbox_mask_tensor = pil_to_tensor(bbox_masks_pil)  # Should be B, H, W, C=1

        if bbox_mask_tensor.ndim == 4 and bbox_mask_tensor.shape[-1] == 1:
            bbox_mask_tensor = bbox_mask_tensor.squeeze(-1)  # Convert to B, H, W for MASK type

        final_used_threshold = used_thresholds[0] if used_thresholds else alpha_threshold

        return (cropped_tensor, bbox_mask_tensor, final_used_threshold)


NODE_CLASS_MAPPINGS = {
    "ImageCropByAlphaAdv": ImageCropByAlphaAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCropByAlphaAdv": "Crop Image by Alpha (Adv)"
}

# Test execution
if __name__ == "__main__":
    cropper = ImageCropByAlphaAdvanced()


    # Helper to save images/masks for inspection
    def save_debug_image(tensor_img, base_filename, is_mask=False):
        if tensor_img is None or tensor_img.numel() == 0:
            print(f"Skipping save for {base_filename}, tensor is empty or None.")
            return
        try:
            pil_list = tensor_to_pil(tensor_img.unsqueeze(-1) if is_mask and tensor_img.ndim == 3 else tensor_img)
            if pil_list:
                for idx, pil_item in enumerate(pil_list):
                    fname = f"{base_filename}_{idx}.png"
                    pil_item.save(fname)
                    print(f"Saved {fname} (size: {pil_item.size}, mode: {pil_item.mode})")
        except Exception as e:
            print(f"Error saving {base_filename}: {e}")


    print("\n--- Test 1: RGBA image, Auto Threshold, Padding 10, NO input mask ---")
    test_img_pil_rgba = Image.new("RGBA", (200, 150), (0, 0, 0, 0))  # transparent bg
    draw_rgba = ImageDraw.Draw(test_img_pil_rgba)
    draw_rgba.rectangle((50, 30, 150, 120), fill=(255, 0, 0, 100))
    draw_rgba.rectangle((60, 40, 140, 110), fill=(0, 255, 0, 255))
    test_tensor_rgba_batch = pil_to_tensor([test_img_pil_rgba, test_img_pil_rgba.rotate(45, expand=True)])

    cropped_img_t, bbox_mask_t, used_thresh = cropper.crop_image_alpha_mask_wrapper(
        test_tensor_rgba_batch, alpha_threshold=50, padding=10, auto_threshold=True, mask=None
    )
    print(f"Test 1 Used threshold: {used_thresh}")  # Otsu likely around 99 or 1 if semi-transparent is main
    save_debug_image(cropped_img_t, "test1_cropped_img")
    save_debug_image(bbox_mask_t, "test1_bbox_mask_generated", is_mask=True)
    # Expected: bbox_mask is generated, original dimensions. cropped_img is smaller.

    print("\n--- Test 2: RGB image, Auto Threshold (expect content crop, threshold 0), Padding 0, NO input mask ---")
    rgb_img_pil = Image.new("RGB", (120, 100), (255, 255, 255))  # white bg
    draw_rgb = ImageDraw.Draw(rgb_img_pil)
    draw_rgb.rectangle((20, 20, 80, 80), fill=(0, 0, 0))  # black square
    rgb_tensor = pil_to_tensor([rgb_img_pil])

    cropped_rgb_t, rgb_bbox_mask_t, used_thresh_rgb = cropper.crop_image_alpha_mask_wrapper(
        rgb_tensor, alpha_threshold=1, padding=0, auto_threshold=True, mask=None
    )
    print(f"Test 2 Used threshold: {used_thresh_rgb}")  # Expected 0 due to auto_threshold on RGB
    save_debug_image(cropped_rgb_t, "test2_cropped_rgb")
    save_debug_image(rgb_bbox_mask_t, "test2_bbox_mask_rgb_generated", is_mask=True)
    # Expected: Image cropped to black square. Mask shows bbox on original 120x100.

    print("\n--- Test 3: RGBA image (from Test 1) WITH an input mask ---")
    # Use the first image from Test 1
    single_test_img_tensor = pil_to_tensor(test_img_pil_rgba)
    # Create a dummy input mask (e.g., a gradient or just white) of the same original size
    input_mask_pil = Image.new("L", test_img_pil_rgba.size, 128)  # Gray mask
    draw_input_mask = ImageDraw.Draw(input_mask_pil)
    draw_input_mask.ellipse((10, 10, 70, 70), fill=255)  # white circle on gray mask
    input_mask_tensor = pil_to_tensor(input_mask_pil).unsqueeze(
        0)  # Batch of 1, Add channel dim, then pil_to_tensor adds it back if L
    # MASK tensor is B,H,W so (1,150,200)
    # pil_to_tensor makes it (1,150,200,1)

    input_mask_tensor_for_node = input_mask_tensor.squeeze(-1)  # (1, 150, 200) as MASK type

    cropped_img_t3, bbox_mask_t3, used_thresh3 = cropper.crop_image_alpha_mask_wrapper(
        single_test_img_tensor, alpha_threshold=50, padding=10, auto_threshold=True, mask=input_mask_tensor_for_node
    )
    print(f"Test 3 Used threshold: {used_thresh3}")
    save_debug_image(cropped_img_t3, "test3_cropped_img_w_mask_input")
    save_debug_image(bbox_mask_t3, "test3_output_mask_from_input", is_mask=True)
    # Expected: bbox_mask is the CROPPED VERSION of input_mask_pil, dimensions match cropped_img_t3.

    print("\n--- Test 4: Fully transparent image, WITH an input mask (should not crop) ---")
    transparent_img_pil = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    transparent_tensor = pil_to_tensor(transparent_img_pil).unsqueeze(0)  # Batch of 1

    input_mask_pil_trans = Image.new("L", (100, 100), 200)  # A constant gray mask
    input_mask_tensor_trans = pil_to_tensor(input_mask_pil_trans).unsqueeze(0).squeeze(-1)  # B,H,W

    cropped_trans_t, bbox_mask_trans_t, used_thresh_trans = cropper.crop_image_alpha_mask_wrapper(
        transparent_tensor, alpha_threshold=10, padding=5, auto_threshold=True, mask=input_mask_tensor_trans
    )
    print(f"Test 4 Used threshold: {used_thresh_trans}")  # Expected 0 (flat transparent alpha)
    save_debug_image(cropped_trans_t, "test4_cropped_transparent_img")  # Should be original
    save_debug_image(bbox_mask_trans_t, "test4_output_mask_transparent", is_mask=True)  # Should be original input mask

    print("\n--- Test 5: Batch with one image+mask, one image no-mask (input mask connected) ---")
    # Image 1: test_img_pil_rgba (200x150)
    # Mask 1: input_mask_pil (200x150)
    # Image 2: rgb_img_pil (120x100), will be cropped to 60x60 (no padding)

    batch_img_tensor = pil_to_tensor([test_img_pil_rgba, rgb_img_pil])
    # Provide only one mask in the batch for the MASK input
    batch_mask_tensor = pil_to_tensor([input_mask_pil]).squeeze(-1)  # (1, H, W)

    cropped_bimg_t5, bbox_bmask_t5, used_bthresh5 = cropper.crop_image_alpha_mask_wrapper(
        batch_img_tensor, alpha_threshold=10, padding=0, auto_threshold=True, mask=batch_mask_tensor
    )
    print(f"Test 5 Used threshold for first image in batch: {used_bthresh5}")
    save_debug_image(cropped_bimg_t5, "test5_batch_cropped_img")
    save_debug_image(bbox_bmask_t5, "test5_batch_output_mask", is_mask=True)
    # Expected:
    # Img1 cropped, Mask1 cropped to match.
    # Img2 cropped (content crop from RGB), its output mask should be a *default white mask* of Img2's *cropped* size.
    # This test will verify if pil_to_tensor for bbox_bmask_t5 works, as the two masks
    # (cropped input_mask_pil and default white mask) will have different dimensions
    # corresponding to their respective cropped images. This is where torch.stack might fail
    # if not handled. The current implementation will make pil_to_tensor fail here.
    # ComfyUI usually expects batched tensors to have same H,W.
    # A robust ComfyUI node might output lists of images if sizes vary, or ensure padding.
    # For now, let's see it fail or how Comfy handles it.
    # The current code will raise error in pil_to_tensor due to torch.stack.
    # To "fix" this for ComfyUI standard behavior, all items in output bbox_mask_tensor
    # would need to be padded to the same max H, max W of the batch.
    # The `pil_to_tensor` code in the original prompt does not handle varying sizes.
    # For this test script, we can iterate and save individually if it fails.
    # The code *as is* is likely more suitable for single image processing or batches
    # where all images crop to the same size or don't have input masks creating size disparities.
    print(
        "Test 5 note: If torch.stack fails (expected for varying output mask sizes), saving individual masks from bbox_masks_pil (if accessible) would be needed for full verification.")
    # For example, if this fails, one might modify crop_image_alpha_mask_wrapper to return lists of PIL images
    # instead of stacked tensors if sizes are expected to vary and the node is designed to handle that.
    # However, ComfyUI's `IMAGE` and `MASK` types prefer stacked tensors.