# File: ComfyUI/custom_nodes/image_crop_alpha_node.py
# (or ComfyUI/custom_nodes/some_subfolder/image_crop_alpha_node.py)

from PIL import Image, ImageOps
import numpy as np
import torch
import os  # Only needed for the original script's output naming, not critical for node return


# Tensor to PIL
def tensor_to_pil(tensor):
    if tensor.ndim == 4:  # Batch of images
        # Assuming CHW or HWC, we'll iterate and assume HWC for ComfyUI convention after permute
        # ComfyUI images are B, H, W, C (0-1 float)
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
        # Convert to NumPy array
        img_np = np.array(pil_image).astype(np.float32) / 255.0
        # Add channel dimension if grayscale
        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, axis=2)  # H, W, C (C=1)

        # Convert to PyTorch tensor
        tensor = torch.from_numpy(img_np)
        if tensor.ndim == 2:  # For safety, ensure 3D for single image (H, W, C)
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
                "alpha_threshold": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "number"
                }),
                "padding": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1024,  # Arbitrary practical max
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")  # Also return the mask used for bbox calculation
    RETURN_NAMES = ("cropped_image", "bbox_mask")
    FUNCTION = "crop_image"
    CATEGORY = "image/transform"  # You can choose your own category

    def _crop_pil_image(self, pil_img, alpha_threshold, padding):
        """
        Internal method to crop a single PIL Image.
        Returns the cropped PIL Image and the thresholded alpha mask.
        """
        # Ensure image is RGBA format
        img_rgba = pil_img.convert("RGBA")
        width, height = img_rgba.size

        thresholded_alpha_mask_pil = None  # Initialize

        if alpha_threshold > 0:
            alpha_channel = img_rgba.split()[-1]
            thresholded_alpha_mask_pil = alpha_channel.point(lambda p: 255 if p >= alpha_threshold else 0)
            bbox = thresholded_alpha_mask_pil.getbbox()
        else:
            # If alpha_threshold is 0, use the original alpha channel for bbox,
            # or even the full image if it has no alpha (getbbox on RGB will find content)
            if img_rgba.mode == "RGBA":
                bbox = img_rgba.getbbox()  # Considers any non-zero alpha
            else:  # For RGB images, effectively a content crop
                bbox = pil_img.getbbox()  # Use original image if not RGBA for some reason
            # Create a dummy full mask if no thresholding was done, for consistent return
            thresholded_alpha_mask_pil = Image.new("L", img_rgba.size, 255)

        if bbox:
            left = max(0, bbox[0] - padding)
            upper = max(0, bbox[1] - padding)
            right = min(width, bbox[2] + padding)
            lower = min(height, bbox[3] + padding)

            if left < right and upper < lower:
                cropped_img = img_rgba.crop((left, upper, right, lower))
                # Create a mask for bbox on the original image dimensions
                final_bbox_mask = Image.new("L", img_rgba.size, 0)
                if thresholded_alpha_mask_pil:
                    # Copy the relevant part of the thresholded mask to the new mask
                    # This is more accurate than drawing a rectangle if thresholded_alpha_mask_pil is complex
                    final_bbox_mask.paste(thresholded_alpha_mask_pil.crop(bbox), bbox)
                elif bbox:  # Fallback if thresholded_alpha_mask_pil wasn't relevant (e.g., alpha_threshold=0)
                    # Draw a white rectangle for the bbox
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(final_bbox_mask)
                    draw.rectangle(bbox, fill=255)

                print(f"Original size: {img_rgba.size}, New size: {cropped_img.size}")
                print(f"Computed bbox (no padding): {bbox}")
                print(f"Final crop box (L,U,R,L): {(left, upper, right, lower)}")
                return cropped_img, final_bbox_mask
            else:
                print(
                    f"Warning: Invalid crop dimensions after padding. Left:{left} Right:{right} Upper:{upper} Lower:{lower}. Returning original.")
                return pil_img, Image.new("L", img_rgba.size, 0)  # Return original image and empty mask
        else:
            print(f"Warning: No content found with alpha_threshold {alpha_threshold}. Returning original.")
            return pil_img, Image.new("L", img_rgba.size, 0)  # Return original image and empty mask

    def crop_image(self, image: torch.Tensor, alpha_threshold: int, padding: int):
        pil_images = tensor_to_pil(image)

        cropped_pil_images = []
        bbox_masks_pil = []

        for pil_img in pil_images:
            cropped_img, bbox_mask = self._crop_pil_image(pil_img, alpha_threshold, padding)
            cropped_pil_images.append(cropped_img)
            bbox_masks_pil.append(bbox_mask)

        cropped_tensor = pil_to_tensor(cropped_pil_images)
        bbox_mask_tensor = pil_to_tensor(bbox_masks_pil)  # This will be (B, H, W, 1)

        # Ensure mask is in the expected ComfyUI format (B, H, W) by squeezing the channel
        if bbox_mask_tensor.ndim == 4 and bbox_mask_tensor.shape[-1] == 1:
            bbox_mask_tensor = bbox_mask_tensor.squeeze(-1)

        return (cropped_tensor, bbox_mask_tensor)


# A dictionary that ComfyUI uses to know what nodes are available.
# You can NAMA_YOUR_NODE_HERE anything you want.
NODE_CLASS_MAPPINGS = {
    "ImageCropByAlpha": ImageCropByAlpha
}

# A dictionary that ComfyUI uses to display the name of the node in the UI.
# (This is optional)
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCropByAlpha": "Crop Image by Alpha"
}

# Test execution (optional, good for direct script testing)
if __name__ == "__main__":
    # Create a dummy RGBA image with transparent borders for testing
    test_img_pil = Image.new("RGBA", (200, 150), (0, 0, 0, 0))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(test_img_pil)
    draw.rectangle((50, 30, 150, 120), fill=(255, 0, 0, 100))  # Semi-transparent red box
    draw.rectangle((60, 40, 140, 110), fill=(0, 255, 0, 255))  # Opaque green box

    # Convert to tensor
    test_tensor = pil_to_tensor([test_img_pil])  # Batch of 1

    cropper = ImageCropByAlpha()

    print("--- Test 1: Threshold 50, Padding 10 ---")
    cropped_image_tensor, bbox_mask_tensor = cropper.crop_image(test_tensor, alpha_threshold=50, padding=10)

    # Convert back to PIL to check/save
    result_pil_batch = tensor_to_pil(cropped_image_tensor)
    mask_pil_batch = tensor_to_pil(bbox_mask_tensor.unsqueeze(-1))  # Add channel dim for tensor_to_pil

    if result_pil_batch:
        result_pil_batch[0].save("test_cropped_output.png")
        print("Saved test_cropped_output.png")
    if mask_pil_batch:
        mask_pil_batch[0].convert("L").save("test_bbox_mask_output.png")
        print("Saved test_bbox_mask_output.png")

    print(f"\nCropped image tensor shape: {cropped_image_tensor.shape}")
    print(f"BBox mask tensor shape: {bbox_mask_tensor.shape}")

    print("\n--- Test 2: Threshold 0 (like getbbox), Padding 0 ---")
    # Create an RGB image with a white background and a black square
    rgb_img_pil = Image.new("RGB", (100, 100), (255, 255, 255))
    draw_rgb = ImageDraw.Draw(rgb_img_pil)
    draw_rgb.rectangle((20, 20, 80, 80), fill=(0, 0, 0))
    rgb_tensor = pil_to_tensor([rgb_img_pil])

    cropped_rgb_tensor, _ = cropper.crop_image(rgb_tensor, alpha_threshold=0, padding=0)
    result_rgb_pil = tensor_to_pil(cropped_rgb_tensor)
    if result_rgb_pil:
        result_rgb_pil[0].save("test_rgb_cropped_output.png")
        print("Saved test_rgb_cropped_output.png")

    print("\n--- Test 3: Fully transparent image ---")
    transparent_img_pil = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    transparent_tensor = pil_to_tensor([transparent_img_pil])
    cropped_trans_tensor, _ = cropper.crop_image(transparent_tensor, alpha_threshold=10, padding=5)
    result_trans_pil = tensor_to_pil(cropped_trans_tensor)
    if result_trans_pil:
        print(f"Transparent image result size: {result_trans_pil[0].size} (should be original size)")