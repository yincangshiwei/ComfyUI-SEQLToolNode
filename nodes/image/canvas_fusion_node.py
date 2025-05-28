import torch
import numpy as np
from PIL import Image, ImageDraw


class CanvasFusionNode:
    CATEGORY = "image"  # 节点将出现在 "image" 类别下
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image",)
    FUNCTION = "fuse_images"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "foreground_image": ("IMAGE",),
                "position": (["center", "left", "right", "top", "bottom"], {"default": "center"}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                # 当没有画布图像输入时，以下参数用于创建画布
                "canvas_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "canvas_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "canvas_color": ("STRING", {"default": "#FFFFFF"}),  # 默认白色
            },
            "optional": {
                "canvas_image": ("IMAGE",),
            }
        }

    def tensor_to_pil(self, tensor_image):
        """将ComfyUI的IMAGE张量 (B, H, W, C) 转换为Pillow图像列表"""
        # ComfyUI 图像是 float32, 0-1 范围
        # Pillow 需要 uint8, 0-255 范围
        images = []
        for i in range(tensor_image.shape[0]):
            img_np = tensor_image[i].cpu().numpy()
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8), 'RGB')
            images.append(img_pil)
        return images[0]  # 我们只处理批次中的第一张图

    def pil_to_tensor(self, pil_image):
        """将Pillow图像转换为ComfyUI的IMAGE张量 (1, H, W, C)"""
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        img_np = np.array(pil_image).astype(np.float32) / 255.0
        tensor_image = torch.from_numpy(img_np).unsqueeze(0)  # 添加批次维度
        return tensor_image

    def fuse_images(self, foreground_image, position, padding, canvas_width, canvas_height, canvas_color,
                    canvas_image=None):
        fg_pil_orig = self.tensor_to_pil(foreground_image)

        # 确保前景图是RGBA模式，以便处理透明度
        # 如果前景图没有alpha通道，则默认不透明
        if fg_pil_orig.mode != 'RGBA':
            fg_pil = fg_pil_orig.convert('RGBA')
        else:
            fg_pil = fg_pil_orig

        fg_w, fg_h = fg_pil.size

        canvas_pil = None
        if canvas_image is not None:
            # 使用输入的画布图像
            canvas_pil_input = self.tensor_to_pil(canvas_image)
            # 确保输入的画布是RGB模式（忽略其alpha，因为我们要基于它创建一个新的融合画布）
            if canvas_pil_input.mode != 'RGB':
                canvas_pil = canvas_pil_input.convert('RGB')
            else:
                canvas_pil = canvas_pil_input.copy()  # 使用副本
            c_w, c_h = canvas_pil.size
        else:
            # 创建新画布
            c_w, c_h = canvas_width, canvas_height
            try:
                # 验证颜色代码
                if not (canvas_color.startswith('#') and (
                        len(canvas_color) == 7 or len(canvas_color) == 9)):  # 支持 #RRGGBB 和 #RRGGBBAA
                    # 尝试直接用作颜色名称
                    Image.new('RGB', (1, 1), canvas_color)
                elif len(canvas_color) == 9:  # #RRGGBBAA
                    # PIL 不直接支持 #RRGGBBAA 创建图像，但支持 (r,g,b,a) 元组
                    # 为了简单，我们这里主要处理 #RRGGBB，如果用户想用带alpha的颜色创建画布，
                    # 他们应该提供一个带alpha的 canvas_image。
                    # 这里我们只取RGB部分
                    canvas_color_rgb = canvas_color[:7]
                    Image.new('RGB', (1, 1), canvas_color_rgb)  # 测试RGB部分
                    canvas_pil = Image.new('RGB', (c_w, c_h), canvas_color_rgb)
                else:  # #RRGGBB
                    canvas_pil = Image.new('RGB', (c_w, c_h), canvas_color)
            except ValueError:
                print(f"警告: 无效的画布颜色值 '{canvas_color}'。将使用默认白色 #FFFFFF。")
                canvas_color = "#FFFFFF"
                canvas_pil = Image.new('RGB', (c_w, c_h), canvas_color)

        # 计算粘贴位置
        paste_x, paste_y = 0, 0

        if position == "center":
            paste_x = (c_w - fg_w) // 2
            paste_y = (c_h - fg_h) // 2
        elif position == "left":
            paste_x = padding
            paste_y = (c_h - fg_h) // 2  # 垂直居中
        elif position == "right":
            paste_x = c_w - fg_w - padding
            paste_y = (c_h - fg_h) // 2  # 垂直居中
        elif position == "top":
            paste_x = (c_w - fg_w) // 2  # 水平居中
            paste_y = padding
        elif position == "bottom":
            paste_x = (c_w - fg_w) // 2  # 水平居中
            paste_y = c_h - fg_h - padding

        # 创建一个输出画布的副本（确保是RGB，因为前景带alpha）
        # 如果 canvas_pil 是从输入来的，它已经是RGB了
        # 如果是新创建的，它也是RGB
        output_pil = canvas_pil.copy()

        # 使用前景图的alpha通道作为遮罩进行粘贴
        # Pillow的paste方法会自动处理超出边界的情况，就像PS一样
        # 如果fg_pil是RGBA，第三个参数mask会使用其alpha通道
        output_pil.paste(fg_pil, (paste_x, paste_y), fg_pil if fg_pil.mode == 'RGBA' else None)

        return (self.pil_to_tensor(output_pil),)


# ComfyUI 节点注册
NODE_CLASS_MAPPINGS = {
    "CanvasFusionNode": CanvasFusionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CanvasFusionNode": "画布融合 (Canvas Fusion)"
}

print("加载 CanvasFusionNode 节点")