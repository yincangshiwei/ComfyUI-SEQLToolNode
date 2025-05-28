- - -
## 项目介绍

> 主要开发一些方便本人在ComfyUI处理遇到的场景工具节点，不打算整合一些复杂的功能（依赖环境过强）

## 节点介绍

> ImageCropAlphaNode：裁剪PNG图多余的透明空间，特别适合抠图后不需要保留原图的尺寸大小，这样插入到画布里就会自动居中。

### 参数解析：

- auto_threshold（自动阈值）：自动识别图像多余空间，计算出阈值进行调整。
- alpha_threshold（alpha阈值）：手动填写alpha层预留阈值，需关闭auto_threshold。
- padding（边距位置）：调整主体与画布的边距位置，暂时只设全局设置，不设置方向padding，有需再调。

### 示例图

![image](https://github.com/user-attachments/assets/ad7cb9ba-663c-4527-85a2-797e1881c02e)


