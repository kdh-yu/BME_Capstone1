import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

def preprocessing_dinov2(img: torch.Tensor, 
                         input_shape: tuple[int, int] = (266, 196)
                         ) -> torch.Tensor:
    """
    Resize a 256x192 medical image to 266x196 for DINOv2 input.

    Args:
        img (torch.Tensor) : (256, 192)
    
    Returns:
        torch.Tensor       : (3, 266, 196)
    """
    _, H, W = img.shape
    h = max(0, input_shape[0] - H)
    w = max(0, input_shape[1] - W)
    padding = (w // 2, h // 2, w // 2, h // 2)
    img = TF.pad(img, padding, fill=0)  # still [1, H', W']
    img = img.expand(3, -1, -1)
    
    # 4) Ensure range [0,1]
    img = img.float()
    img = (img - img.min()) / img.max()
    
    return img

def preprocessing_sam2(img: Image,
                       input_shape=256
                       ) -> torch.Tensor:
    """
    Prepare MRI Slice for SAM 2 Input.

    Args:
        img (PIL.Image): Input image with shape (256, 192)
    
    Returns:
        torch.Tensor: Output image with shape (3, 256, 256)
    """
    if type(img) != torch.Tensor:
        img = TF.to_tensor(img)
    _, H, W = img.shape
    h_pad = max(0, input_shape - H)
    w_pad = max(0, input_shape - W)
    
    padding = (w_pad // 2, h_pad // 2, w_pad // 2, h_pad // 2)
    img = TF.pad(img, padding, fill=0)  # still [1, H', W']
    # 3) 1채널 → 3채널 복제
    img = img.expand(3, -1, -1)  # [3, H', W']

    # 4) Ensure range [0,1]
    img = img.float()
    img = (img - img.min()) / img.max()
    
    return img

def preprocessing_mask(img: Image,
                       input_shape: int = 256):
    img = TF.to_tensor(img)
    _, H, W = img.shape
    h_pad = max(0, input_shape - H)
    w_pad = max(0, input_shape - W)
    
    padding = (w_pad // 2, h_pad // 2, w_pad // 2, h_pad // 2)
    img = TF.pad(img, padding, fill=0)  # still [1, H', W']
    
    if img.max() > 1.0:
        img = img / img.max()
        img = img.float()
        
    return img