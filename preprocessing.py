import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

def preprocessing_dinov2(img: np.ndarray, 
                         input_shape=(266, 196)) -> torch.Tensor:
    """
    Resize a 256x192 medical image to 266x196 for DINOv2 input.

    Args:
        img (numpy.ndarray): Input image with shape (256, 192) or (256, 192, 3)
    
    Returns:
        torch.Tensor: Padded image of shape (3, 196, 266)
    """
    H, W = img.shape
    h = input_shape[0] - H
    w = input_shape[1] - W
    img = np.pad(img, ((h//2, h//2), (w//2, w//2)), mode='constant', constant_values=0)  # (H, W) -> (H', W')
    
    img = np.stack([img]*3, axis=-1)  # (H', W') -> (H', W', 3)
    
    # Convert to tensor
    transform = T.ToTensor() 
    img = transform(img)
    
    return img

def preprocessing_sam2(img: Image,
                       resize=False,
                       input_shape=(266, 196)) -> torch.Tensor:
    """
    Prepare MRI Slice for SAM 2 Input.

    Args:
        img (PIL.Image): Input image with shape (256, 192)
    
    Returns:
        torch.Tensor: Output image with shape (3, 256, 292)
    """
    if resize:
        H, W = img.shape
        h = input_shape[0] - H
        w = input_shape[1] - W
        img = np.pad(img, ((h//2, h//2), (w//2, w//2)), mode='constant', constant_values=0)  # (H, W) -> (H', W')
    #img = np.stack([img]*3, axis=-1)  # (H', W') -> (H', W', 3)
    # Convert to tensor
    transform = T.ToTensor() 
    img = transform(img)
    
    return img