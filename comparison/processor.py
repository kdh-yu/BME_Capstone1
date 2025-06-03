import numpy as np
import PIL
import PIL.PngImagePlugin
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])


def processor(img,
              mode: str):
    # Image Type
    if type(img) == PIL.PngImagePlugin.PngImageFile or type(img) == Image.Image:
        img = TF.to_tensor(img)
    elif type(img) == np.ndarray:
        img = torch.from_numpy(img)
    if img.ndim == 2:
        img = img.unsqueeze(0)
        
    if mode == 'mask':
        _, H, W = img.shape
        h_pad = max(0, 256 - H)
        w_pad = max(0, 256 - W)
        padding = (w_pad // 2, h_pad // 2, w_pad // 2, h_pad // 2)
        
        img = TF.pad(img, padding, fill=0)  # still [1, H', W']
        
        return img
    
    if mode == 'dino':
        img = img.expand(3, -1, -1)  # [3, H', W']  
        img = img.unsqueeze(0)
        img = F.interpolate(img, size=(518, 518))
        img = img.squeeze(0)
        
        img = (img - img.min()) / (img.max() - img.min())
        
        return img
    
    if mode == 'sam':
        *_, H, W = img.shape
        h_pad = max(0, 256 - H)
        w_pad = max(0, 256 - W)
        padding = (w_pad // 2, h_pad // 2, w_pad // 2, h_pad // 2)
        img = TF.pad(img, padding, fill=0)  # still [1, H', W']

        img = img.expand(3, -1, -1)
        img = (img - img.min()) / (img.max() - img.min())
        
        return img