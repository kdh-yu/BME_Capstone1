import matplotlib.pyplot as plt
import torch

def denormalize(img: torch.Tensor):
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(img.device).view(1, 3, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).to(img.device).view(1, 3, 1, 1)
    img_denormalized = img * std + mean
    img_denormalized = img_denormalized.clamp(0, 1)
    img_denormalized = img_denormalized.squeeze(0)
    return img_denormalized


    