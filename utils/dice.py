import torch

def DSC(pred: torch.Tensor, 
        y: torch.Tensor):
    pred_flat = pred.contiguous().view(-1)
    y_flat = y.contiguous().view(-1)
    intersection = (pred_flat * y_flat).sum()
    union = pred_flat.sum() + y_flat.sum()
    
    return (2. * intersection) / (union + 1e-8)