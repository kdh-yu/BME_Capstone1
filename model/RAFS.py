import torch
import torch.nn as nn

from sam2.sam2_video_predictor import SAM2VideoPredictor
from processor import processor

from model.modeling import Retrieval
from model.modeling import Propagate

class RAFS(nn.Module):
    def __init__(self, 
                 faiss_index: str, 
                 faiss_json: str,
                 n: int = 8,
                 faiss_dir: str = "/home/kdh/code/BME_Capstone1/FAISS",
                 dinov2_model: str = "dinov2_vits14_reg",
                 sam2_model: str = "facebook/sam2-hiera-large",
                 device: str = "cuda"):
        super().__init__()
        self.device = device
        self.sam2 = SAM2VideoPredictor.from_pretrained(
            sam2_model,
            hydra_overrides_extra=[
                "model.image_size=256",
                "+model.backbone_stride=16",
            ],
        )
        
        # Retrieval
        self.retrieval_module = Retrieval(faiss_index=faiss_index,
                                          faiss_json=faiss_json,
                                          dino_model=dinov2_model,
                                          device=device)
        self.n = n
        
        self.propagate = Propagate(sam2_model=self.sam2,
                                   device=device)
        
    def forward(self, 
                img: torch.Tensor):
        
        # 1. Retrieve related image
        retrieval = self.retrieval_module(img, n=self.n)
        
        mask, mask_logit = self.propagate(img, retrieval)
        
        return mask, mask_logit
    
    def forward_manual(self,
                       img,
                       retrieval: dict[str]):
        
        if not isinstance(img, torch.Tensor):
            img = processor(img, mode='sam')
                
        mask, mask_logit = self.propagate(img, retrieval)
        
        return mask, mask_logit
        