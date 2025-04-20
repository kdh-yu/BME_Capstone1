# Libraries
import torch
import json
from PIL import Image
import torch.nn as nn
import torchvision.transforms.functional as TF

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.build_sam import build_sam2_video_predictor

from preprocessing import preprocessing_dinov2, preprocessing_sam2

class MemoryEncoder(nn.Module):
    def __init__(self,
                 sam2_model: str = "facebook/sam2-hiera-large",
                 device: str = "cuda"):
        super().__init__()
        self.device = device
        self.sam2 = SAM2VideoPredictor.from_pretrained(
            "facebook/sam2-hiera-large",
            hydra_overrides_extra=[
                "model.image_size=256",
                "+model.backbone_stride=16",
            ],
        )
        self.image_encoder = self.sam2.image_encoder.to(self.device)
        self.memory_encoder = self.sam2.memory_encoder.to(self.device)
    
    def forward(self, 
                retrieval: dict):

        # 1. Get image and mask from retrieved data
        image = retrieval['image']
        mask = retrieval['mask']
        
        mem_feats = []
        mem_pos = []
        
        # 2. Encode them as memory, and store
        for img, msk in zip(image, mask):
            img = img.unsqueeze(0).to(self.device)
            msk = msk.unsqueeze(0).to(self.device)
            
            maskmem_feat, maskmem_pos = self.encode_memory(img, msk)
            mem_feats.append(maskmem_feat)
            mem_pos.append(maskmem_pos[-1])
        
        # 3. Concatenate them
        # [B, H, W] * N -> [N, B, H, W]
        memory = torch.cat(mem_feats, dim=0).to(self.device)
        memory_pos = torch.cat(mem_pos, dim=0).to(self.device)
        
        return memory, memory_pos
        
    def encode_memory(self, support_img, support_mask):
        backbone_out = self.sam2.forward_image(support_img)
        _, vision_feats, vision_pos, feat_sizes = self.sam2._prepare_backbone_features(backbone_out)

        maskmem_feat, maskmem_pos = self.sam2._encode_new_memory(
            current_vision_feats=vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=support_mask,
            object_score_logits=torch.ones_like(support_mask[:,0]),  # dummy scores
            is_mask_from_pts=False,
        )
        return maskmem_feat, maskmem_pos