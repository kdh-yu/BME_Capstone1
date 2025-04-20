from sam2.sam2_video_predictor import SAM2VideoPredictor
from preprocessing import preprocessing_dinov2, preprocessing_sam2
import torch
import faiss
import json
import os
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torch.nn as nn

from model.modeling.retrieval import Retrieval
from model.modeling.memory_encoder import MemoryEncoder
from model.modeling.mask_decoder import MaskDecoder

class RAFS(nn.Module):
    def __init__(self, 
                 faiss_index: str, 
                 faiss_json: str,
                 faiss_dir: str = "/home/kdh/code/BME_Capstone1/FAISS",
                 dinov2_model: str = "dinov2_vits14_reg",
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
        
        # Image Encoder
        self.image_encoder = self.sam2.image_encoder
        
        # Retrieval
        self.retrieval_module = Retrieval(faiss_index=faiss_index,
                                          faiss_json=faiss_json,
                                          dino_model=dinov2_model,
                                          device=device)
        
        # Memory Encoder
        self.memory_encoder = MemoryEncoder(sam2_model=sam2_model,
                                            device=device)
        
        # Mask Decoder
        self.mask_decoder = MaskDecoder(sam2_model=sam2_model,
                                        device=device)
        
    def forward(self, 
                img: torch.Tensor):
        
        # 1. Retrieve related image
        retrieval = self.retrieval_module(img)
        
        # 2. Get memory bank
        memory_feat, memory_pos = self.memory_encoder(retrieval)
        
        # 3. Get mask
        mask = self.mask_decoder(img,
                                 memory_feat,
                                 memory_pos)
        
        self.retrieval = retrieval
        self.memory_feat = memory_feat
        self.memory_pos = memory_pos
        self.mask = mask
        
        return mask
    
    def reset_cache(self):
        self.retrieval = []
        self.memory_feat = []
        self.memory_pos = []
        self.mask = []