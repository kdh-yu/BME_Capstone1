# Libraries
import faiss
import nibabel as nib
import matplotlib.pyplot as plt
import torch
import json
from PIL import Image
import torch.nn as nn
import torchvision.transforms.functional as TF

from preprocessing import preprocessing_dinov2, preprocessing_sam2, preprocessing_mask

class Retrieval(nn.Module):
    def __init__(self,
                 faiss_index: str = "BME_faiss.index",
                 faiss_json: str = "faiss_idx.json",
                 dino_model: str = "dinov2_vits14_reg",
                 device: str = "cuda"):
        super().__init__()
        self.device = device
        self.index = faiss.read_index(faiss_index)
        self.dino = torch.hub.load('facebookresearch/dinov2', dino_model).to(self.device)
        with open(faiss_json, 'r') as f:
            self.table = json.load(f)
    
    def forward(self, 
                img: torch.Tensor,
                n: int = 8):
        img = preprocessing_dinov2(img)
        img = img.unsqueeze(0)  # 배치차원 추가
        img = img.to(self.device)

        query = self.dino.forward_features(img)['x_norm_clstoken']
        query = query.detach().cpu().numpy()
        
        _, index = self.index.search(query, n)
        
        image_retrieval = []
        mask_retrieval = []
        
        for i in index[0]:
            image = Image.open('./FAISS/' + self.table[str(i)] + '.png')
            image = preprocessing_sam2(image)
            image_retrieval.append(image)
            
            mask = Image.open('./FAISS/' + self.table[str(i)] + '_mask.png')
            mask = preprocessing_mask(mask)
            mask_retrieval.append(mask)
            
        return {
            'image' : image_retrieval,
            'mask'  : mask_retrieval
        }