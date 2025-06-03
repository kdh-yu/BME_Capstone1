# Libraries
import faiss
import torch
import json
import torch.nn as nn
import numpy as np

from processor import processor

class Retrieval(nn.Module):
    def __init__(self,
                 faiss_index: str = "BME_faiss.index",
                 faiss_json: str = "faiss_idx.json",
                 dino_model: str = "dinov2_vits14_reg",
                 device: str = "cuda"):
        super().__init__()
        self.device = device
        self.index = faiss.read_index(faiss_index)
        self.dino = torch.hub.load('facebookresearch/dinov2', dino_model).to(self.device).eval()
        with open(faiss_json, 'r') as f:
            self.table = json.load(f)
            
    def forward(self, 
                img: torch.Tensor,
                n: int = 8):
        
        img = processor(img, mode='dino')
        img = img.unsqueeze(0) 
        img = img.to(self.device)
        
        with torch.no_grad():
            query = self.dino(img)
        query = query.detach().cpu().numpy()
        faiss.normalize_L2(query)
        
        dist, index = self.index.search(query, n)
        
        image_retrieval = []
        mask_retrieval = []
        
        for i in reversed(index[0]):
            if i == -1:
                image = np.zeros((3, 256, 256))
                mask = np.zeros((1, 256, 256))
            else:
                image = np.load('./FAISS/' + self.table[str(i)] + '.npy')
                mask = np.load('./FAISS/' + self.table[str(i)] + '_mask.npy')
                
            image = processor(image, mode='sam')
            image_retrieval.append(image)
            
            mask = processor(mask, mode='mask')
            mask_retrieval.append(mask)
            
        return {
            'image' : image_retrieval,
            'mask'  : mask_retrieval,
            'n'     : n
        }
        
    def forward_manual(self, images, masks):
        image_retrieval = []
        mask_retrieval = []
        
        for img, msk in zip(images, masks):
            img = processor(img, mode='sam').float()
            image_retrieval.append(img)
            
            msk = processor(msk, mode='mask').float()
            mask_retrieval.append(msk)
        
        return {
            'image' : image_retrieval,
            'mask'  : mask_retrieval,
            'n'     : len(images)
        }