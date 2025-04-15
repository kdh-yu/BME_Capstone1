from sam2.sam2_image_predictor import SAM2ImagePredictor, SAM2Base
from preprocessing import preprocessing_dinov2, preprocessing_sam2
import torch
import faiss
import json
import os
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torch.nn as nn

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
        # FAISS Index
        self.faiss = faiss.read_index(faiss_index)
        with open(faiss_json, 'r') as f:
            self.faiss_table = json.load(f)
        self.faiss_dir = faiss_dir
            
        # DINOv2
        self.dino = torch.hub.load('facebookresearch/dinov2', dinov2_model).to(device)
            
        # SAM 2 Modules
        self.sam2_model = SAM2ImagePredictor.from_pretrained(sam2_model).model.to(device)
        self.image_encoder = self.sam2_model.image_encoder
        self.memory_encoder = self.sam2_model.memory_encoder
        self.memory_attention = self.sam2_model.memory_attention
        self.image_decoder = self.sam2_model.sam_mask_decoder
        
    def _retrieve_image(self, img, n=3):
        """
        이미지를 인풋으로 받아, n개의 Retrieval된 이미지를 반환.
        
        input
            - img : torch.Tensor [H, W]
        output
            - image : torch.Tensor [B, 3, H, W]
            - mask : torch.Tensor [B, 1, H, W]
        """
        img = preprocessing_dinov2(img).unsqueeze(0)
        img = img.to(self.device)
        img_feat = self.dino.forward_features(img)['x_norm_clstoken'].detach().cpu().numpy()
        _, index = self.faiss.search(img_feat, n)
        
        retrieved_image = []
        retrieved_mask = []
        
        for i in index[0]:
            r = self.faiss_table[str(i)]
            r_image = Image.open(os.path.join(self.faiss_dir, r+'.png')).convert('RGB')
            r_image = preprocessing_sam2(r_image)
            retrieved_image.append(r_image)
            
            r_mask = Image.open(os.path.join(self.faiss_dir, r+'_mask.png')).convert("L")
            r_mask = T.ToTensor()(r_mask)
            r_mask = (r_mask > 0).float()
            retrieved_mask.append(r_mask)
            
        retrieved_image = torch.stack(retrieved_image).to(self.device)  # [B, 3, H, W]
        retrieved_mask = torch.stack(retrieved_mask).to(self.device)  # [B, 1, H, W]
        
        return {
            'image' : retrieved_image,
            'mask' : retrieved_mask
        }
    
    def _get_memory_bank(self, retrieval):
        image = retrieval['image']
        mask = retrieval['mask']
        
        pix_feat = self.image_encoder(image)['vision_features']
        
        if mask.dtype != torch.float32:
            mask = mask.float()
        if mask.max() > 1.0:
            mask = mask / 255.0

        memory_bank = self.memory_encoder(
            pix_feat, mask, skip_mask_sigmoid=True
        )
        return memory_bank
    
    def _apply_memory_attention(self, image, memory_bank):
        """
        query image에 memory attention을 적용

        input
            - query_img: [1, 3, H, W]
            - memory_bank: [B, C, H', W']  # from _get_memory_bank

        output
            - attended_feat: [1, C, H', W']
        """
        query_feat = self.image_encoder(image)  # [1, C, H', W']

        attended_feat = self.memory_attention(
            curr=query_feat['vision_features'],
            curr_pos=query_feat['vision_pos_enc'],
            memory=memory_bank['vision_features'],
            memory_pos=['vision_pos_enc'],
        )

        return attended_feat, query_feat
    
    def _postprocess_mask(self, mask_logits, threshold=0.5, resize_to=None):
        """
        input
            - mask_logits: [1, 1, H, W] (float logits)
            - threshold: binary mask로 만들 기준값
            - resize_to: (H_orig, W_orig) - 원래 이미지 크기

        output
            - soft_mask: [H, W] in (0,1)
            - binary_mask: [H, W] in {0,1}
        """
        # 1. sigmoid 적용
        soft_mask = torch.sigmoid(mask_logits.squeeze(0).squeeze(0))  # [H, W]

        # 2. binary mask 생성
        binary_mask = (soft_mask > threshold).float()

        # 3. 필요시 resize
        if resize_to is not None:
            import torch.nn.functional as F
            H, W = resize_to
            soft_mask = F.interpolate(soft_mask.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze()
            binary_mask = (soft_mask > threshold).float()

        return soft_mask, binary_mask

    def forward(self, 
                query_img: torch.Tensor, 
                n: int = 3, 
                threshold: float = 0.5):
        """
        RAFS 전체 추론 파이프라인

        input
            - query_img: [H, W] 또는 [1, H, W] (grayscale)
            - n: retrieval할 support 이미지 수
            - threshold: binary mask화 임계값

        output
            - soft_mask: [H, W], float in (0, 1)
            - binary_mask: [H, W], float in {0, 1}
        """
        # 1. (옵션) query image preprocessing (grayscale → RGB)
        if query_img.ndim == 2:
            query_img = query_img.unsqueeze(0)  # [1, H, W]
        if query_img.shape[0] == 1:
            query_img = query_img.repeat(3, 1, 1)  # [3, H, W]

        query_img = query_img.unsqueeze(0).to(self.device)  # [1, 3, H, W]
        orig_size = query_img.shape[-2:]  # (H, W)

        # 2. retrieve similar images + masks
        retrieval = self._retrieve_image(query_img.squeeze(0), n=n)  # input은 [3, H, W]

        # 3. get memory bank
        memory_bank = self._get_memory_bank(retrieval)  # [B, C, H', W']

        # 4. apply memory attention
        attended_feat, query_feat = self._apply_memory_attention(query_img, memory_bank)

        # 5. decode to segmentation mask
        output_mask = self.image_decoder(attended_feat, query_feat)  # [1, 1, H', W']

        # 6. postprocess
        soft_mask, binary_mask = self._postprocess_mask(output_mask, threshold=threshold, resize_to=orig_size)

        return soft_mask, binary_mask