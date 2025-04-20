# Libraries
import torch
import json
from PIL import Image
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.build_sam import build_sam2_video_predictor

from preprocessing import preprocessing_dinov2, preprocessing_sam2

class MaskDecoder(nn.Module):
    def __init__(self,
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
        self.memory_attention = self.sam2.memory_attention
        self.mask_decoder = self.sam2.sam_mask_decoder
        
    def forward(self,
                query_img: torch.Tensor,
                memory: torch.Tensor,
                memory_pos: torch.Tensor
                ) -> torch.Tensor:
        """
        query 이미지 + memory bank를 입력받아 segmentation mask 생성
        output: [1, 1, H, W] mask logits
        """
        img = preprocessing_sam2(query_img)
        img = img.unsqueeze(0).to(self.device)  # [1, C, H, W]

        # backbone feature 뽑기
        backbone_out = self.sam2.forward_image(img)
        _, vision_feats, vision_pos, feat_sizes = self.sam2._prepare_backbone_features(backbone_out)

        # 픽셀 feature 준비
        pix_feat = vision_feats[-1]  # [B, C, H, W]
        pix_feat = pix_feat.view(1, 256, 16, 16)
        pix_seq = pix_feat.flatten(2).permute(2, 0, 1)  # [HW, B, C]
        pos_seq = vision_pos[-1]
        pos_seq = pos_seq.view(1, 256, 16, 16)
        pos_seq = pos_seq.flatten(2).permute(2, 0, 1)  # [HW, B, C]

        N, C, H, W = memory.shape
        memory = memory.permute(0, 2, 3, 1)  # [N, H, W, C]
        memory_pos = memory_pos.permute(0, 2, 3, 1)
        memory = memory.reshape(-1, C)
        memory_pos = memory_pos.reshape(-1, C)
        memory = memory.unsqueeze(1)
        memory_pos = memory_pos.unsqueeze(1)
        
        #print(pix_seq.shape)
        #print(pos_seq.shape)
        #print(memory.shape)
        #print(memory_pos.shape)
        
        # memory attention 적용
        pix_feat_updated = self.memory_attention(
            curr=pix_seq,
            curr_pos=pos_seq,
            memory=memory,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=0,
        )
        
        # [HW, B, C] -> [B, C, HW] -> [B, C, H, W]
        pix_feat_updated = pix_feat_updated.permute(1, 2, 0).view(1, 256, 16, 16)
        
        feat0 = vision_feats[0].permute(1, 2, 0).contiguous()  # [B, C, HW]
        feat0 = feat0.view(1, 32, 64, 64)  # (B, C, H, W)

        feat1 = vision_feats[1].permute(1, 2, 0).contiguous()  # [B, C, HW]
        feat1 = feat1.view(1, 64, 32, 32)  # (B, C, H, W)

        # (필요하면 여기서 16x16으로 resize)
        #feat0 = F.interpolate(feat0, size=(16,16), mode='bilinear', align_corners=False)
        #feat1 = F.interpolate(feat1, size=(16,16), mode='bilinear', align_corners=False)

        high_res_features = [feat0, feat1]

        dummy_points = torch.tensor([[[0, 0]]], device=img.device).float()  # 이미지 중앙
        dummy_labels = torch.tensor([[0]], device=img.device)  # foreground

        
        point_inputs = {
            "point_coords": dummy_points,
            "point_labels": dummy_labels,
        }

        low_res_multimasks, high_res_multimasks, ious, low_res_masks, high_res_masks, obj_ptr, obj_scores = \
        self.sam2._forward_sam_heads(
            backbone_features=pix_feat_updated,
            point_inputs=None,
            mask_inputs=None,
            high_res_features=high_res_features,      # 또는 [high_res_map] 리스트
            multimask_output=False,
        )

        return high_res_masks  # [1, 1, H, W]        