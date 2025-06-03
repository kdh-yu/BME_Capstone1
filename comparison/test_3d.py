from unet import UNet
from swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from dataset import NFBS3D
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from processor import processor
import sys
import nibabel as nib
from torch.utils.data import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['unet', 'swin'], default='unet', help='Choose model to train.')
parser.add_argument('--batch_size', type=int)
parser.add_argument('--mode', type=str)
parser.add_argument('--split', type=str)
parser.add_argument('--weights', type=str)
args = parser.parse_args()

os.makedirs('checkpoints', exist_ok=True)
if args.model == 'unet':
    model = UNet().cuda()
    model.load_state_dict(torch.load(args.weights))
elif args.model == 'swin':
    model = SwinTransformerSys(
        img_size=256,  # 입력 이미지 크기
        patch_size=4,  # 패치 크기
        in_chans=3,    # 입력 채널 수
        num_classes=1, # 출력 클래스 수 (세그멘테이션 클래스 수)
        embed_dim=96,  # 임베딩 차원
        depths=[2, 2, 2, 2],  # 인코더의 각 레이어 깊이
        depths_decoder=[1, 2, 2, 2],  # 디코더의 각 레이어 깊이
        num_heads=[3, 6, 12, 24],  # 각 레이어의 어텐션 헤드 수
        window_size=8,  # 윈도우 크기
        mlp_ratio=4.,   # MLP 비율
        qkv_bias=True,  # QKV 바이어스 사용 여부
        drop_rate=0.,   # 드롭아웃 비율
        attn_drop_rate=0.,  # 어텐션 드롭아웃 비율
        drop_path_rate=0.1,  # 드롭 패스 비율
    ).cuda()
    model.load_state_dict(torch.load(args.weights))

class NFBS(Dataset):
    def __init__(self, faiss_list, data_path):
        self.faiss_list = faiss_list
        self.data_path = data_path
        self.data_list = [i for i in os.listdir(self.data_path) if i not in self.faiss_list]
        self.data_name = '{}/sub-{}_ses-NFB3_T1w.nii.gz'
        self.mask_name = '{}/sub-{}_ses-NFB3_T1w_brainmask.nii.gz'
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        dname = self.data_list[idx]
        
        data_path = os.path.join(self.data_path, self.data_name.format(dname, dname))
        mask_path = os.path.join(self.data_path, self.mask_name.format(dname, dname))
        
        data = nib.load(data_path).get_fdata().astype(np.float32)
        mask = nib.load(mask_path).get_fdata().astype(np.float32)
        
        return data, mask
    
testset = NFBS(faiss_list=['A00060407' 'A00028185' 'A00060925'], 
               data_path='/home/kdh/code/BME_Capstone1/data/NFBS_Dataset')

#################### TEST ####################
model.eval()
with torch.no_grad():
    dice = 0
    
    for data, mask in tqdm(testset):
        pred_sum = 0
        target_sum = 0
        intersection_sum = 0
        num_slices = data.shape[1]
        
        # Process each slice
        for i in range(num_slices):
            # Get current slice
            current_data = data[:, i, :].copy()  # Use copy to avoid modifying original
            current_mask = mask[:, i, :].copy()
            
            # Process data
            current_data = processor(current_data, mode='sam').cuda().unsqueeze(0)
            current_mask = processor(current_mask, mode='mask').cuda().unsqueeze(0)
            
            # Get prediction
            pred = model(current_data)
            pred_mask = (torch.sigmoid(pred) > 0.5)
            
            # Calculate metrics for this slice
            intersection = (pred_mask * current_mask).sum()
            pred_sum += pred_mask.sum()
            target_sum += current_mask.sum()
            intersection_sum += intersection
            
            torch.cuda.empty_cache()
        
        # Calculate Dice score for this volume
        volume_dice = (2. * intersection_sum) / (pred_sum + target_sum)
        dice += volume_dice
    
    # Calculate average Dice score across all volumes
    avg_dice = dice / len(testset)
    print(f"Average Dice Score: {avg_dice:.4f}")
