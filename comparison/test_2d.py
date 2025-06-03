from unet import UNet
from swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from dataset import NFBS
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import sys

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
    
testset = NFBS(json_path='nfbs_axial.json', mode=args.mode, split='test', base_dir='/home/kdh/code/BME_Capstone1/data/')

testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)

loss_log = []

def dice_score(pred, target):
    """
    pred: 모델의 예측값 (B, 1, H, W)
    target: 실제 마스크 (B, 1, H, W)
    """
    pred = torch.sigmoid(pred)  # BCEWithLogitsLoss를 사용하므로 sigmoid 적용
    pred = (pred > 0.5).float()  # 0.5를 임계값으로 이진화
    
    intersection = (pred * target).sum()
    dice = (2. * intersection) / (pred.sum() + target.sum())
    
    return dice

#################### TEST ####################
model.eval()
with torch.no_grad():
    total_dice = 0
    total_samples = 0
    
    for data, mask in tqdm(testloader):
        data = data.cuda()
        mask = mask.cuda()
        
        pred = model(data)
        
        # 각 샘플별로 Dice score 계산
        for i in range(data.size(0)):
            if mask[i].sum() > 0:  # 빈 마스크가 아닌 경우만 처리
                sample_dice = dice_score(pred[i:i+1], mask[i:i+1])
                total_dice += sample_dice
                total_samples += 1
    
    if total_samples > 0:
        avg_dice = total_dice / total_samples
        print(f"Average Dice Score: {avg_dice:.4f}")
        print(f"Total processed samples: {total_samples}")
    else:
        print("No valid samples were processed")
