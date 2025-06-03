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
parser.add_argument('--epoch', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--mode', type=str)
parser.add_argument('--split', type=str)
args = parser.parse_args()

############ HYPERPARAMETER #################
EPOCH = args.epoch
BATCH_SIZE = args.batch_size
LR = args.lr
####################

os.makedirs('checkpoints', exist_ok=True)
if args.model == 'unet':
    model = UNet().cuda()
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
    
trainset = NFBS(json_path='nfbs_axial.json', mode=args.mode, split='train', base_dir='/home/kdh/code/BME_Capstone1/data/')

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

loss_fn = nn.BCEWithLogitsLoss()
optim = torch.optim.Adam(model.parameters(), lr=LR)

loss_log = []
best_valid_loss = float('inf')

#################### TRAIN ####################
model.train()
for epoch in range(EPOCH):
    epoch_loss = 0
    for data, mask in tqdm(trainloader):
        # 데이터 검증
        if torch.isnan(data).any() or torch.isinf(data).any():
            print("Problematic data detected!")
            continue
        
        optim.zero_grad()
        
        data = data.cuda()
        mask = mask.cuda()
        
        pred = model(data)
        loss = loss_fn(pred, mask)
        
        if torch.isnan(loss):
            print("Data stats:", data.mean().item(), data.std().item())
            print("Pred stats:", pred.mean().item(), pred.std().item())
            continue
        
        epoch_loss += loss.item()
        
        loss.backward()
        optim.step()
    
    avg_train_loss = epoch_loss / len(trainset)
        
    print(f"Epoch:\t{epoch+1}")
    print(f"Train Loss:\t{avg_train_loss:.4f}")
    
    loss_log.append(avg_train_loss)

torch.save(model.state_dict(), f'checkpoints/{args.model}_{args.mode}.pth')