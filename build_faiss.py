import nibabel as nib
import argparse
import os
import torch
import torchvision.transforms.functional as TF
import numpy as np
import faiss
from PIL import Image
from tqdm import tqdm
from preprocessing import preprocessing_dinov2

np.random.seed(2022094093)

parser = argparse.ArgumentParser()

parser.add_argument('--backbone', type=str, choices=['vits14', 'vitb14', 'vitl14', 'vitg14'], default='vits14', help='Specify the backbone size.')
parser.add_argument('--register', type=bool, default=True, help='Specify whether to use model with registers.')
parser.add_argument('--num-data', type=int, default=3, help='Choose how many data you will use to build FAISS Index.')
parser.add_argument('--data-path', type=str, default='/home/kdh/code/BME_Capstone1/FAISS', help='Specify where the data is.')
parser.add_argument('--slice', type=str, default='axial', help='Specify the direction of lices to build the FAISS Index.')
args = parser.parse_args()

device = ('cuda' if torch.cuda.is_available()
          else 'mps' if torch.backends.mps.is_available()
          else 'cpu')

reg = ''
if args.register:
    reg = '_reg'
dino = torch.hub.load('facebookresearch/dinov2', f'dinov2_{args.backbone}{reg}').to(device)

datalist = [i for i in os.listdir(args.data_path) if '_mask' not in i]
ids = 0

faiss_data = {}

faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(384))
feat_vectors = []

for data in tqdm(datalist):
    img = TF.to_tensor(Image.open(os.path.join('./FAISS', data)))
    img = preprocessing_dinov2(img, input_shape=(266, 196)).unsqueeze(0).to(device)
    data_feat = dino.forward_features(img)['x_norm_clstoken']
    data_feat = data_feat.detach().cpu()
    feat_vectors.append(data_feat.squeeze(0))
feat_vectors = np.vstack(feat_vectors)
faiss_index.add_with_ids(feat_vectors, np.arange(0, len(feat_vectors)))
faiss.write_index(faiss_index, 'BME_faiss.index')