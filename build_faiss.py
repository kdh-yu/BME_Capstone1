import argparse
import json
import os

import faiss
import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from processor import processor

np.random.seed(2022094093)
torch.manual_seed(2022094093)

#############################################################
# Parse Arguments
#############################################################
parser = argparse.ArgumentParser()

parser.add_argument('--backbone', type=str, choices=['vits14', 'vitb14', 'vitl14', 'vitg14'], default='vits14', help='Specify the backbone size.')
parser.add_argument('--register', type=bool, default=True, help='Specify whether to use model with registers.')
parser.add_argument('--num-data', type=int, default=3, help='Choose how many data you will use to build FAISS Index.')
parser.add_argument('--data-path', type=str, default='BME_Capstone1/FAISS', help='Specify where the data is.')
parser.add_argument('--slice', type=str, default='axial', help='Specify the direction of lices to build the FAISS Index.')
args = parser.parse_args()

#############################################################
# Settings
#############################################################

device = ('cuda' if torch.cuda.is_available()
          else 'mps' if torch.backends.mps.is_available()
          else 'cpu')

reg = ''
if args.register:
    reg = '_reg'
dino = torch.hub.load('facebookresearch/dinov2', f'dinov2_{args.backbone}{reg}').to(device).eval()

# Delete all files in FAISS directory
faiss_dir = '/home/kdh/code/BME_Capstone1/FAISS'
if os.path.exists(faiss_dir):
    for filename in os.listdir(faiss_dir):
        file_path = os.path.join(faiss_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print("All files in FAISS directory have been deleted.")

dsize = 3
datalist = os.listdir('/home/kdh/code/BME_Capstone1/data/NFBS_Dataset')

#############################################################
# Prepare FAISS Data
#############################################################
chosen_data = np.random.choice(datalist, size=dsize)

print(f'''Among {len(datalist)} data, {",".join(chosen_data)} will be used to construct FAISS Index.''')
ids = 0
raw_filename = "sub-{}_ses-NFB3_T1w.nii.gz"
mask_filename = "sub-{}_ses-NFB3_T1w_brainmask.nii.gz"
index = {}
idx = 0

#############################################################
# Slice data and Build FAISS
#############################################################
print(f"Now preparing axial slice from {dsize} MRI data.")

faiss_index = faiss.IndexFlatL2(384)

for data in chosen_data:
    data_root = os.path.join('/home/kdh/code/BME_Capstone1/data/NFBS_Dataset', data)
    raw_data = nib.load(os.path.join(data_root, raw_filename.format(data))).get_fdata()
    mask_data = nib.load(os.path.join(data_root, mask_filename.format(data))).get_fdata()
    
    for i in tqdm(range(raw_data.shape[1])):
        sliced_raw = raw_data[:, i, :]
        sliced_raw = sliced_raw.astype(np.float32)
        sliced_mask = mask_data[:, i, :]
        
        np.save(f'./FAISS/{data}_axial_{i}.npy', sliced_raw)
        np.save(f'./FAISS/{data}_axial_{i}_mask.npy', sliced_mask)
        
        img = processor(sliced_raw, mode='dino')
        img = img.unsqueeze(0)
        img = img.to(device)
        
        with torch.no_grad():
            data_feat = dino(img)
        data_feat = data_feat.detach().cpu().numpy()
        faiss.normalize_L2(data_feat)
        faiss_index.add(data_feat)
                
        index[idx] = f"{data}_axial_{i}"
        idx += 1
        
with open('faiss_idx.json', 'w') as f:
    json.dump(index, f, indent=4)
    
faiss.write_index(faiss_index, 'BME_faiss.index')

print(f"Successfully saved {faiss_index.ntotal} vectors with {faiss_index.d} dimensions")

