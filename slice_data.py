import os
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm

import json

np.random.seed(2022094093)
dsize = 3
datalist = os.listdir('/home/kdh/code/BME_Capstone1/NFBS_Dataset')
chosen_data = np.random.choice(datalist, size=dsize)
ids = 0
raw_filename = "sub-{}_ses-NFB3_T1w.nii.gz"
mask_filename = "sub-{}_ses-NFB3_T1w_brainmask.nii.gz"
index = {}
idx = 0

print(f"Now preparing axial slice from {dsize} MRI data.")

for d in chosen_data:
    data_root = os.path.join('/home/kdh/code/BME_Capstone1/NFBS_Dataset', d)
    raw_data = nib.load(os.path.join(data_root, raw_filename.format(d))).get_fdata()
    mask_data = nib.load(os.path.join(data_root, mask_filename.format(d))).get_fdata()
    dmax = np.max(raw_data)
    
    for i in tqdm(range(raw_data.shape[1])):
        sliced_raw = raw_data[:, i, :]
        sliced_raw = (sliced_raw * 255 / dmax).astype(np.uint8)
        sliced_mask = mask_data[:, i, :]
        sliced_mask = (sliced_mask * 255).astype(np.uint8)
        
        img = Image.fromarray(sliced_raw)
        img.save(f'./FAISS/{d}_axial_{i}.png')
        img = Image.fromarray(sliced_mask)
        img.save(f'./FAISS/{d}_axial_{i}_mask.png')
        
        index[idx] = f"{d}_axial_{i}"
        idx += 1
        
with open('faiss_idx.json', 'w') as f:
    json.dump(index, f, indent=4)