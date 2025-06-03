from model.RAFS import RAFS

model = RAFS(faiss_index='/home/kdh/code/BME_Capstone1/BME_faiss.index',
             faiss_json='/home/kdh/code/BME_Capstone1/faiss_idx.json',
             n=8)

from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import os

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
    
nfbs = NFBS(faiss_list=['A00060407' 'A00028185' 'A00060925'], 
            data_path='/home/kdh/code/BME_Capstone1/data/NFBS_Dataset')

def DSC(pred, y):
    pred_flat = pred.contiguous().view(-1)
    y_flat = y.contiguous().view(-1)
    intersection = (pred_flat * y_flat).sum()
    union = pred_flat.sum() + y_flat.sum()
    
    return (2. * intersection) / (union)

import torch
from tqdm import tqdm
import numpy as np
from processor import processor
import torchvision.transforms.functional as TF

dice_slice = 0
num_slice = 0

dice_volume = 0
num_volume = 0

model.eval()
with torch.no_grad():
    for mri, msk in nfbs:
        intersection_vol = 0
        pred_vol = 0
        gt_vol = 0
        
        for i in tqdm(range(mri.shape[1])):
            x = mri[:, i, :]
            if x.sum() == 0: continue
            x = (x - x.min()) / (x.max() - x.min())
            y_true = msk[:, i, :]
            y_true = processor(y_true, mode='mask')
            
            pred, _ = model(x)
            pred = processor(pred[0], mode='mask')
            
            pred_vol += pred.sum()
            gt_vol += y_true.sum()
            intersection_vol += (pred * y_true).sum()
            
            
            if y_true.sum() == 0:
                continue
            
            dice_score = DSC(pred, y_true)
            dice_slice += dice_score
            num_slice += 1
            
        dice_volume += (2. * intersection_vol) / (pred_vol + gt_vol)
        num_volume += 1
        
print(f"Valid slice:\t{num_slice}")
print(f"Average Slice-wise Dice Score:\t{dice_slice / num_slice}")
print(f"Average Volume-wise Dice Score:\t{dice_volume / num_volume}")