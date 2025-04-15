import nibabel as nib
import numpy as np
import os

def get_data(pid, basedir):
    route = os.path.join(basedir, pid)
    
    original = os.path.join(route, f'sub-{pid}_ses-NFB3_T1w.nii.gz')
    mask = os.path.join(route, f'sub-{pid}_ses-NFB3_T1w_brainmask.nii.gz')
    brain = os.path.join(route, f'sub-{pid}_ses-NFB3_T1w_brain.nii.gz')
    
    original = nib.load(original).get_fdata()
    mask = nib.load(mask).get_fdata()
    brain = nib.load(brain).get_fdata()
    
    return original, mask, brain

def get_slice(data, direction, idx):
    if direction == 'coronal':
        return data[idx, :, :]
    elif direction == 'axial':
        return data[:, idx, :]
    elif direction == 'sagittal':
        return data[:, :, idx]
    else:
        raise ValueError("Direction must be 0 (Coronal), 1 (Axial), or 2 (Sagittal)")
    
def get_slice_range(data, direction):
    slice_idx = {'coronal' : 0, 'axial' : 1, 'sagittal' : 2}
    st_idx = float('inf')
    end_idx = -1
    
    for i in range(data.shape[slice_idx[direction]]):
        if direction == 'coronal':
            data_slice = data[i, :, :]
        elif direction == 'axial':
            data_slice = data[:, i, :]
        elif direction == 'sagittal':
            data_slice = data[:, :, i]
        
        if np.count_nonzero(data_slice) > 0:
            st_idx = min(st_idx, i)
            end_idx = max(end_idx, i)
            
    if st_idx == float('inf') or end_idx == -1:
        return None, None
    
    return st_idx, end_idx
