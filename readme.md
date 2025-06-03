# Retrieval Augmented Few-shot Skull Stripping: Implementation and Evaluation
202510HY10027 BME Capstone Design 1  
Dohoon Kim, Jong-Min Lee

![pipeline](/src//img/pipeline.png)

---
---
# How to run?

## How to build faiss index?
For more details, please check [```faiss.md```](faiss.md). 


## How to predict mask?
For more details, please check [```notebook/RAFS.ipynb```](notebook/RAFS.ipynb). 
```python
from model.RAFS import RAFS
import nibabel as nib

# Load Model
model = RAFS(faiss_index='BME_Capstone1/BME_faiss.index',
             faiss_json='BME_Capstone1/faiss_idx.json',
             n=8)

# Load Data
target_mri = nib.load('/home/kdh/code/BME_Capstone1/NFBS_Dataset/A00055447/sub-A00055447_ses-NFB3_T1w.nii.gz').get_fdata()
target_mri = target_mri[:, 150, :]
target_msk = nib.load('/home/kdh/code/BME_Capstone1/NFBS_Dataset/A00055447/sub-A00055447_ses-NFB3_T1w_brainmask.nii.gz').get_fdata()
target_msk = target_msk[:, 150, :]

# Predict Mask
out = model(target_mri.astype(np.float32))
```

## Does model really retrieve similar images?
For more details, please check [```notebook/retrieval.ipynb```](notebook/retrieval.ipynb). 

```python
import numpy as np
from model.modeling import Retrieval

# Load Retrieval module only. Or, You can use RAFS().retrieval_module.
retrieval_module = Retrieval()

target = nib.load('BME_Capstone1/data/NFBS_Dataset/A00028185/sub-A00028185_ses-NFB3_T1w.nii.gz').get_fdata()
target = target.astype(np.float32)
target = processor(target[:, 92, :], mode='dino')

retrieval_result = retrieval_module(target, n=8)
```