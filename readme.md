# Retrieval Augmented Few-shot Skull Stripping via Foundation Models
202510HY10027 BME Capstone Design 1  
Dohoon Kim, Jongmin Lee

![pipeline](/src//img/pipeline.svg)

---
# ❗ To-Do ❗
- [ ] 마스크 생성 오류 확인 & 고치기
- [ ] retrival 결과 검증
- [ ] 성능 측정
---
## How to run?
For more details, please check ```notebook/RAFS.ipynb```. 
```python
from model.RAFS import RAFS
import nibabel as nib
import torchvision.transforms.functional as TF

# Load Model
model = RAFS(faiss_index='BME_faiss.index',
             faiss_json='faiss_idx.json')

# Load Data
target_mri = nib.load('/home/kdh/code/BME_Capstone1/NFBS_Dataset/A00055447/sub-A00055447_ses-NFB3_T1w.nii.gz').get_fdata()
target_mri = target_mri[:, 150, :]
target_msk = nib.load('/home/kdh/code/BME_Capstone1/NFBS_Dataset/A00055447/sub-A00055447_ses-NFB3_T1w_brainmask.nii.gz').get_fdata()
target_msk = target_msk[:, 150, :]

# Prepare Data
image_input = TF.to_tensor(target_mri).float().cuda()#.shape
mask = model(image_input)

# Visualize Prediction
import matplotlib.pyplot as plt
plt.subplot(1, 3, 1)
plt.imshow(target_mri, cmap='gray')
plt.title('Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(target_msk, cmap='gray')
plt.title('GT')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(mask.detach().cpu().numpy()[0, 0]>0.0, cmap='gray')
plt.title('Prediction')
plt.axis('off')
```