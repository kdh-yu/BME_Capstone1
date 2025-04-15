from sam2.sam2_image_predictor import SAM2ImagePredictor, SAM2Base
from preprocessing import preprocessing_dinov2, preprocessing_sam2
import torch
import faiss
import json
from PIL import Image

# Device
device = ('cuda' if torch.cuda.is_available()
          else 'mps' if torch.backends.mps.is_available()
          else 'cpu')

# Prepare SAM 2
model = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large").model.to(device)

image_encoder = model.image_encoder
memory_encoder = model.memory_encoder
memory_attention = model.memory_attention
image_decoder = model.sam_mask_decoder

# Prepare DINOv2
dino = torch.hub.load('facebookresearch/dinov2', f'dinov2_vits14_reg').to(device)

# Prepare FAISS
index = faiss.read_index('BME_faiss.index')
with open('faiss_idx.json', 'r') as f:
    retrieval_table = json.load(f)

# Img
img = torch.rand(1, 3, 256, 192).cuda()

# Retrieval
img_dino = preprocessing_dinov2(img)
feat_dino = dino.forward_features(img_dino)['x_norm_clstoken']

Dist, retrieval = faiss.search(img_dino.detach().numpy(), 3)

memory_img = []
memory_img_pos = []
memory_lbl = []
for ridx in retrieval[0]:
    filename = retrieval_table[ridx]
    
    rimg = Image.open(f"./FAISS/{filename}.png")
    rimg_feat = image_encoder(preprocessing_sam2(rimg))
    memory_img.append(rimg_feat['vision_features'])
    memory_img_pos.append(rimg_feat['vision_pos_enc'])
    
    rlbl = Image.open(f'./FAISS/{filename}_mask.png')
    memory_lbl.append(rlbl)
    
memory_img = torch.cat(memory_img, dim=0)
memory_img_pos = torch.cat(memory_img_pos, dim=0)
memory_lbl = torch.cat(memory_lbl, dim=0)

memory_bank = memory_encoder(memory_img, memory_lbl)

# Image
img_sam2 = preprocessing_sam2(img)

img_feat = image_encoder(img_sam2)

img_feat = memory_attention(
    curr=img_feat['vision_features'],
    curr_pos=img_feat['vision_pos_enc'],
    memory=memory_bank['vision_features'],
    memory_pos=memory_bank['vision_pos_enc'],
    num_obj_ptr_tokens=2
)

# Decocde
segmentation_map = image_decoder()