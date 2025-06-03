# Libraries
import torch.nn as nn

from model.modeling import init_state_by_retrieval

from processor import processor

class Propagate(nn.Module):
    def __init__(self,
                 sam2_model,
                 device: str = "cuda"):
        super().__init__()
        self.device = device
        self.sam2 = sam2_model
    
    def forward(self, 
                target,
                retrieval: dict):
               
        image = retrieval['image']
        mask  = retrieval['mask']
        
        target = processor(target, mode='sam')
        
        state = init_state_by_retrieval(image+[target], self.sam2)
        
        for i, m in enumerate(mask):
            self.sam2.add_new_mask(
                inference_state=state,
                frame_idx=i,
                obj_id=1,
                mask=m[0]
            )
        _, _, mask_logit = next(self.sam2.propagate_in_video(state, start_frame_idx=len(image)))
        mask = (mask_logit > 0.0).cpu().numpy()
        
        return mask, mask_logit