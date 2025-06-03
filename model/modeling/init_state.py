import torch
from collections import OrderedDict

def init_state_by_retrieval(
    retrieval,
    model,
    offload_video_to_cpu=False,
    offload_state_to_cpu=False,
    async_loading_frames=False,
    ):
    """Initialize an inference state."""
    images = torch.stack(retrieval)
    video_height = images.shape[2]
    video_width = images.shape[3]
    inference_state = {}
    inference_state["images"] = images
    inference_state["num_frames"] = len(images)
    inference_state["offload_video_to_cpu"] = offload_video_to_cpu
    inference_state["offload_state_to_cpu"] = offload_state_to_cpu
    inference_state["video_height"] = video_height
    inference_state["video_width"] = video_width
    inference_state["device"] = 'cuda'
    inference_state["storage_device"] = 'cuda'
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    inference_state["cached_features"] = {}
    inference_state["constants"] = {}
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    inference_state["output_dict_per_obj"] = {}
    inference_state["temp_output_dict_per_obj"] = {}
    inference_state["frames_tracked_per_obj"] = {}
    model._get_image_feature(inference_state, frame_idx=0, batch_size=1)
    return inference_state