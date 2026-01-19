from datasets import *
import torch
from tqdm import tqdm
from models import *
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as F2
import warnings
import logging
import torchvision.transforms.functional as VF
from torch.utils.data import DataLoader, default_collate
from run import SCALE_FACTOR, Engine
import gradio as gr
import cv2
from PIL import Image
from torchvision import transforms

warnings.filterwarnings("ignore")

# ===============================
# Sliding window utilities
# ===============================
def window_composite(patches, window_size=(384, 384), stride=128):
    image = None
    patch_h, patch_w = window_size
    for i, patch in enumerate(patches):
        if i == 0:
            image = patch
        else:
            blend_width = patch_w - stride
            prev_to_blend = image[:, :, -blend_width:]
            next_to_blend = patch[:, :, :blend_width]
            blend_factor = torch.sigmoid(
                torch.tensor(np.linspace(-3, 3, blend_width), device=image.device)
            )
            blend = (1 - blend_factor) * prev_to_blend + blend_factor * next_to_blend
            image[:, :, -blend_width:] = blend
            patch_remain = patch[:, :, blend_width:]
            image = torch.cat([image, patch_remain], dim=-1)
    return image


def sliding_window(image, window_size=(384, 384), stride=128):
    if isinstance(image, torch.Tensor):
        if image.shape[0] == 1:
            image = image.squeeze(0)
        image = image.permute(1, 2, 0).detach().cpu().numpy()

    image = np.pad(
        image,
        ((0, 0), (0, stride - image.shape[1] % stride), (0, 0)),
        'constant'
    )
    h, w, _ = image.shape
    assert h == 384, "FSC-147 assume image height is 384."

    patches, intervals = [], []
    for i in range(0, w - window_size[1] + 1, stride):
        patch = image[:, i:i + window_size[1], :]
        patches.append(patch)
        intervals.append([i, i + window_size[1]])

    patches = np.array(patches).transpose(0, 3, 1, 2)
    return patches, np.array(intervals)



img_path = "/home/anhkhoa/Downloads/birds_heatmap_overlay.png"
prompt = 14.381801452
# ===============================
# Gradio inference
# ===============================
def infer(img, prompt):
    # đọc heatmap
    heatmap_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if heatmap_bgr is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")

    # BGR -> RGB cho Gradio
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # prompt từ textbox là string -> float
    pred_cnt = prompt

    return heatmap_rgb, pred_cnt

demo = gr.Interface(
    fn=infer,
    inputs=[
        gr.Image(label="Image", type="numpy"),
        gr.Textbox(lines=1, label="Prompt (What would you like to count)", value="14.381801452"),
    ],
    outputs=[
        gr.Image(label="Heatmap Pred"),
        gr.Number(label="Pred Count"),
    ],
)

demo.launch(share=True)

