import os

import torch
import torch.nn.functional as F
import numpy as np

from einops import rearrange

from .masactrl_utils import AttentionBase
import cv2
from torchvision.utils import save_image
#from gaussian_smoothing import GaussianSmoothing
from PIL import Image
import matplotlib.pyplot as plt
class MutualSelfAttentionControl(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }

    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, model_type="SD"):
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            model_type: the model type, SD or SDXL
        """
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        print("MasaCtrl at denoising steps: ", self.step_idx)
        print("MasaCtrl at U-Net layers: ", self.layer_idx)

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        out_u = self.attn_batch(qu, ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c = self.attn_batch(qc, kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)
        out = torch.cat([out_u, out_c], dim=0)

        return out

class MutualSelfAttentionControlMaskAuto(MutualSelfAttentionControl):
    @torch.no_grad()
    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, thres=0.35, ref_token_idx=[1], cur_token_idx=[1], mask_save_dir=None, model_type="SD",last_idx=None,cross_attns_mask=None,box=None,batch_size=1,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps, model_type)
        print("Using MutualSelfAttentionControlMaskAuto")
        self.thres = thres
        self.ref_token_idx = ref_token_idx
        self.cur_token_idx = cur_token_idx
        self.box = box
        self.self_attns = []
        self.img_save_path=None
        self.cross_attns = []
        self.cross_attns_mask = cross_attns_mask
        self.vary=0
        self.vary2=0
        self.cur_att_layer=0
        self.bool_foward=False
        self.batch_size=batch_size
        self.num_self_replace = [int(0), int(2)] 
        if box is not None:
            mask_tensor = torch.zeros((512, 512), dtype=torch.float32)
            x, y, x2, y2 = map(int, box)
            mask_tensor[y:y2, x:x2] = 1
            self.box_mask_16 = F.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0), size=(16, 16), mode='bilinear', align_corners=False)
            
            self.box_mask_32 = F.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False)
            self.box_mask_16 =self.box_mask_16.squeeze(0).unsqueeze(-1).reshape(1,256,1).to(device)
            self.box_mask_32 =self.box_mask_32.squeeze(0).unsqueeze(-1).reshape(1,1024,1).to(device)
        if cross_attns_mask is not None:
            self.cross_attns_mask_16 = F.interpolate(cross_attns_mask.unsqueeze(0), size=(16, 16), mode='bilinear', align_corners=False)
            self.cross_attns_mask_32 = F.interpolate(cross_attns_mask.unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False)
        self.self_attns_mask = None
        self.last_idx=last_idx
        self.mask_save_dir = mask_save_dir
        if self.mask_save_dir is not None:
            os.makedirs(self.mask_save_dir, exist_ok=True)
    @torch.no_grad()
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:#1024 256
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)#*self.vary
            return attn_base
        else:
            return att_replace
    @torch.no_grad()
    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if self.bool_foward:
            if  is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]) :
                h = attn.shape[0] // (self.batch_size)
                attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
                attn_base, attn_repalce = attn[0], attn[1:]
                if is_cross:
                    pass
                else:
                    attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
                attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out