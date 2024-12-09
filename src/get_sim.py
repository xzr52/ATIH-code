

from torchvision import transforms
import numpy as np
import pandas as pd
import os
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel
import open_clip
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
dinov2_transforms_origin = transforms.Compose([
    #transforms.ToTensor(),
    transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size=(224, 224)),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229, 0.224, 0.225))
])
dinov2_transforms = transforms.Compose([
    transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size=(224, 224)),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229, 0.224, 0.225))
])
cliptransforms_origin = transforms.Compose([
    #transforms.ToTensor(),
    transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size=(224, 224)),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])
cliptransforms = torch.nn.Sequential(
        transforms.Resize(size=224, interpolation=InterpolationMode.BICUBIC,max_size=None, antialias=None),
        transforms.CenterCrop(size=(224,224)),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)))
to_tensor = transforms.ToTensor()
class ImageSimilarityModel:
    def __init__(self,device):
        with torch.no_grad():
            self.dino_processor = AutoImageProcessor.from_pretrained('/root/.cache/huggingface/hub/models--facebook--dinov2-base/snapshots/369b1309d0ea32e09fdc25ee029917e2e5a2db8f')
            self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='/root/.cache/huggingface/hub/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/snapshots/8c7a3583335de4dba1b07182dbf81c75137ce67b/open_clip_pytorch_model.bin',device=device)
            self.dinov2_model = AutoModel.from_pretrained('/root/.cache/huggingface/hub/models--facebook--dinov2-base/snapshots/369b1309d0ea32e09fdc25ee029917e2e5a2db8f').to(device)
            # self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            # self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-bigG-14',device=device)
            # self.dinov2_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
            self.tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
            self.device=device
            self.clip_model=self.model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad = False
    def get_origin_image_tensor(self,origin_img,loaded_detections,save_path):
        rgb_image_tensor= to_tensor(origin_img)
        if loaded_detections is not None:
            self.x, self.y, self.x2, self.y2 = map(int, loaded_detections)
            rgb_image_tensor = rgb_image_tensor[:, self.y:self.y2, self.x:self.x2]

        origin_image = cliptransforms_origin(rgb_image_tensor).unsqueeze(0)#0.1005 
        clip_origin_image_features = self.clip_model.encode_image(origin_image.to(self.device))
        clip_origin_image_features /= clip_origin_image_features.norm(dim=-1, keepdim=True)
        dinov2_image=dinov2_transforms_origin(rgb_image_tensor)
        outputs1 = self.dinov2_model(dinov2_image.unsqueeze(0).to(self.device))#
        dinov2_origin_image_features = outputs1.last_hidden_state
        self.dinov2_origin_image_features = dinov2_origin_image_features.mean(dim=1)
    def get_text_features(self,target_prompt):
        text = self.tokenizer(f'{target_prompt}')
        self.text_features = self.clip_model.encode_text(text.to(self.device))    
    def get_image_text_sim(self, target_image):
        target_image_tensor = to_tensor(target_image) 

        target_image_tensor = target_image_tensor[:, self.y:self.y2, self.x:self.x2]

        clip_image = cliptransforms(target_image_tensor).unsqueeze(0)

        target_dinov2_image = dinov2_transforms(target_image_tensor)
        
        dinov2_image_feature = self.dinov2_model(target_dinov2_image.unsqueeze(0).to(self.device))
        dinov2_target_image_features = dinov2_image_feature.last_hidden_state.mean(dim=1)

        clip_image_features = self.clip_model.encode_image(clip_image.to(self.device))
        clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True) 

        dino_image_sim = F.cosine_similarity(self.dinov2_origin_image_features, dinov2_target_image_features, dim=-1).item()

        clip_text_sim = F.cosine_similarity(clip_image_features, self.text_features, dim=-1).item()

        return dino_image_sim, clip_text_sim
    def get_image_tensor_text_image_sim(self, target_image):
        """
        Calculate the similarity between the target image and the original image using DINOv2 and CLIP models.
        - DINOv2 computes the image similarity.
        - CLIP computes the similarity between the image and text.
        """
        target_image_tensor = (target_image / 2 + 0.5).clamp(0, 1)  # Normalize to [0, 1]

        target_image_tensor = target_image_tensor[:, self.y:self.y2, self.x:self.x2]

        clip_image = cliptransforms(target_image_tensor).unsqueeze(0)  # Add batch dimension

        target_dinov2_image = dinov2_transforms(target_image_tensor)

        dinov2_image_feature = self.dinov2_model(target_dinov2_image.unsqueeze(0).to(self.device))
        dinov2_target_image_features = dinov2_image_feature.last_hidden_state.mean(dim=1)  

        clip_image_features = self.clip_model.encode_image(clip_image.to(self.device))
        clip_image_features = clip_image_features / clip_image_features.norm(dim=-1, keepdim=True)  # Normalize

        dino_image_sim = F.cosine_similarity(self.dinov2_origin_image_features, dinov2_target_image_features, dim=-1).item()

        clip_text_sim = F.cosine_similarity(clip_image_features, self.text_features, dim=-1).item()

        return dino_image_sim, clip_text_sim
    def get_image_tensor_text_image_sim_no_box(self, target_image):
        """
        Calculate the similarity between the target image and the original image using DINOv2 and CLIP models,
        but without using a bounding box (i.e., considering the entire image).
        - DINOv2 computes the image similarity.
        - CLIP computes the similarity between the image and text.
        """
        target_image_tensor = (target_image / 2 + 0.5).clamp(0, 1)  # Normalize to [0, 1]

        clip_image = cliptransforms(target_image_tensor).unsqueeze(0)  # Add batch dimension

        target_dinov2_image = dinov2_transforms(target_image_tensor)

        dinov2_image_feature = self.dinov2_model(target_dinov2_image.unsqueeze(0).to(self.device))

        dinov2_target_image_features = dinov2_image_feature.last_hidden_state.mean(dim=1)

        clip_image_features = self.clip_model.encode_image(clip_image.to(self.device))
        clip_image_features = clip_image_features / clip_image_features.norm(dim=-1, keepdim=True)

        dino_image_sim = F.cosine_similarity(self.dinov2_origin_image_features, dinov2_target_image_features, dim=-1).item()

        clip_text_sim = F.cosine_similarity(clip_image_features, self.text_features, dim=-1).item()

        return dino_image_sim, clip_text_sim

                