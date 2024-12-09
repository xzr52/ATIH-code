from src.eunms import Model_Type, Scheduler_Type
from src.utils.enums_utils import get_pipes
from src.config import RunConfig
from src.get_sim import ImageSimilarityModel
from PIL import Image
import torch
from src.utils.enums_utils import model_type_to_size, is_stochastic
from diffusers.utils.torch_utils import randn_tensor
from masactrl.masactrl2 import MutualSelfAttentionControlMaskAuto
from masactrl.masactrl_utils import regiter_attention_editor_diffusers
import numpy as np
import cv2
import supervision as sv
import torchvision
import os
from groundingdino.util.inference import Model
import argparse
def step1(image, classes, save_path):
    box_threshold = 0.25  # Confidence threshold for detecting boxes
    text_threshold = 0.25  # Confidence threshold for detecting text
    nms_threshold = 0.8  # Threshold for non-maximum suppression (NMS)

    # Perform object detection and filter with specified thresholds
    detections = grounding_dino_model.predict_with_classes(image=image, classes=classes, box_threshold=box_threshold, text_threshold=text_threshold)
    
    # Apply NMS to remove redundant boxes
    nms_idx = torchvision.ops.nms(torch.from_numpy(detections.xyxy), torch.from_numpy(detections.confidence), nms_threshold).numpy().tolist()
    detections.xyxy = detections.xyxy[nms_idx]

    # Save the first detected bounding box to a file
    if len(detections.xyxy) > 0:
        x, y, x2, y2 = map(float, detections.xyxy[0])  # Get the coordinates of the first bounding box
        np.save(os.path.join(save_path, "xyxy.npy"), np.array([x, y, x2, y2]))  # Save the coordinates to a .npy file


def create_noise_list(model_type, length, generator=None):
    img_size = model_type_to_size(model_type)  # Get the image size based on the model type
    latents_size = (1, 4, img_size[0] // 8, img_size[1] // 8)  # Define the latent tensor size
    return [randn_tensor(latents_size, dtype=torch.float16, device=device, generator=generator) for _ in range(length)]  # Return a list of random noise tensors
def step2(prompt, image_path, box_path, save_path,pipe_inversion,pipe_inference):
    model_type = Model_Type.SDXL_Turbo
    scheduler_type = Scheduler_Type.EULER

    cfg = RunConfig(model_type = model_type,
                                scheduler_type = scheduler_type)

    box_path = box_path
    attention_image_sim_max=0.85
    attention_image_sim_min=0.45
    K=2.3
    def performance_score(image_sim, text_sim):
        score = image_sim+text_sim*K-abs(image_sim - K*text_sim)#
        return score#image_sim - text_penalty
    img=Image.open(image_path)
    inverse_prompt=f" "#{object_name}
    
    latents=None
    if os.path.exists(box_path):
            loaded_detections = np.load(box_path)
            x, y, x2, y2 = map(int, loaded_detections)
            #rgb_image_tensor = rgb_image_tensor[:, y:y2, x:x2]
    else:
        loaded_detections=None
    
    editor = MutualSelfAttentionControlMaskAuto(1, 44,model_type="SDXL",box=loaded_detections)#,cross_attns_mask=mask_image_tensor,box=loaded_detections)       
    regiter_attention_editor_diffusers(pipe_inference, editor)
    print("Inverting...")
    editor.bool_foward=False
    res = pipe_inversion(prompt = inverse_prompt,
                    num_inversion_steps = cfg.num_inversion_steps,
                    num_inference_steps = cfg.num_inference_steps,
                    generator = generator,
                    image = img,
                    guidance_scale = cfg.guidance_scale,
                    strength = cfg.inversion_max_step,
                    denoising_start = 1.0-cfg.inversion_max_step,
                    num_renoise_steps = cfg.num_renoise_steps)
    with torch.no_grad():
        ImageSimmodel.get_origin_image_tensor(origin_img=img,loaded_detections=loaded_detections,save_path=save_path)
        ImageSimmodel.get_text_features(prompt)
            
        originlatents = res[0][0]

        editor.bool_foward=True
        together_prompt = [inverse_prompt, prompt,prompt]
        latents = originlatents.expand(len(together_prompt), -1, -1, -1)
        ################
        guidance_scale = 0.0
        editor.batch_size=len(together_prompt)
        editor.cur_step=0
        print("Generating...")
        all_iter_images = []
        ###########attention_step_select##########
        iter_bool=True
        current_iteration=0
        editor.batch_size=len(together_prompt)
        editor.num_self_replace[1]=2
        while iter_bool and current_iteration < 4:
            current_iteration += 1  
            editor.cur_step = 0
            editor.vary = 1.0
            editor.vary2 = 1.0

            image1 = pipe_inference(prompt = together_prompt,
                                    num_inference_steps = cfg.num_inference_steps,
                                    negative_prompt = '',
                                    image = latents,  # 0.2837 -11.2969
                                    strength = cfg.inversion_max_step,
                                    denoising_start = 1.0 - cfg.inversion_max_step,
                                    guidance_scale = guidance_scale,
                                    return_dict=False)#.images
            if loaded_detections is None:
                dino_image_sim1, clip_text_sim1 = ImageSimmodel.get_image_tensor_text_image_sim_no_box(image1[1][1])
                dino_image_sim2, clip_text_sim2 = ImageSimmodel.get_image_tensor_text_image_sim_no_box(image1[1][2])
            else:
                dino_image_sim1, clip_text_sim1 = ImageSimmodel.get_image_tensor_text_image_sim(image1[1][1])
                dino_image_sim2, clip_text_sim2 = ImageSimmodel.get_image_tensor_text_image_sim(image1[1][2])
            if dino_image_sim1 < attention_image_sim_min: 
                if editor.num_self_replace[1] > 3:
                    iter_bool = False
                else:
                    editor.num_self_replace[1] += 1
            if dino_image_sim2 >= attention_image_sim_max:
                if editor.num_self_replace[1] == 0:
                    iter_bool = False
                else:
                    editor.num_self_replace[1] -= 1
            else:
                iter_bool = False

        #####################################
        ################ get sim##########
        editor.batch_size=len(together_prompt)
        high = 1.5
        low = 0.0
        max_score = float('-inf')
        best_alpha = None
        tolerance=0.01
        phi = 1.618
        while high - low > tolerance:
            editor.cur_step=0
            mid1 = round(low + (high - low) / phi, 3)
            mid2 = round(high - (high - low) / phi, 3)
            
            # Set the varying parameters for the editor object
            editor.vary = mid1
            editor.vary2 = mid2
            
            # Execute the inference pipeline
            images = pipe_inference(
                prompt=together_prompt,
                num_inference_steps=cfg.num_inference_steps,
                negative_prompt='',
                image=latents,
                strength=cfg.inversion_max_step,
                denoising_start=1.0 - cfg.inversion_max_step,
                guidance_scale=guidance_scale,
                return_dict=False
            )#.images
            if loaded_detections is None:
                dino_image_sim1, clip_text_sim1 = ImageSimmodel.get_image_tensor_text_image_sim_no_box(images[1][1])
                dino_image_sim2, clip_text_sim2 = ImageSimmodel.get_image_tensor_text_image_sim_no_box(images[1][2])
            else:
                dino_image_sim1, clip_text_sim1 = ImageSimmodel.get_image_tensor_text_image_sim(images[1][1])
                dino_image_sim2, clip_text_sim2 = ImageSimmodel.get_image_tensor_text_image_sim(images[1][2])
            
            score1 = performance_score(dino_image_sim1, clip_text_sim1)
            score2 = performance_score(dino_image_sim2, clip_text_sim2)
            
            # Decide which interval to use based on the scores
            if score1 >= score2:
                low = mid2
                if score1 > max_score:

                    taget_image=images[0][1]
                    max_score = round(score1, 3)
                    best_alpha = mid1
            else:
                high = mid1
                
                if score2 > max_score:
                    final_clip_text_sim=clip_text_sim2
                    final_dino_sim= dino_image_sim2,
                    taget_image=images[0][2]
                    max_score = round(score2, 3)
                    best_alpha = mid2

        if best_alpha is not None and max_score is not None:
            maxfilename=f"result.png"

            save_path_max=os.path.join(save_path, maxfilename)
            taget_image.save(save_path_max)

            # 保存参数
            params = {
                "vary": round(best_alpha, 3),
                "astep": editor.num_self_replace[1],
                "score": round(max_score, 3)
            }
            params_filename = f"params.npy"
            save_path_params = os.path.join(save_path, params_filename)
            np.save(save_path_params, params)

    return taget_image



def run_step1(image, save_path,image_path,CLASSES):
    image_name=os.path.basename(image_path)
    save_dir = os.path.join(save_path,'step1',os.path.splitext(image_name)[0])
    os.makedirs(save_dir, exist_ok=True)

    processed_image_path = os.path.join(save_dir, image_name)
    cv2.imwrite(processed_image_path,image)

    step1(image,CLASSES,save_dir)
    xyxy_npy_path = os.path.join(save_dir, "xyxy.npy")
    
    print(f"Processed image path: {processed_image_path}")
    print(f"Bounding box coordinates path: {xyxy_npy_path}")
    
    return processed_image_path, xyxy_npy_path

def run_step2(prompt,save_path, image_path, box_path,pipe_inversion,pipe_inference):

    image_name=os.path.basename(image_path)
    save_dir = os.path.join(save_path,'step2',os.path.splitext(image_name)[0])
    os.makedirs(save_dir, exist_ok=True)

    final_image = step2(prompt, image_path, box_path, save_dir,pipe_inversion,pipe_inference)

    return final_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Inversion and Generation")
    
    # Define command-line arguments
    parser.add_argument('--image_path', type=str,default='examples/rabbit.png', help="Path to the input image")
    parser.add_argument('--seg_image_name', type=str, default='main object', help="Segmented image name or description")
    parser.add_argument('--target_prompt', type=str, default='peacock', help="Text prompt describing the target object")
    parser.add_argument('--save_path', type=str, default='outputs', help="Directory where the result will be saved")

    # Parse the arguments
    args = parser.parse_args()
    image = cv2.imread(args.image_path)
    resized_image = cv2.resize(image, (512, 512))
    CLASSES = [f"{args.seg_image_name}"]#[f"{image_name}"]
    
    ###########load#########################
    # GroundingDINO config and checkpoint 
    print("Loading pipeline components(step1)...........")
    GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/Ground_DINO_config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "ckpts/groundingdino_swint_ogc.pth"
    EFFICIENT_SAM_CHECHPOINT_PATH = "ckpts/efficientsam_s_gpu.jit"
    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
    efficientsam = torch.jit.load(EFFICIENT_SAM_CHECHPOINT_PATH)#
    model_type = Model_Type.SDXL_Turbo
    scheduler_type = Scheduler_Type.EULER
    cfg = RunConfig(model_type = model_type,
                                scheduler_type = scheduler_type)
    with torch.no_grad():
        device2=torch.device("cuda:1")
        device=torch.device("cuda:0")
        pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)
        ImageSimmodel = ImageSimilarityModel(device2)
        generator = torch.Generator().manual_seed(cfg.seed)
        print("Loading pipeline components(step2)...........")
        noise = create_noise_list(cfg.model_type, cfg.num_inversion_steps, generator=generator)
        pipe_inversion.scheduler.set_noise_list(noise)
        pipe_inference.scheduler.set_noise_list(noise)
        pipe_inversion.scheduler.set_iteration(125)
        pipe_inference.scheduler.set_iteration(125)
        pipe_inversion.cfg = cfg
        pipe_inference.cfg = cfg
    #########################
    
    processed_image_path, xyxy_npy_path = run_step1(resized_image, args.save_path,args.image_path,CLASSES)
    final_image = run_step2(args.target_prompt,args.save_path, processed_image_path, xyxy_npy_path,pipe_inversion,pipe_inference)
