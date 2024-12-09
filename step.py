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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def step1(image, grounding_dino_model,save_dir,save_image_name):
    image_name='main object'#输入

    CLASSES = [f"{image_name}"]#[f"{image_name}"]
    BOX_THRESHOLD = 0.25
    TEXT_THRESHOLD = 0.25
    NMS_THRESHOLD = 0.8

    image = cv2.imread(image)
    resized_image = cv2.resize(image, (512, 512))
    #image.resize((512, 512))
    cv2.imwrite(save_dir+f'/{save_image_name}.png', resized_image)
    image=resized_image
    # detect objectsc
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=BOX_THRESHOLD
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _ ,_
        in detections]
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections) #  labels=labels)
    print(f"Before NMS: {len(detections.xyxy)} boxes")
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    save_path_final=os.path.join(save_dir, "xyxy.npy") 
    if len(detections.xyxy)!= 0:
        x, y, x2, y2 = map(float, detections.xyxy[0])
        #x, y, x2, y2 = (x - 2 if x > 3 else x, y - 2 if y > 3 else y, x2 + 2 if x2 < 510 else x2, y2 + 2 if y2 < 510 else y2)
        np.save(save_path_final, np.array([x, y, x2, y2]))
        #loaded_detections = np.load(save_path_final)

def step2(prompt, image_path, box_path, pipe_inversion, pipe_inference, ImageSimmodel, certain_step, certain_vary, seed):
    model_type = Model_Type.SDXL_Turbo
    scheduler_type = Scheduler_Type.EULER
    def create_noise_list(model_type, length, generator=None):
        img_size = model_type_to_size(model_type)
        VQAE_SCALE = 8
        latents_size = (1, 4, img_size[0] // VQAE_SCALE, img_size[1] // VQAE_SCALE)
        return [randn_tensor(latents_size, dtype=torch.float16, device=torch.device("cuda:0"), generator=generator) for i in range(length)]
    results_list=[]
    cfg = RunConfig(model_type = model_type,
                                scheduler_type = scheduler_type)

    save_path_origin = 'outputs/step2'
    image_path=image_path
    target_prompt=prompt
    box_path=box_path
    certain_step=certain_step
    certain_vary=certain_vary
    cfg.seed=seed
    if certain_step>4:
        certain_step=4
        print("certain_step should be less than 4 set certain_step 4")
    if certain_step<0:
        certain_step=0
        print("certain_step should be greater than 0 set certain_step 0")
    if certain_vary<0:
        certain_vary=0
        print("certain_vary should be greater than 0 set certain_vary 0")
    attention_image_sim_max=0.85
    attention_image_sim_min=0.45
    K=2.3
    def performance_score(image_sim, text_sim):
        score = image_sim+text_sim*K-abs(image_sim - K*text_sim)#
        return score#image_sim - text_penalty
    save_image_name = os.path.basename(image_path)
    save_image_name=save_image_name.split('.')[0]
    object_name=save_image_name.replace('_',' ')
    object_name=object_name.replace('1','')
    object_name=object_name.replace('2',' ')
    # save_path=save_path_origin+f"/{save_image_name}"#/{imagenet_class}"
    save_path=save_path_origin
    os.makedirs(save_path,exist_ok=True)
    img2=Image.open(image_path)
    img2=img2.resize((512,512))
    inverse_prompt=f" "#{object_name}
    input_image = img2
    generator = torch.Generator().manual_seed(cfg.seed)

    if os.path.exists(box_path):
            loaded_detections = np.load(box_path)
            x, y, x2, y2 = map(int, loaded_detections)
            #rgb_image_tensor = rgb_image_tensor[:, y:y2, x:x2]
    else:
        loaded_detections=None
    latents=None
    if is_stochastic(cfg.scheduler_type):
        if latents is None:
            noise = create_noise_list(cfg.model_type, cfg.num_inversion_steps, generator=generator)
        pipe_inversion.scheduler.set_noise_list(noise)
        pipe_inference.scheduler.set_noise_list(noise)
    pipe_inversion.scheduler.set_iteration(125)
    pipe_inference.scheduler.set_iteration(125)
    pipe_inversion.cfg = cfg
    pipe_inference.cfg = cfg
    all_latents = None
    with torch.no_grad():
        editor = MutualSelfAttentionControlMaskAuto(1, 44,model_type="SDXL",box=loaded_detections)#,cross_attns_mask=mask_image_tensor,box=loaded_detections)       
        regiter_attention_editor_diffusers(pipe_inference, editor)
    print("Inverting...")
    editor.bool_foward=False
    res = pipe_inversion(prompt = inverse_prompt,
                    num_inversion_steps = cfg.num_inversion_steps,
                    num_inference_steps = cfg.num_inference_steps,
                    generator = generator,
                    image = img2,
                    guidance_scale = cfg.guidance_scale,
                    strength = cfg.inversion_max_step,
                    denoising_start = 1.0-cfg.inversion_max_step,
                    num_renoise_steps = cfg.num_renoise_steps)

    ImageSimmodel.get_origin_image_tensor(origin_img=img2,loaded_detections=loaded_detections,save_path=save_path)
    ImageSimmodel.get_text_features(target_prompt)
    with torch.no_grad():
        together_prompt = [inverse_prompt, target_prompt,target_prompt]
        guidance_scale = 0.0
        editor.batch_size=len(together_prompt)
        originlatents = res[0][0]
        latents = originlatents.expand(len(together_prompt), -1, -1, -1)
        if certain_step == 0:
            #{object_name} fuse with
            editor.bool_foward=True
            ################
            editor.cur_step=0
            print("Generating...")
            all_iter_images = []
            ###########attention_step_select##########
            iter_bool=True
            current_iteration=0
            editor.num_self_replace[1]=2
            while iter_bool and current_iteration < 4:
                current_iteration += 1  # 每次循环开始时增加迭代计数器
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
                if dino_image_sim2 >= attention_image_sim_max:#or clip_text_sim2 < 0.24:
                    if editor.num_self_replace[1] == 0:
                        iter_bool = False
                    else:
                        editor.num_self_replace[1] -= 1
                else:
                    iter_bool = False
        else:
            editor.num_self_replace[1] =certain_step
        #####################################
        ################ get sim##########
        if certain_vary == 0:
            high = 1.5
            low = 0.0
            max_score = float('-inf')
            best_alpha = None
            tolerance=0.01
            phi = 1.618
            iter=0
            editor.bool_foward=True
            while high - low > tolerance:
                iter+=1
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
                print(f'mid1:{mid1}')
                print(f'mid2:{mid2}')
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
                    certain_step=0
                    certain_vary=0
        else:
            editor.bool_foward=True
            editor.cur_step=0
            editor.vary = certain_vary
            editor.vary2 = certain_vary
            images = pipe_inference(
                    prompt=together_prompt,
                    num_inference_steps=cfg.num_inference_steps,
                    negative_prompt='',
                    image=latents,
                    strength=cfg.inversion_max_step,
                    denoising_start=1.0 - cfg.inversion_max_step,
                    guidance_scale=guidance_scale,
                    return_dict=False
                )
            maxfilename=f"result.png"
            # 保存参数
            params = {
                "vary": round(best_alpha, 3),
                "astep": editor.num_self_replace[1],
                "score": round(max_score, 3)
            }
            params_filename = f"params.npy"
            save_path_params = os.path.join(save_path, params_filename)
            np.save(save_path_params, params)

            save_path_max=os.path.join(save_path, maxfilename)
            images[0][1].save(save_path_max)
            certain_step=0
            certain_vary=0


