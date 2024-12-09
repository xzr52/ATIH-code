import sys
#sys.path.append('/opt/data/private/xzr_com/creative_object/seg/GroundingDINO')
import os
from step import step1, step2
from src.eunms import Model_Type, Scheduler_Type
from src.utils.enums_utils import get_pipes
from src.config import RunConfig
from src.get_sim import ImageSimilarityModel
from PIL import Image
import torch
import numpy as np
import os
import gradio as gr
from PIL import Image
import numpy as np
from groundingdino.util.inference import Model
import json
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading pipeline components(step1)...........")
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/Ground_DINO_config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "ckpts/groundingdino_swint_ogc.pth"
EFFICIENT_SAM_CHECHPOINT_PATH = "ckpts/efficientsam_s_gpu.jit"
# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
efficientsam = torch.jit.load(EFFICIENT_SAM_CHECHPOINT_PATH)
print("(step1)...........")

cfg = RunConfig(model_type = Model_Type.SDXL_Turbo, scheduler_type = Scheduler_Type.EULER)
with torch.no_grad():
    device2=torch.device("cuda:1")
    device=torch.device("cuda:0")
    pipe_inversion, pipe_inference = get_pipes(Model_Type.SDXL_Turbo, Scheduler_Type.EULER, device=device)
    ImageSimmodel = ImageSimilarityModel(device2)
print("(step2)...........")

os.makedirs("outputs/step1", exist_ok=True)
os.makedirs("outputs/step2", exist_ok=True)
os.makedirs("cache", exist_ok=True)

def run_step1(image, image_name):
    save_dir = f"outputs/step1/{os.path.splitext(image_name)[0]}"
    os.makedirs(save_dir, exist_ok=True)
    input_image_path = os.path.join(save_dir, image_name)
    image.save(input_image_path)
    step1(input_image_path, grounding_dino_model,save_dir,image_name) 
    processed_image_path = os.path.join(save_dir, image_name)
    xyxy_npy_path = os.path.join(save_dir, "xyxy.npy")
    return processed_image_path, xyxy_npy_path

def run_step2(prompt, processed_image_path, xyxy_npy_path, certain_step, certain_vary, seed):
    step2(prompt, processed_image_path, xyxy_npy_path, pipe_inversion, pipe_inference, ImageSimmodel, certain_step, certain_vary, seed)
    final_image_path = "outputs/step2/result.png"  # 
    data_path = "outputs/step2/params.npy"  # 
    final_image = Image.open(final_image_path)

    data = np.load(data_path, allow_pickle=True).item()  # 
    return final_image, data

def main(image, prompt, certain_step, certain_vary, seed):
    image_name = os.path.basename(image.name) if hasattr(image, 'name') else "input_image.png"
    processed_image_path, xyxy_npy_path = run_step1(image, image_name)
    final_image, data = run_step2(prompt, processed_image_path, xyxy_npy_path, certain_step, certain_vary, seed)

    formatted_html = f"""
    <table style='width:100%; border: 1px solid #ddd; border-collapse: collapse;'>
        <tr><th style='border: 1px solid #ddd; padding: 8px;'>Key</th><th style='border: 1px solid #ddd; padding: 8px;'>Value</th></tr>
        {"".join([f"<tr><td style='border: 1px solid #ddd; padding: 8px;'>{k}</td><td style='border: 1px solid #ddd; padding: 8px; color: blue;'>{v}</td></tr>" for k, v in data.items()])}
    </table>
    """
    
    return final_image, formatted_html

examples = [
    ["examples/rabbit.png", "peacock", 0, 0, 42],
    ["examples/glass2jar.png", "European fire salamander", 0, 0, 42],
    ["examples/wolf.png", "bighorn", 0, 0, 42],
    ["examples/car.png", "airship", 0, 0, 42],
    ["examples/naruto.png", "cougar", 0, 0, 42],
    ["examples/owl.png", "strawberry", 0, 0, 42],
]

def format_json_output(json_data):
    return json.dumps(json_data, indent=4, sort_keys=True, ensure_ascii=False)

iface = gr.Interface(
    fn=main,
    inputs=[
        gr.Image(type="pil", label="Upload Image"), 
        gr.Textbox(lines=2, placeholder="Enter your object prompt here...", label="Object Prompt"),
        gr.Slider(minimum=0, maximum=4, step=1, label="Certain Step (The larger the step, the closer it resembles the original image.When set to 0, it will automatically search for the appropriate value.)"),
        gr.Number(label="Certain Vary (The smaller the Vary, the closer it resembles the original image.When set to 0, it will automatically search for the appropriate value.)", minimum=0),  # 
        gr.Number(label="Seed (Different seeds have different results)", precision=0) 
    ],
    outputs=[
        gr.Image(type="pil", label="Processed Image"),
        gr.HTML(label="Formatted JSON")
    ],
    title="Novel Object Synthesis via Adaptive Text-Image Harmony",
    description="Upload an image and provide a object prompt to run (It would be best to upload an image of an animal, if possible.).",
    examples=examples
)

if __name__ == "__main__":
    # iface.launch(share=True)
    iface.launch(share=True)