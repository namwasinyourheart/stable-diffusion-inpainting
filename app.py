import gradio as gr
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

device = "cuda"
sam_checkpoint = "weights/"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)

pipe = pipe.to(device)

with gr.Blocks() as demo:
    with gr.Row():
        input_img = gr.Image(label="Input")
        mask_img = gr.Image(label="Mas")
        output_img = gr.Image(label="Output")

    with gr.Block():
        prompt_text = gr.Textbox(lines=1, label="Prompt")
    
    with gr.Row():
        submit = gr.Button("Submit")

    def generate_mask(image, evt:gr.SelectData):
        pass

    def inpaint(image, mask, prompt):
        pass

    
