from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from PIL import Image
import torch
import os
import numpy as np
from io import BytesIO
import zipfile
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from diffusers import DDPMScheduler, AutoencoderKL
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPImageProcessor
from RealESRGAN import RealESRGAN
from typing import List
import cv2
import base64
from imwatermark import WatermarkEncoder
import time
from pathlib import Path
from fastapi.staticfiles import StaticFiles


deviceforesrgan = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models
model2 = RealESRGAN(deviceforesrgan, scale=2)
model2.load_weights('weights/RealESRGAN_x2.pth', download=True)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class TryonRequest(BaseModel):
    is_checked_crop: bool = False
    clothes_type: str

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i, j]:
                mask[i, j] = 1
    mask = (mask * 255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask

base_path = 'yisol/IDM-VTON'

unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet", torch_dtype=torch.float16).to(device)
tokenizer_one = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer", use_fast=False)
tokenizer_two = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer_2", use_fast=False)
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(base_path, subfolder="text_encoder_2", torch_dtype=torch.float16).to(device)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(base_path, subfolder="image_encoder", torch_dtype=torch.float16).to(device)
vae = AutoencoderKL.from_pretrained(base_path, subfolder="vae", torch_dtype=torch.float16).to(device)
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(base_path, subfolder="unet_encoder", torch_dtype=torch.float16).to(device)

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

pipe = StableDiffusionXLInpaintPipeline.from_pretrained(base_path,
                                     unet=unet,
                                     vae=vae,
                                     feature_extractor=CLIPImageProcessor(),
                                     text_encoder=text_encoder_one,
                                     text_encoder_2=text_encoder_two,
                                     tokenizer=tokenizer_one,
                                     tokenizer_2=tokenizer_two,
                                     scheduler=noise_scheduler,
                                     image_encoder=image_encoder,
                                     torch_dtype=torch.float16).to(device)
pipe.unet_encoder = UNet_Encoder.to(device)
tensor_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

categories = {
    "upper_body": ["shirt", "sweater", "t-shirt", "blouse", "hoodie", "vest", "jacket"],
    "dresses": ["dress", "jumpsuit", "overalls", "suit", "coat", "bikini",],
    "lower_body": ["pants", "shorts", "jeans", "leggings", "swimsuit",  "long-skirt", "short-skirt"],
}

def determine_category(option: str) -> str:
    for category, values in categories.items():
        if option in values:
            return category
    return "unknown"

@app.post("/tryon_steps")
async def tryon_steps(human: UploadFile = File(...), garment: UploadFile = File(...), 
                      is_checked_crop: bool = Form(...), clothes_type: str = Form(...),
                      seed: str = Form(None)):

    request = TryonRequest(
        is_checked_crop=is_checked_crop,
        clothes_type=clothes_type
    )

    is_checked = True
    
    body_part = determine_category(clothes_type)
    
    human_img = Image.open(BytesIO(await human.read())).convert("RGB")
    garm_img = Image.open(BytesIO(await garment.read())).convert("RGB").resize((768, 1024))

    human_img_path = "human_image.png"
    garm_img_path = "garment_image.png"
    human_img.save(human_img_path)
    garm_img.save(garm_img_path)
    
    if request.is_checked_crop:
        width, height = human_img.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768, 1024))
    else:
        human_img = human_img.resize((768, 1024))
    
    if is_checked:
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        mask, mask_gray = get_mask_location('hd', body_part, model_parse, keypoints)
        mask = mask.resize((768, 1024))
    else:
        mask = pil_to_binary_mask(human_img.resize((768, 1024)))
        mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transform(human_img)
        mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)
    
    human_img_path = "human_image2.png"
    garm_img_path = "garment_image2.png"
    human_img.save(human_img_path)
    garm_img.save(garm_img_path)
    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
    
    args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    pose_img = args.func(args, human_img_arg)[:,:,::-1]
    pose_img = Image.fromarray(pose_img).resize((768, 1024))
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            prompt = "model is wearing " + request.clothes_type
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, wrong hands, wrong legs, bad foots, bad hands, bad legs, bas foots"
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt
            )
            
            prompt_c = "a photo of " + request.clothes_type
            if not isinstance(prompt_c, list):
                prompt_c = [prompt_c] * 1
            if not isinstance(negative_prompt, list):
                negative_prompt = [negative_prompt] * 1
            prompt_embeds_c = pipe.encode_prompt(
                prompt_c,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=negative_prompt
            )[0]
            
            pose_img_tensor = tensor_transform(pose_img).unsqueeze(0).to(device, torch.float16)
            garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(device, torch.float16)
            generator = torch.Generator(device).manual_seed(int(seed)) if seed is not None else None
            
            image_dir = "generated_images"
            os.makedirs(image_dir, exist_ok=True)
            image_paths = []

            for step in range(15, 100, 12):
                images = pipe(
                    prompt_embeds=prompt_embeds.to(device, torch.float16),
                    negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                    num_inference_steps=step,
                    generator=generator,
                    strength=1.0,
                    pose_img=pose_img_tensor,
                    text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                    cloth=garm_tensor,
                    mask_image=mask,
                    image=human_img,
                    height=1024,
                    width=768,
                    ip_adapter_image=garm_img,
                    guidance_scale=2.0,
                )[0]
                
                output_image = images[0]
                output_path = os.path.join(image_dir, f"output_image_step_{step}.png")
                output_image.save(output_path)
                watermark_start = time.time()
                watermark_text = 'beko'
                output_path_marked = os.path.join(image_dir, f"output_image_marked_{step}.png")
                wm_b16 = base64.b16encode(watermark_text.encode('utf-8')).decode('utf-8')
                encoder = WatermarkEncoder()
                encoder.set_watermark('b16', wm_b16)
                bgr = cv2.imread(output_path)
                bgr_encoded = encoder.encode(bgr, 'dwtDctSvd')
                cv2.imwrite(output_path_marked, bgr_encoded)
                watermark_end = time.time()
                print(f"Watermarking finished in {watermark_end - watermark_start} seconds")
                image_paths.append(output_path_marked)
    
    zip_filename = "generated_images.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for img_path in image_paths:
            zipf.write(img_path, os.path.basename(img_path))
    
    return FileResponse(zip_filename, media_type="application/zip", filename=zip_filename)


@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/indexson")
def read_root():
    return FileResponse("static/indexSON2.html")

# Run the app with `uvicorn`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000,)
