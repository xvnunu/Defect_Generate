import torch
from diffusers.utils import load_image, check_min_version
from diffusers import ControlNetModel
from pipeline import StableDiffusionControlNetInpaintImg2ImgPipeline
from PIL import Image
import numpy as np
import os

# device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionControlNetInpaintImg2ImgPipeline.from_pretrained(
    "pretrained_model/stable-diffusion-v1-5/stable-diffusion-inpainting",
    controlnet=ControlNetModel.from_pretrained(
        "output/controlnet-inpainting-segment-plusplus", torch_dtype=torch.float16
    ),
    safety_checker=None,
    torch_dtype=torch.float16
)
# pipe.text_encoder.to(torch.float16)
# pipe.controlnet.to(torch.float16)
# pipe.to("cuda")

# remove following line if xformers is not installed or when using Torch 2.0.
pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()

width = 512
height = 512
size = (width, height)

filename = "defect (1210).jpg"
scale_txt_path = "croped_defect/scales.txt"

ori_img = load_image(
    "diffuser/images/defect (1023).jpg"
).convert("RGB")

crop_mask = load_image(
    "croped_defect/conditioning_images/defect (1210)_mask.png"
).convert("L").resize(size)

prompt = "a surface with scratch defects and a surface with spot defects."

scale_map = {}
with open(scale_txt_path, "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split(',')
        if len(parts) >= 6:
            fname, scale, x1, y1, x2, y2 = parts
            scale_map[fname.strip()] = (float(scale), tuple(map(int, [x1, y1, x2, y2])))

scale, (x1, y1, x2, y2) = scale_map[filename]
ori_crop = ori_img.crop((x1, y1, x2, y2))

crop_image = ori_crop.resize(size, Image.BICUBIC)

# generator = torch.Generator(device="cuda").manual_seed(24)
generator = torch.manual_seed(0)

# 采用mask作为ControlNet的输入
crop_gen_image = pipe(
    prompt=prompt,
    image=crop_image,
    mask_image=crop_mask,
    controlnet_conditioning_image=crop_mask.convert("RGB"),  # ControlNet 用 mask 作为控制输入
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

orig_np = np.array(ori_img)

paste_w = x2 - x1
paste_h = y2 - y1

if paste_w <= 0 or paste_h <= 0:
    raise ValueError(f"Invalid crop size: width={paste_w}, height={paste_h} from x1={x1}, x2={x2}, y1={y1}, y2={y2}")

gen_resized = crop_gen_image.resize((paste_w, paste_h), Image.BICUBIC)
gen_resized_np = np.array(gen_resized)

orig_np[y1:y2, x1:x2] = gen_resized_np
blended_img = Image.fromarray(orig_np)
blended_img.save(f"full_gen.png")
