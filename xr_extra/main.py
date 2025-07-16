# import torch
# from diffusers.utils import load_image, check_min_version
# from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
# from PIL import Image
# import numpy as np

# controlnet = ControlNetModel.from_pretrained(
#     "output/controlnet-inpainting-train-new/checkpoint-1500/controlnet", torch_dtype=torch.float16
# )
# # 加载轻量 inpainting pipeline
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#      "pretrained_model/runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
# )
# # pipe.text_encoder.to(torch.float16)
# # pipe.controlnet.to(torch.float16)
# # pipe.to("cuda")

# # remove following line if xformers is not installed or when using Torch 2.0.
# pipe.enable_xformers_memory_efficient_attention()
# # memory optimization.
# pipe.enable_model_cpu_offload()

# def make_inpaint_condition(image, image_mask):
#     image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
#     image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

#     assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
#     image[image_mask > 0.5] = -1.0  # set as masked pixel
#     image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
#     image = torch.from_numpy(image)
#     return image

# width = 512
# height = 512
# size = (width, height)
# image = load_image(
#     "/home/me241123/xr/diffusers/validation/defect (19).jpg"
# ).convert("RGB").resize(size)
# mask = load_image(
#     "validation/example.png"
# ).convert("L").resize(size)

# control_image = make_inpaint_condition(image, mask)

# prompt = "a surface with a dent defect."
# # generator = torch.Generator(device="cuda").manual_seed(24)
# generator = torch.manual_seed(0)
# res_image = pipe(
#     prompt=prompt,
#     image=image,
#     mask_image=mask,
#     control_image=control_image, 
#     controlnet_conditioning_scale=0.9,
#     num_inference_steps=30,
#     guidance_scale=7.5
# ).images[0]
# res_image.save(f"sd3.png")


import torch
from diffusers.utils import load_image, check_min_version
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from PIL import Image
import numpy as np

controlnet = ControlNetModel.from_pretrained(
    "output/controlnet-inpainting-train", torch_dtype=torch.float16
)
# 加载轻量 inpainting pipeline
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
     "pretrained_model/stable-diffusion-v1-5/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16
)
# pipe.text_encoder.to(torch.float16)
# pipe.controlnet.to(torch.float16)
# pipe.to("cuda")

# remove following line if xformers is not installed or when using Torch 2.0.
pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()


width = 1024
height = 1024
size = (width, height)
image = load_image(
    "validation/defect (19).jpg"
).convert("RGB").resize(size)
mask = load_image(
    "validation/defect (419)_mask.png"
).convert("L").resize(size)


prompt = "a surface with a scratch defect."
# generator = torch.Generator(device="cuda").manual_seed(24)
generator = torch.manual_seed(0)
res_image = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask,
    control_image=mask.convert("RGB"),  # ControlNet 用 mask 作为控制输入
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]
res_image.save(f"sd3.png")
