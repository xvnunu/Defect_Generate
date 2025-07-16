import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from diffusers import ControlNetModel
from pipeline import StableDiffusionControlNetInpaintImg2ImgPipeline
from cleanfid import fid
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from lpips import LPIPS
import csv, json

# 配置路径
base_dir = "diffuser"
ori_dir = os.path.join(base_dir, "images")
mask_dir = os.path.join(base_dir, "conditioning_images")
prompt_dir = os.path.join(base_dir, "captions")
real_dir = os.path.join(base_dir, "images")
gen_dir = "validation/output-seg-masks"
os.makedirs(gen_dir, exist_ok=True)

# 控制生成与否
generate_images = False

# 模型加载
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionControlNetInpaintImg2ImgPipeline.from_pretrained(
    "pretrained_model/stable-diffusion-v1-5/stable-diffusion-inpainting",
    controlnet=ControlNetModel.from_pretrained(
    "output/controlnet-inpainting-segment-masks", torch_dtype=torch.float16
    ),
    safety_checker=None,
    torch_dtype=torch.float16
).to(device)

pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

# 图像转换
to_tensor = transforms.ToTensor()
resize = transforms.Resize((512, 512))

# LPIPS 模型
lpips_model = LPIPS(net='alex').to(device)

# 初始化指标容器
results = {"spot": {"lpips": [], "ssim": [], "psnr": []},
           "scratch": {"lpips": [], "ssim": [], "psnr": []},
           "dent": {"lpips": [], "ssim": [], "psnr": []},
           "bulge": {"lpips": [], "ssim": [], "psnr": []}}

files = sorted(os.listdir(ori_dir))
for filename in tqdm(files, desc="Processing"):
    name = os.path.splitext(filename)[0]
    prompt_path = os.path.join(prompt_dir, name + ".txt")
    mask_path = os.path.join(mask_dir, name + "_mask.png")
    real_path = os.path.join(real_dir, name + ".jpg")
    ori_path = os.path.join(ori_dir, filename)
    gen_path = os.path.join(gen_dir, name + ".png")

    # 读取数据
    image = resize(Image.open(ori_path).convert("RGB"))
    mask = resize(Image.open(mask_path).convert("L"))
    with open(prompt_path, "r") as f:
        prompt = f.read().strip()
    if generate_images or not os.path.exists(gen_path):
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            controlnet_conditioning_image=mask.convert("RGB"),
            num_inference_steps=50,
            guidance_scale=7.5
        )
        gen_image = result.images[0]
        gen_image.save(gen_path)
    else:
        gen_image = Image.open(gen_path).convert("RGB")

    # 读取真实缺陷图
    real_image = resize(Image.open(real_path).convert("RGB"))

    # 转换为张量并归一化
    gen_tensor = to_tensor(gen_image).unsqueeze(0).to(device)
    real_tensor = to_tensor(real_image).unsqueeze(0).to(device)

    # LPIPS
    lp = lpips_model(gen_tensor, real_tensor).item()

    # SSIM & PSNR (仅在mask区域)
    gen_np = np.array(gen_image)
    real_np = np.array(real_image)
    mask_np = np.array(mask).astype(bool)

    # SSIM
    try:
        ssim = compare_ssim(
            gen_np,
            real_np,
            channel_axis=2,
            data_range=255,
            win_size=7,
            gaussian_weights=True,
            use_sample_covariance=False,
        )
    except Exception as e:
        print(f"SSIM计算失败: {e}")
        ssim = None

    # PSNR (mask区域)
    try:
        psnr = compare_psnr(real_np[mask_np], gen_np[mask_np], data_range=255)
    except Exception as e:
        print(f"PSNR计算失败: {e}")
        psnr = None

    # 分类
    cls = None
    for key in results.keys():
        if key in prompt.lower():
            cls = key
            break
    if cls is None:
        cls = "unknown"
        if cls not in results:
            results[cls] = {"lpips": [], "ssim": [], "psnr": []}

    # 一定要把指标追加放在循环体内，否则只统计最后一个文件！
    if lp is not None:
        results[cls]["lpips"].append(lp)
    if ssim is not None:
        results[cls]["ssim"].append(ssim)
    if psnr is not None:
        results[cls]["psnr"].append(psnr)


# 保存指标数据
os.makedirs(gen_dir, exist_ok=True) 
csv_path = os.path.join(gen_dir, "per_image_metrics.csv")
json_path = os.path.join(gen_dir, "summary_metrics.json")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "DefectType", "LPIPS", "SSIM", "PSNR"])
    for cls in results:
        for i in range(len(results[cls]["lpips"])):
            writer.writerow([
                f"{cls}_{i}",
                cls,
                results[cls]["lpips"][i],
                results[cls]["ssim"][i] if i < len(results[cls]["ssim"]) else "",
                results[cls]["psnr"][i] if i < len(results[cls]["psnr"]) else ""
            ])

# 计算每类平均值和总体加权平均
summary = {}
total_count = 0
w_lpips = w_ssim = w_psnr = 0
for cls, res in results.items():
    n = len(res["lpips"])
    if n == 0:
        continue
    l, s, p = np.mean(res["lpips"]), np.mean(res["ssim"]), np.mean(res["psnr"])
    summary[cls] = {"LPIPS": round(l, 4), "SSIM": round(s, 4), "PSNR": round(p, 2), "Count": n}
    total_count += n
    w_lpips += l * n
    w_ssim += s * n
    w_psnr += p * n
    print(f"{cls:<8}: LPIPS={l:.4f}, SSIM={s:.4f}, PSNR={p:.2f}, Count={n}")

summary["__overall__"] = {
    "LPIPS": round(w_lpips / total_count, 4),
    "SSIM": round(w_ssim / total_count, 4),
    "PSNR": round(w_psnr / total_count, 2)
}

print("\nCalculating FID...")
fid_score = fid.compute_fid(gen_dir, real_dir)
print(f"FID: {fid_score:.2f}")

# 把 FID 加入 summary 字典
summary["__overall__"]["FID"] = round(fid_score, 4)

# 写 summary JSON（覆盖之前写法）
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"所有指标（含FID）已保存到: {json_path}")