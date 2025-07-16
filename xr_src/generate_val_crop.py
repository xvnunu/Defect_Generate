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
# 原始数据
diffuser_dir = "diffuser"
ori_dir = os.path.join(diffuser_dir, "images")
mask_dir = os.path.join(diffuser_dir, "conditioning_images")
prompt_dir = os.path.join(diffuser_dir, "captions")
real_dir = ori_dir  # 真实图像同原图

# 裁剪数据
crop_dir = "croped_scratch_defect"
crop_img_dir = os.path.join(crop_dir, "images")
crop_mask_dir = os.path.join(crop_dir, "conditioning_images")
scale_txt_path = os.path.join(crop_dir, "scales.txt")

# 生成图像与融合图像输出
output_dir = "validation/output-seg-masks-crop-scratch"
gen_dir = os.path.join(output_dir, "generated_defect_crops")
blended_dir = os.path.join(output_dir, "blended_full_images")       
os.makedirs(gen_dir, exist_ok=True)
os.makedirs(blended_dir, exist_ok=True)

# 控制生成与否
generate_images = False

# 模型加载
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionControlNetInpaintImg2ImgPipeline.from_pretrained(
    "pretrained_model/stable-diffusion-v1-5/stable-diffusion-inpainting",
    controlnet=ControlNetModel.from_pretrained(
        "output/controlnet-inpainting-segment-only-scratch", torch_dtype=torch.float16
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

# 指标统计
results = {k: {"lpips": [], "ssim": [], "psnr": []} for k in ["spot", "scratch", "dent", "bulge"]}

# === 读取 scale 信息 ===
scale_map = {}
with open(scale_txt_path, "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split(',')
        if len(parts) >= 6:
            fname, scale, x1, y1, x2, y2 = parts
            scale_map[fname] = (float(scale), tuple(map(int, [x1, y1, x2, y2])))

# 遍历裁剪图像并生成缺陷
files = sorted(os.listdir(crop_img_dir))
for filename in tqdm(files, desc="Processing"):
    name = os.path.splitext(filename)[0]
    prompt_path = os.path.join(prompt_dir, name + ".txt")
    crop_mask_path = os.path.join(crop_mask_dir, name + "_mask.png")
    mask_path = os.path.join(mask_dir, name + "_mask.png")
    crop_img_path = os.path.join(crop_img_dir, name + ".jpg")
    real_path = os.path.join(real_dir, name + ".jpg")
    ori_path = os.path.join(ori_dir, filename)
    gen_path = os.path.join(gen_dir, name + ".png")

    # 读取数据
    image = resize(Image.open(crop_img_path).convert("RGB"))
    crop_mask = resize(Image.open(crop_mask_path).convert("L"))
    mask = resize(Image.open(mask_path).convert("L"))
    with open(prompt_path, "r") as f:
        prompt = f.read().strip()

    # 生成图像
    if generate_images or not os.path.exists(gen_path):
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=crop_mask,
            controlnet_conditioning_image=crop_mask.convert("RGB"),
            num_inference_steps=50,
            guidance_scale=7.5
        )
        gen_image = result.images[0]
        gen_image.save(gen_path)
    else:
        gen_image = Image.open(gen_path).convert("RGB")

    # === 返回原始图并进行融合 ===
    if filename in scale_map:
        scale, (x1, y1, x2, y2) = scale_map[filename]
        orig_img = Image.open(ori_path).convert("RGB")
        orig_np = np.array(orig_img)

        paste_w = x2 - x1
        paste_h = y2 - y1
        gen_resized = gen_image.resize((paste_w, paste_h), Image.BICUBIC)
        gen_resized_np = np.array(gen_resized)

        orig_np[y1:y2, x1:x2] = gen_resized_np
        blended_img = Image.fromarray(orig_np)
        blended_path = os.path.join(blended_dir, name + ".jpg")
        blended_img.save(blended_path)

    # === 指标计算 ===
    # 比较的是未裁剪的原始图像和重新粘贴后的图像
    real_image = resize(Image.open(real_path).convert("RGB"))
    blended_image = resize(Image.open(blended_path).convert("RGB"))
    
    # 将图像转换为张量
    blended_tensor = to_tensor(blended_image).unsqueeze(0).to(device)
    real_tensor = to_tensor(real_image).unsqueeze(0).to(device)

    # LPIPS
    lp = lpips_model(blended_tensor, real_tensor).item()

    # SSIM & PSNR (仅在mask区域)
    blended_np = np.array(blended_image)
    real_np = np.array(real_image)
    mask_np = np.array(resize(mask)).astype(bool)

    try:
        ssim = compare_ssim(blended_np, real_np, channel_axis=2, data_range=255,
                            win_size=7, gaussian_weights=True, use_sample_covariance=False)
    except Exception as e:
        print(f"SSIM计算失败: {e}")
        ssim = None

    if np.sum(mask_np) > 0:
        try:
            psnr = compare_psnr(real_np[mask_np], blended_np[mask_np], data_range=255)
        except Exception as e:
            print(f"PSNR计算失败: {e}")
            psnr = None
    else:
        print(f"[跳过] mask 区域为空: {filename}")
        psnr = None

    cls = next((key for key in results if key in prompt.lower()), "unknown")
    if cls not in results:
        results[cls] = {"lpips": [], "ssim": [], "psnr": []}

    if lp is not None: results[cls]["lpips"].append(lp)
    if ssim is not None: results[cls]["ssim"].append(ssim)
    if psnr is not None: results[cls]["psnr"].append(psnr)

# === 保存指标 ===
csv_path = os.path.join(output_dir, "per_image_metrics.csv")
json_path = os.path.join(output_dir, "summary_metrics.json")
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

# === 统计 ===
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
summary["__overall__"]["FID"] = round(fid_score, 4)

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"所有指标（含FID）已保存到: {json_path}")
