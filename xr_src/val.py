import os
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from tqdm import tqdm
import csv
from collections import defaultdict

# 路径配置
gen_dir = "validation/output-seg-masks"
image_dir = "diffuser/images"
mask_dir = "diffuser/conditioning_images"
prompt_dir = "diffuser/captions"
eval_csv = "/home/me241123/xr/diffusers/1.csv"

defect_types = ["spot", "scratch", "dent", "bulge"]

# 确保输出路径存在
os.makedirs(os.path.dirname(eval_csv), exist_ok=True)

# 初始化 LPIPS 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
lpips_model = lpips.LPIPS(net='alex').to(device)

def load_mask(path, size=(512, 512)):
    mask = Image.open(path).convert("L").resize(size)
    return (np.array(mask) > 127).astype(np.uint8)

def load_image(path, size=(512, 512), mode='RGB'):
    return Image.open(path).convert(mode).resize(size)

def normalize(img):
    return img.astype(np.float32) / 255.0

def compute_lpips(img1, img2, mask):
    img1 = normalize(img1)
    img2 = normalize(img2)
    img1_t = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).to(device)
    img2_t = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).to(device)
    mask_t = torch.tensor(mask).unsqueeze(0).unsqueeze(0).to(device)
    img1_t = img1_t * mask_t
    img2_t = img2_t * mask_t
    with torch.no_grad():
        dist = lpips_model(img1_t, img2_t).item()
    return dist

def compute_ssim_psnr(img1, img2, mask):
    inv_mask = 1 - mask
    img1 = normalize(img1)
    img2 = normalize(img2)
    ssim_total, psnr_total, count = 0, 0, 0
    for c in range(3):
        img1_c = img1[:, :, c] * inv_mask
        img2_c = img2[:, :, c] * inv_mask
        if np.any(inv_mask):
            ssim_total += ssim(img1_c, img2_c, data_range=1)
            psnr_total += psnr(img1_c, img2_c, data_range=1)
            count += 1
    return ssim_total / count, psnr_total / count

# 存储结果
all_results = []
by_defect = defaultdict(list)

generated_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png") or f.endswith(".jpg")])

for gen_file in tqdm(generated_files, desc="评估生成图像"):
    name = os.path.splitext(gen_file)[0]
    image_path = os.path.join(image_dir, f"{name}.jpg")
    mask_path = os.path.join(mask_dir, f"{name}_mask.png")
    gen_path = os.path.join(gen_dir, f"{name}.png")
    prompt_path = os.path.join(prompt_dir, f"{name}.txt")

    if not (os.path.exists(image_path) and os.path.exists(mask_path) and os.path.exists(gen_path)):
        print(f"[跳过] 缺少文件：{name}")
        continue

    prompt = ""
    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as f:
            prompt = f.read().strip()

    # 推断 defect 类型
    defect_type = "unknown"
    for d in defect_types:
        if d in prompt.lower():
            defect_type = d
            break

    # 加载图像
    gt = np.array(Image.open(image_path).convert("RGB").resize((512, 512)))
    gen = np.array(Image.open(gen_path).convert("RGB").resize((512, 512)))
    mask_bin = load_mask(mask_path)

    # 评估
    lp = compute_lpips(gt, gen, mask_bin)
    ssim_score, psnr_score = compute_ssim_psnr(gt, gen, mask_bin)

    result = {
        "filename": name,
        "defect_type": defect_type,
        "prompt": prompt,
        "LPIPS (defect)": lp,
        "SSIM (background)": ssim_score,
        "PSNR (background)": psnr_score
    }

    all_results.append(result)
    by_defect[defect_type].append(result)

# 写入 CSV
with open(eval_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
    writer.writeheader()
    writer.writerows(all_results)

# 打印每类平均值
print("\n=== 每类缺陷平均指标 ===")
for defect_type, group in by_defect.items():
    avg_lp = np.mean([r["LPIPS (defect)"] for r in group])
    avg_ssim = np.mean([r["SSIM (background)"] for r in group])
    avg_psnr = np.mean([r["PSNR (background)"] for r in group])
    print(f"→ {defect_type}: LPIPS={avg_lp:.4f}, SSIM={avg_ssim:.4f}, PSNR={avg_psnr:.2f}")

# 打印总体平均
print("\n=== 所有图像平均指标 ===")
avg_lp = np.mean([r["LPIPS (defect)"] for r in all_results])
avg_ssim = np.mean([r["SSIM (background)"] for r in all_results])
avg_psnr = np.mean([r["PSNR (background)"] for r in all_results])
print(f"总体: LPIPS={avg_lp:.4f}, SSIM={avg_ssim:.4f}, PSNR={avg_psnr:.2f}")