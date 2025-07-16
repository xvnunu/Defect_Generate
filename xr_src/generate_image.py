import os
import cv2
import numpy as np

def crop_resize_from_mask(image, mask, min_size=50, target_size=1000):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, None, None, None  # 无有效mask

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    width = x_max - x_min + 1
    height = y_max - y_min + 1
    side = max(width, height, min_size)

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    half_side = side // 2

    x1 = max(cx - half_side, 0)
    y1 = max(cy - half_side, 0)
    x2 = x1 + side
    y2 = y1 + side

    h, w = image.shape[:2]
    if x2 > w:
        x1 = max(w - side, 0)
        x2 = w
    if y2 > h:
        y1 = max(h - side, 0)
        y2 = h

    cropped_image = image[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]

    actual_side = cropped_image.shape[0]
    scale_ratio = target_size / actual_side

    resized_image = cv2.resize(cropped_image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(cropped_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

    return resized_image, resized_mask, scale_ratio, (x1, y1, x2, y2)

def batch_process(image_dir, mask_dir, output_dir, min_size=50, target_size=1000):
    output_img_dir = os.path.join(output_dir, "images")
    output_mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    scale_file_path = os.path.join(output_dir, "scales.txt")
    with open(scale_file_path, "w") as scale_file:
        scale_file.write("filename,scale,x1,y1,x2,y2\n")  # header

        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for fname in image_files:
            name_no_ext, ext = os.path.splitext(fname)
            image_path = os.path.join(image_dir, fname)

            mask_name = f"{name_no_ext}_mask.png"
            mask_path = os.path.join(mask_dir, mask_name)

            out_img_path = os.path.join(output_img_dir, fname)
            out_mask_path = os.path.join(output_mask_dir, mask_name)

            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                print(f"跳过（读取失败）: {fname}")
                continue

            result_img, result_mask, scale, crop_box = crop_resize_from_mask(image, mask, min_size, target_size)
            if result_img is None:
                print(f"跳过（无有效mask）: {fname}")
                continue

            cv2.imwrite(out_img_path, result_img)
            cv2.imwrite(out_mask_path, result_mask)

            x1, y1, x2, y2 = crop_box
            scale_file.write(f"{fname},{scale:.6f},{x1},{y1},{x2},{y2}\n")
            print(f"已处理: {fname}, 缩放比例: {scale:.6f}, 裁剪框: ({x1},{y1})-({x2},{y2})")

# ✅ 示例调用
if __name__ == "__main__":
    image_folder = "/home/me241123/xr/diffusers/diffuser/images"     # 原图路径
    mask_folder = "/home/me241123/xr/diffusers/diffuser/conditioning_images"       # mask路径（同名）
    output_folder = "/home/me241123/xr/diffusers/croped_dataset"    # 输出路径：output/images, output/masks, output/scales.txt

    batch_process(image_folder, mask_folder, output_folder)
