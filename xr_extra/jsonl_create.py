import os
import json

# 文件夹路径
image_dir = "diffuser/images"
mask_dir = "diffuser/conditioning_images"
caption_dir = "diffuser/captions"

# 输出文件路径
output_jsonl = "train.jsonl"

# 获取所有图像文件（.jpg）
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])

with open(output_jsonl, "w", encoding="utf-8") as out_file:
    count = 0
    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]  # 如 "0"

        image_path = os.path.join(image_dir, f"{base_name}.jpg")
        mask_path = os.path.join(mask_dir, f"{base_name}_mask.png")
        caption_file = os.path.join(caption_dir, f"{base_name}.txt")

        # 检查 mask 和 caption 是否存在
        if not os.path.exists(mask_path):
            print(f"Missing mask for {base_name}")
            continue
        if not os.path.exists(caption_file):
            print(f"Missing caption for {base_name}")
            continue

        # 读取 caption 内容
        with open(caption_file, "r", encoding="utf-8") as f:
            caption = f.read().strip()

        # 写入一行 JSON
        json_obj = {
            "text": caption,
            "image": image_path,
            "conditioning_image": mask_path
        }
        out_file.write(json.dumps(json_obj) + "\n")
        count += 1

print(f"Finished writing {count} entries to {output_jsonl}")
