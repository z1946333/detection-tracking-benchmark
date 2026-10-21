import json
import os
from pathlib import Path
import cv2

# 你的数据路径
DATA_DIR = Path(r"D:\Datasets\processed_detection")
OUT_DIR = Path(r"D:\Datasets\processed_detection\annotations")
OUT_DIR.mkdir(exist_ok=True)

splits = ["train", "val", "test"]  # 加上 test

def convert_split(split):
    img_dir = DATA_DIR / "images" / split
    label_dir = DATA_DIR / "labels" / split
    
    images = []
    annotations = []
    ann_id = 1
    
    img_files = sorted(list(img_dir.glob("*.jpg")))
    for img_id, img_path in enumerate(img_files, start=1):
        img = cv2.imread(str(img_path))
        h, w, _ = img.shape
        
        images.append({
            "id": img_id,
            "file_name": str(img_path.absolute()),  # 修改这里！改成绝对路径
            "width": w,
            "height": h
        })
        
        txt_path = label_dir / (img_path.stem + ".txt")
        if txt_path.exists():
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    cls_id = int(parts[0])
                    x_c, y_c, bw, bh = map(float, parts[1:])
                    
                    w_pix = bw * w
                    h_pix = bh * h
                    x_pix = (x_c * w) - (w_pix / 2)
                    y_pix = (y_c * h) - (h_pix / 2)
                    
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1, # YOLOX 默认类别从 1 开始
                        "bbox": [x_pix, y_pix, w_pix, h_pix],
                        "area": w_pix * h_pix,
                        "iscrowd": 0
                    })
                    ann_id += 1
                    
    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "pig", "supercategory": "animal"}]
    }
    
    out_file = OUT_DIR / f"instances_{split}.json"
    with open(out_file, "w") as f:
        json.dump(coco_dict, f)
    print(f"✅ 生成 {split} 集 JSON: {out_file}")

if __name__ == "__main__":
    for s in splits:
        convert_split(s)