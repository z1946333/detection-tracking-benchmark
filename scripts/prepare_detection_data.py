import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# ================= 配置区域 =================
# 1. 原始数据集路径（请根据你的实际情况修改！）
# 假设图片在 'a' 文件夹，标注在 'label' 文件夹
RAW_IMG_DIR = Path(r"D:\Datasets\dataset_A\pig-objection detection\data")
RAW_XML_DIR = Path(r"D:\Datasets\dataset_A\pig-objection detection\label")

# 2. 转换后的数据集存放位置（放在原目录旁边，方便管理）
OUT_DIR = Path(r"D:\Datasets\processed_detection")

# 3. 划分比例 (训练集 : 验证集 : 测试集)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 4. 类别映射
CLASS_MAP = {"pig": 0}
# ============================================

def create_dirs():
    for split in ["train", "val", "test"]:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def convert_xml_to_yolo(xml_path, img_width, img_height):
    """解析 XML 并转换为 YOLO 格式字符串列表"""
    yolo_lines = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    for obj in root.findall('object'):
        cls_name = obj.find('name').text.strip()
        if cls_name not in CLASS_MAP:
            continue # 如果遇到未知类别，跳过（或者你可以记录报错）
            
        cls_id = CLASS_MAP[cls_name]
        
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        # 计算 YOLO 格式：中心点 x, y 和宽高 w, h (均需归一化)
        x_center = ((xmin + xmax) / 2.0) / img_width
        y_center = ((ymin + ymax) / 2.0) / img_height
        w = (xmax - xmin) / img_width
        h = (ymax - ymin) / img_height
        
        # 格式化为字符串，保留 6 位小数
        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
        
    return yolo_lines

def main():
    print("开始处理...")
    create_dirs()
    
    # 找到所有 XML 文件
    xml_files = list(RAW_XML_DIR.glob("*.xml"))
    # 随机打乱
    random.seed(42) # 固定随机种子，保证每次划分一样
    random.shuffle(xml_files)
    
    total = len(xml_files)
    train_end = int(total * TRAIN_RATIO)
    val_end = int(total * (TRAIN_RATIO + VAL_RATIO))
    
    splits = {
        "train": xml_files[:train_end],
        "val": xml_files[train_end:val_end],
        "test": xml_files[val_end:]
    }
    
    for split_name, files in splits.items():
        print(f"正在处理 {split_name} 集，共 {len(files)} 张图片...")
        for xml_path in files:
            # 解析 XML 获取图片宽高和文件名
            tree = ET.parse(xml_path)
            root = tree.getroot()
            filename = root.find('filename').text
            width = float(root.find('size/width').text)
            height = float(root.find('size/height').text)
            
            # 获取 YOLO 标注
            yolo_lines = convert_xml_to_yolo(xml_path, width, height)
            
            if not yolo_lines:
                continue # 如果这图里没有 pig，跳过
                
            # 写入 TXT 标注
            txt_name = xml_path.stem + ".txt"
            txt_out_path = OUT_DIR / "labels" / split_name / txt_name
            with open(txt_out_path, "w") as f:
                f.writelines(yolo_lines)
                
            # 复制图片
            img_src_path = RAW_IMG_DIR / filename
            if img_src_path.exists():
                img_dst_path = OUT_DIR / "images" / split_name / filename
                shutil.copy(img_src_path, img_dst_path)
            else:
                print(f"警告: 找不到图片 {img_src_path}")

    print(f"全部完成！数据保存在: {OUT_DIR}")

if __name__ == "__main__":
    main()