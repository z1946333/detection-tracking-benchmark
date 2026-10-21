import json
import cv2
import os
import sys
from pathlib import Path
import argparse

# 把你的本地 YOLOv5 路径加入环境变量
YOLOV5_DIR = r"D:\GitHub\detection-tracking-benchmark\third_party\yolov5"
sys.path.append(YOLOV5_DIR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='模型的 .pt 路径')
    parser.add_argument('--model_name', type=str, required=True, help='模型名称，用于保存文件名')
    parser.add_argument('--conf', type=float, default=0.001, help='置信度阈值')
    args = parser.parse_args()

    DATA_DIR = Path(r"D:\Datasets\processed_detection")
    TEST_JSON = DATA_DIR / "annotations" / "instances_test.json"
    OUT_DIR = Path(r"D:\GitHub\detection-tracking-benchmark\evaluation\detection")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(TEST_JSON, 'r') as f:
        coco_data = json.load(f)
    img_name_to_id = {img['file_name']: img['id'] for img in coco_data['images']}
    
    results_list = []
    img_dir = DATA_DIR / "images" / "test"
    img_paths = list(img_dir.glob("*.jpg"))
    print(f"测试集共 {len(img_paths)} 张图片，开始推理...")

    # 判断是 YOLOv5 还是 YOLOv8
    if 'yolov5' in args.model_name.lower():
        print("检测到 YOLOv5 权重，正在使用 YOLOv5 本地引擎加载...")
        import torch
        # 使用 torch.hub 加载本地 YOLOv5 源码和权重
        model = torch.hub.load(YOLOV5_DIR, 'custom', path=args.weights, source='local')
        model.conf = args.conf
        
        for img_path in img_paths:
            abs_path = str(img_path.absolute())
            results = model(abs_path, size=640)
            df = results.pandas().xyxy[0] # 获取预测结果 DataFrame
            
            image_id = img_name_to_id.get(abs_path)
            for _, row in df.iterrows():
                results_list.append({
                    "image_id": image_id,
                    "category_id": int(row['class']) + 1, # 类别 ID 从 1 开始
                    "bbox": [round(row['xmin'], 2), round(row['ymin'], 2), round(row['xmax'] - row['xmin'], 2), round(row['ymax'] - row['ymin'], 2)],
                    "score": round(row['confidence'], 4)
                })
    else:
        print("检测到 YOLOv8 权重，正在使用 Ultralytics 引擎加载...")
        from ultralytics import YOLO
        model = YOLO(args.weights)
        
        for img_path in img_paths:
            abs_path = str(img_path.absolute())
            results = model.predict(abs_path, conf=args.conf, verbose=False)
            image_id = img_name_to_id.get(abs_path)
            
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    results_list.append({
                        "image_id": image_id,
                        "category_id": int(box.cls[0].item()) + 1,
                        "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                        "score": round(box.conf[0].item(), 4)
                    })

    out_file = OUT_DIR / f"{args.model_name}_predictions.json"
    with open(out_file, 'w') as f:
        json.dump(results_list, f)
    print(f"✅ 预测结果已保存至: {out_file} (共 {len(results_list)} 个预测框)")

if __name__ == '__main__':
    main()