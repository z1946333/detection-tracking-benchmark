import torch
import cv2
import json
import sys
from pathlib import Path

# 把 YOLOX 源码加入路径
sys.path.append(r"D:\GitHub\detection-tracking-benchmark\third_party\YOLOX")
from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.data.data_augment import ValTransform

def main():
    DATA_DIR = Path(r"D:\Datasets\processed_detection")
    TEST_JSON = DATA_DIR / "annotations" / "instances_test.json"
    OUT_DIR = Path(r"D:\GitHub\detection-tracking-benchmark\evaluation\detection")
    
    with open(TEST_JSON, 'r') as f:
        coco_data = json.load(f)
    img_name_to_id = {img['file_name']: img['id'] for img in coco_data['images']}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载配置和模型
    exp = get_exp(r"D:\GitHub\detection-tracking-benchmark\configs\detection\yolox_pig.py", None)
    model = exp.get_model().to(device)
    
    # 2. 加载训练好的权重 (best_ckpt.pth)
    ckpt_path = r"D:\GitHub\detection-tracking-benchmark\YOLOX_outputs\yolox_pig\best_ckpt.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    preproc = ValTransform(legacy=False)
    img_dir = DATA_DIR / "images" / "test"
    img_paths = list(img_dir.glob("*.jpg"))
    results_list = []

    print("开始推理 YOLOX...")
    with torch.no_grad():
        for img_path in img_paths:
            abs_path = str(img_path.absolute())
            img = cv2.imread(abs_path)
            ratio = min(exp.test_size[0] / img.shape[0], exp.test_size[1] / img.shape[1])
            
            # 预处理
            img_proc, _ = preproc(img, None, exp.test_size)
            img_tensor = torch.from_numpy(img_proc).unsqueeze(0).to(device)

            # 推理
            outputs = model(img_tensor)
            outputs = postprocess(outputs, exp.num_classes, 0.001, 0.65)[0]

            image_id = img_name_to_id.get(abs_path)
            if outputs is not None:
                outputs = outputs.cpu().numpy()
                for out in outputs:
                    x1, y1, x2, y2 = out[0:4]
                    score = out[4] * out[5]
                    cls_id = int(out[6])
                    
                    # 坐标还原到原图尺度
                    x1, y1, x2, y2 = x1 / ratio, y1 / ratio, x2 / ratio, y2 / ratio
                    
                    results_list.append({
                        "image_id": image_id,
                        "category_id": cls_id + 1,
                        "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                        "score": round(float(score), 4)
                    })

    out_file = OUT_DIR / "yolox_predictions.json"
    with open(out_file, 'w') as f:
        json.dump(results_list, f)
    print(f"✅ 导出成功: {out_file} (共 {len(results_list)} 个框)")

if __name__ == '__main__':
    main()