import numpy as np
from pathlib import Path
import cv2
import sys
import os
from unittest.mock import MagicMock
from ultralytics import YOLO

sys.modules['opts'] = MagicMock()
strongsort_dir = r"D:\GitHub\detection-tracking-benchmark\third_party\StrongSORT"
sys.path.append(strongsort_dir)
sys.path.append(os.path.join(strongsort_dir, "deep_sort"))

# 直接从 deep_sort 目录下的 deepsort 或 deep_sort 文件里找
try:
    from deep_sort.deepsort import DeepSort
    print("✅ 成功导入 DeepSort (来自 deepsort.py)")
except ImportError:
    try:
        from deep_sort.deep_sort import DeepSort
        print("✅ 成功导入 DeepSort (来自 deep_sort.py)")
    except ImportError as e:
        print(f"❌ 还是没找到，报错: {e}")
        sys.exit(1)

def main():
    print("正在加载 YOLOv8 检测器...")
    model = YOLO(r"D:\GitHub\detection-tracking-benchmark\experiments\detection\yolov8n\weights\best.pt")
    
    out_dir = Path(r"D:\GitHub\detection-tracking-benchmark\third_party\TrackEval\data\trackers\mot_challenge\PIG-train\my_strongsort\data")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 权重路径
    reid_weights = r"D:\GitHub\detection-tracking-benchmark\third_party\StrongSORT\osnet_x0_25_msmt17.pt"
    
    for seq in ["191", "192", "193"]:
        print(f"========== 正在用 StrongSORT 处理序列 {seq} ==========")
        img_dir = Path(rf"D:\Datasets\dataset_B\PIG_MOT_TEST_DATA\{seq}\img1")
        img_paths = sorted(list(img_dir.glob("*.jpg")))
        
         
                    # 注意：参数名是 model_path
        tracker = DeepSort(
            model_path=reid_weights,  # 👈 改成了 model_path
            device="cuda",
            fp16=True,
            max_dist=0.2,
            max_iou_distance=0.7,
            max_age=30,
            n_init=3,
            nn_budget=100
        )
        
        mot_lines = []

        for frame_id, img_path in enumerate(img_paths, start=1):
            img = cv2.imread(str(img_path))
            
            results = model(img, verbose=False, conf=0.3)[0]
            dets = results.boxes.data.cpu().numpy()
            
            if len(dets) > 0:
                bboxes = dets[:, :4]
                confidences = dets[:, 4]
                classes = dets[:, 5]
            else:
                bboxes = np.empty((0, 4))
                confidences = np.empty(0)
                classes = np.empty(0)

            tracker.update(bboxes, confidences, classes, img)
            
            for t in tracker.tracker.tracks:
                if not t.is_confirmed() or t.time_since_update > 1:
                    continue
                bbox = t.to_tlwh()
                track_id = t.track_id
                
                mot_lines.append(f"{frame_id},{track_id},{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f},1.0,-1,-1,-1")
                
            if frame_id % 200 == 0:
                print(f"  已处理 {frame_id} 帧...")

        with open(out_dir / f"{seq}.txt", "w") as f:
            f.write("\n".join(mot_lines))
        print(f"✅ 序列 {seq} StrongSORT 结果已写入 {seq}.txt")

if __name__ == "__main__":
    main()