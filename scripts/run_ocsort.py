import numpy as np
from pathlib import Path
import cv2
import sys
from ultralytics import YOLO

# 1. 动态导入 OC-SORT 的跟踪器
sys.path.append(r"D:\GitHub\detection-tracking-benchmark\third_party\OC_SORT")
# 注意：不同版本的 OC-SORT 路径可能略有不同，常见的导入是下面这个
try:
    from trackers.oc_sort import OCSort
except ImportError:
    from trackers.ocsort_tracker.ocsort import OCSort

def main():
    # 2. 加载检测器
    model = YOLO(r"D:\GitHub\detection-tracking-benchmark\experiments\detection\yolov8n\weights\best.pt")
    
    # 3. 输出路径
    out_dir = Path(r"D:\GitHub\detection-tracking-benchmark\third_party\TrackEval\data\trackers\mot_challenge\PIG-train\my_ocsort_tracker\data")
    out_dir.mkdir(parents=True, exist_ok=True)

    for seq in ["191", "192", "193"]:
        print(f"========== 正在用 OC-SORT 处理序列 {seq} ==========")
        img_dir = Path(rf"D:\Datasets\dataset_B\PIG_MOT_TEST_DATA\{seq}\img1")
        img_paths = sorted(list(img_dir.glob("*.jpg")))
        
        # 初始化 OC-SORT 跟踪器
        # 参数一般不需要大改，可以根据需要调 det_thresh
        tracker = OCSort(det_thresh=0.3, max_age=30, min_hits=3, iou_threshold=0.3)
        mot_lines = []

        for frame_id, img_path in enumerate(img_paths, start=1):
            img = cv2.imread(str(img_path))
            h, w, _ = img.shape
            
            # 检测
            results = model(img, verbose=False, conf=0.1)[0]
            dets = results.boxes.data.cpu().numpy()
            
            # 提取 [x1, y1, x2, y2, score] 给 OC-SORT
            if len(dets) > 0:
                dets_for_tracker = dets[:, :5]
            else:
                dets_for_tracker = np.empty((0, 5))

            # 更新轨迹
            online_targets = tracker.update(dets_for_tracker, [h, w], [h, w])
            
            for t in online_targets:
                # OC-SORT 通常返回 [x1, y1, x2, y2, track_id]
                x1, y1, x2, y2 = t[0], t[1], t[2], t[3]
                track_id = int(t[4])
                
                # 转为 MOT 格式
                w_pix = x2 - x1
                h_pix = y2 - y1
                mot_lines.append(f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w_pix:.2f},{h_pix:.2f},1.0,-1,-1,-1")
                
            if frame_id % 200 == 0:
                print(f"  已处理 {frame_id} 帧...")

        with open(out_dir / f"{seq}.txt", "w") as f:
            f.write("\n".join(mot_lines))
        print(f"✅ 序列 {seq} OC-SORT 结果已写入 {seq}.txt")

if __name__ == "__main__":
    main()