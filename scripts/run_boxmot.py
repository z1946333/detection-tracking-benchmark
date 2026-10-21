import numpy as np
from pathlib import Path
import cv2
import sys
from ultralytics import YOLO

# 稳定版可以直接从顶层导入
try:
    # BoxMOT 10.0.74/10.0.75 exports the class names with capital SORT.
    from boxmot import StrongSORT, HybridSORT
    print("✅ 成功导入 StrongSORT 和 HybridSORT！")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def main():
    model = YOLO(r"D:\GitHub\detection-tracking-benchmark\experiments\detection\yolov8n\weights\best.pt")
    
    out_dir = Path(r"D:\GitHub\detection-tracking-benchmark\third_party\TrackEval\data\trackers\mot_challenge\PIG-train")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 必须使用 Path 对象
    reid_weights = Path(r"D:\GitHub\detection-tracking-benchmark\third_party\StrongSORT\osnet_x0_25_msmt17.pt")

    print("\n正在初始化 StrongSORT...")
    strongsort_tracker = StrongSORT(
        model_weights=reid_weights,
        # BoxMOT 要传 GPU 编号 "0"，不能传 "cuda"。
        device="0",
        fp16=True
    )
        
    print("正在初始化 HybridSORT...")
    hybridsort_tracker = HybridSORT(
        reid_weights=reid_weights,
        device="0",
        half=True,
        det_thresh=0.3
    )

    algorithms = {
        "my_strongsort": strongsort_tracker,
        "my_hybridsort": hybridsort_tracker
    }

    for algo_name, tracker in algorithms.items():
        print(f"\n========== 正在运行 {algo_name} ==========")
        algo_out_dir = out_dir / algo_name / "data"
        algo_out_dir.mkdir(parents=True, exist_ok=True)

        for seq in ["191", "192", "193"]:
            print(f"  --> 处理序列 {seq}")
            img_dir = Path(rf"D:\Datasets\dataset_B\PIG_MOT_TEST_DATA\{seq}\img1")
            img_paths = sorted(list(img_dir.glob("*.jpg")))
            
            mot_lines = []
            
            if hasattr(tracker, 'reset'):
                tracker.reset()

            for frame_id, img_path in enumerate(img_paths, start=1):
                img = cv2.imread(str(img_path))
                
                results = model(img, verbose=False, conf=0.3)[0]
                dets = results.boxes.data.cpu().numpy()
                
                if len(dets) > 0:
                    online_targets = tracker.update(dets, img)
                else:
                    online_targets = tracker.update(np.empty((0, 6)), img)
                
                if online_targets is not None and len(online_targets) > 0:
                    for t in online_targets:
                        x1, y1, x2, y2 = t[:4]
                        track_id = int(t[4])
                        w_pix = x2 - x1
                        h_pix = y2 - y1
                        mot_lines.append(f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w_pix:.2f},{h_pix:.2f},1.0,-1,-1,-1")
                        
                if frame_id % 500 == 0:
                    print(f"      已处理 {frame_id}/{len(img_paths)} 帧...")

            with open(algo_out_dir / f"{seq}.txt", "w") as f:
                f.write("\n".join(mot_lines))
            print(f"  ✅ 序列 {seq} 完成！")

if __name__ == "__main__":
    main()
