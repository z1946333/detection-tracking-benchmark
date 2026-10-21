import numpy as np
from pathlib import Path
import cv2
import sys
import argparse
from types import SimpleNamespace
from ultralytics import YOLO

# 导入 Hybrid-SORT 核心类
sys.path.append(r"D:\GitHub\detection-tracking-benchmark\third_party\HybridSORT")
from trackers.hybrid_sort_tracker.hybrid_sort import Hybrid_Sort

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequences', nargs='+', default=["191", "192", "193"])
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    model = YOLO(r"D:\GitHub\detection-tracking-benchmark\experiments\detection\yolov8n\weights\best.pt")
    out_dir = Path(args.output) / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    for seq in args.sequences:
        print(f"========== 正在运行 Hybrid-SORT 序列 {seq} ==========")
        img_dir = Path(rf"D:\Datasets\dataset_B\PIG_MOT_TEST_DATA\{seq}\img1")
        img_paths = sorted(list(img_dir.glob("*.jpg")))
        if args.max_frames:
            img_paths = img_paths[:args.max_frames]

                # 按文档要求的参数起点构造 args
        tracker_args = SimpleNamespace(
            track_thresh=0.30,
            TCM_first_step=False,
            TCM_byte_step=False,
            TCM_first_step_weight=1.0,
            TCM_byte_step_weight=1.0,
            low_thresh=0.10,
            use_byte=False,
        )
        
        # 补齐 Hybrid-SORT 初始化需要的其他显式参数
        tracker = Hybrid_Sort(
            tracker_args,
            det_thresh=0.30,  # 补上这个
            max_age=30,
            min_hits=3,
            iou_threshold=0.30,
            delta_t=3,
            asso_func='iou',
            inertia=0.20,
            use_byte=False
        )
        
        mot_lines = []

        for frame_id, img_path in enumerate(img_paths, start=1):
            img = cv2.imread(str(img_path))
            img_h, img_w = img.shape[:2]
            
            results = model(img, verbose=False, conf=0.3)[0]
            dets = results.boxes.data.cpu().numpy()
            
            if len(dets) > 0:
                # 只需要 [x1, y1, x2, y2, conf]
                dets_for_tracker = dets[:, :5].astype(np.float32)
            else:
                dets_for_tracker = np.empty((0, 5), dtype=np.float32)

            # 更新
            tracks = tracker.update(dets_for_tracker, [img_h, img_w], [img_h, img_w])
            
            if tracks is not None and len(tracks) > 0:
                for t in tracks:
                    x1, y1, x2, y2 = t[:4]
                    track_id = int(t[4])
                    mot_lines.append(f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},1,-1,-1,-1")
                    
            if frame_id % 500 == 0:
                print(f"  已处理 {frame_id}/{len(img_paths)} 帧...")

        with open(out_dir / f"{seq}.txt", "w") as f:
            f.write("\n".join(mot_lines))
        print(f"✅ 序列 {seq} 完成，共 {len(mot_lines)} 个框。")

if __name__ == "__main__":
    main()