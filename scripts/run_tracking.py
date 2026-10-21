from ultralytics import YOLO
from pathlib import Path
import cv2

# 1. 加载模型
model = YOLO(r"D:\GitHub\detection-tracking-benchmark\experiments\detection\yolov8n\weights\best.pt")

# 专门为 BoT-SORT 建一个独立文件夹！
out_dir = Path(r"D:\GitHub\detection-tracking-benchmark\third_party\TrackEval\data\trackers\mot_challenge\PIG-train\my_botsort_tracker\data")
out_dir.mkdir(parents=True, exist_ok=True)

# 2. 循环处理三个序列
for seq in ["191", "192", "193"]:
    print(f"========== 正在处理序列 {seq} ==========")
    img_dir = Path(rf"D:\Datasets\dataset_B\PIG_MOT_TEST_DATA\{seq}\img1")
    
    # 运行跟踪 (这里改成了 botsort.yaml)
    results = model.track(
        source=str(img_dir), 
        tracker="botsort.yaml", # 👈 BoT-SORT
        persist=True, 
        stream=True, 
        save_txt=True, 
        project=str(out_dir),
        name=f"test_run_botsort_{seq}", # 👈 改了名字
        verbose=False
    )
    
    for r in results:
        pass

    # 3. 自动转换为 MOT 格式
    labels_dir = out_dir / f"test_run_botsort_{seq}" / "labels"
    txt_files = sorted(list(labels_dir.glob("*.txt")))
    
    if not txt_files:
        continue
        
    sample_img = cv2.imread(str(next(img_dir.glob("*.jpg"))))
    h, w, _ = sample_img.shape
    
    mot_lines = []
    for txt_file in txt_files:
        frame_id = int(txt_file.stem)
        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    x_c, y_c, bw, bh = map(float, parts[1:5])
                    track_id = int(parts[5])
                    
                    x_min = (x_c - bw / 2.0) * w
                    y_min = (y_c - bh / 2.0) * h
                    w_pix = bw * w
                    h_pix = bh * h
                    
                    mot_lines.append(f"{frame_id},{track_id},{x_min:.2f},{y_min:.2f},{w_pix:.2f},{h_pix:.2f},1.0,-1,-1,-1")

    with open(out_dir / f"{seq}.txt", "w") as f:
        f.write("\n".join(mot_lines))
    print(f"✅ 序列 {seq} BoT-SORT 结果已写入 {seq}.txt")