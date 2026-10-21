from pathlib import Path
import cv2

# 1. 路径配置 (改成你刚才生成的 test_run-2 路径)
LABELS_DIR = Path(r"D:\GitHub\detection-tracking-benchmark\third_party\TrackEval\data\trackers\mot_challenge\PIG-train\my_tracker\data\test_run-2\labels")
# 原图路径 (为了获取宽高，还原像素坐标)
IMG_DIR = Path(r"D:\Datasets\dataset_B\PIG_MOT_TEST_DATA\191\img1")
OUT_FILE = Path(r"D:\GitHub\detection-tracking-benchmark\third_party\TrackEval\data\trackers\mot_challenge\PIG-train\my_tracker\data\191.txt")

# 获取图片宽高 (1280x720)
sample_img = cv2.imread(str(next(IMG_DIR.glob("*.jpg"))))
h, w, _ = sample_img.shape

mot_lines = []
# 遍历所有的帧文件 (000001.txt 等)
txt_files = sorted(list(LABELS_DIR.glob("*.txt")))

for txt_file in txt_files:
    # 提取帧号 (例如 000001.txt -> 1)
    frame_id = int(txt_file.stem)
    
    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                # YOLO 格式: class_id, x_center, y_center, width, height, track_id
                x_c = float(parts[1])
                y_c = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
                track_id = int(parts[5])
                
                # 转换为像素的 x_min, y_min, w, h
                x_min = (x_c - bw / 2.0) * w
                y_min = (y_c - bh / 2.0) * h
                w_pix = bw * w
                h_pix = bh * h
                
                # MOT 格式: frame, id, x, y, w, h, score, -1, -1, -1
                mot_lines.append(f"{frame_id},{track_id},{x_min:.2f},{y_min:.2f},{w_pix:.2f},{h_pix:.2f},1.0,-1,-1,-1")

# 写入最终的 191.txt
with open(OUT_FILE, "w") as f:
    f.write("\n".join(mot_lines))

print(f"✅ 成功转换 {len(txt_files)} 帧，共 {len(mot_lines)} 个跟踪框，已保存至 {OUT_FILE}")