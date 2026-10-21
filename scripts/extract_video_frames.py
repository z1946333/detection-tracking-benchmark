import cv2
from pathlib import Path

# 以 191 序列为例
VIDEO_DIR = Path(r"D:\Datasets\dataset_B\PIG_MOT_TEST_DATA")
SEQ_NAME = "191" # 你需要确认 mp4 的真实名字
VIDEO_PATH = VIDEO_DIR / f"MyVideo_{SEQ_NAME}.mp4" # 假设视频名是这个
OUT_DIR = VIDEO_DIR / SEQ_NAME / "img1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    if not VIDEO_PATH.exists():
        print(f"找不到视频: {VIDEO_PATH}")
        exit()
    
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    frame_id = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # MOT 标准通常是 6 位数字命名: 000001.jpg
        img_path = OUT_DIR / f"{frame_id:06d}.jpg"
        cv2.imwrite(str(img_path), frame)
        frame_id += 1
        
    cap.release()
    print(f"序列 {SEQ_NAME} 抽帧完成，共 {frame_id-1} 帧，保存在 {OUT_DIR}")