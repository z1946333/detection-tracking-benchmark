import cv2
from pathlib import Path

# 配置区域
VIDEO_DIR = Path(r"D:\Datasets\dataset_B\PIG_MOT_TEST_DATA")
# 在这里填入所有你需要处理的序列名字，比如 191, 192, 193
SEQUENCES = ["191", "192", "193"] 

def process_sequence(seq_name):
    # 猜测你的视频命名规则，如果是 191.mp4 就改成 f"{seq_name}.mp4"
    video_path = VIDEO_DIR / f"MyVideo_{seq_name}.mp4"
    if not video_path.exists():
        print(f"[跳过] 找不到视频: {video_path}")
        return

    # 建立 MOT 标准的 img1 文件夹
    out_dir = VIDEO_DIR / seq_name / "img1"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"正在处理视频: {video_path.name} ...")
    cap = cv2.VideoCapture(str(video_path))
    frame_id = 1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # MOT 规范：图片名必须是 6 位数字，例如 000001.jpg
        img_path = out_dir / f"{frame_id:06d}.jpg"
        cv2.imwrite(str(img_path), frame)
        frame_id += 1
        
    cap.release()
    print(f"✅ 序列 {seq_name} 抽帧完成，共生成 {frame_id-1} 张图片，保存在 {out_dir}")

if __name__ == "__main__":
    for seq in SEQUENCES:
        process_sequence(seq)