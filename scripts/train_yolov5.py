import subprocess
import sys

if __name__ == '__main__':
    cmd = [
        sys.executable, "third_party/yolov5/train.py",
        "--img", "640",
        "--batch", "4",
        "--epochs", "100",
        "--data", r"D:\GitHub\detection-tracking-benchmark\configs\detection\data_yolov5.yaml",
        "--weights", r"D:\GitHub\detection-tracking-benchmark\third_party\yolov5\yolov5n.pt", 
         "--resume",  # 如果下载失败，看下面第三步
        "--project", r"D:\GitHub\detection-tracking-benchmark\experiments\detection",
        "--name", "yolov5n",
        "--device", "0",
        "--workers", "2",
        "--seed", "42"
    ]
    
    print("正在启动 YOLOv5 训练...")
    subprocess.run(cmd, check=True)