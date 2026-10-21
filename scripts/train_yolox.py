import subprocess
import sys

if __name__ == '__main__':
    cmd = [
        sys.executable, "third_party/YOLOX/tools/train.py",
        "-f", r"D:\GitHub\detection-tracking-benchmark\configs\detection\yolox_pig.py",
        "-d", "1",
        "-b", "2",
        "--fp16",
        "-o"
    ]
    print("正在启动 YOLOX 训练...")
    subprocess.run(cmd, check=True)