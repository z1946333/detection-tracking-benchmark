import cv2
from pathlib import Path

# 注意：这里换成你 D:\Datasets\processed_detection\images\train 里真实存在的一个文件名
# 比如你看到有 100.jpg，就改成 100.jpg
IMG_PATH = Path(r"D:\Datasets\processed_detection\images\train\100.jpg") 
TXT_PATH = Path(r"D:\Datasets\processed_detection\labels\train\100.txt")

if not IMG_PATH.exists():
    print(f"找不到图片：{IMG_PATH}，请换一个真实存在的文件名！")
    exit()

img = cv2.imread(str(IMG_PATH))
h, w, _ = img.shape

with open(TXT_PATH, "r") as f:
    for line in f:
        parts = line.strip().split()
        cls = int(parts[0])
        x_c, y_c, bw, bh = map(float, parts[1:])
        
        # 反归一化，把 0-1 的小数还原成像素坐标
        xmin = int((x_c - bw/2) * w)
        ymin = int((y_c - bh/2) * h)
        xmax = int((x_c + bw/2) * w)
        ymax = int((y_c + bh/2) * h)
        
        # 画绿框
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(img, f"pig", (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("Check Label", img)
print("按任意键关闭图片窗口...")
cv2.waitKey(0)