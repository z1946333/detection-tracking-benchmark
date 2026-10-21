from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载我们刚才测通的模型（这里用预训练的，或者用刚才冒烟测试跑出来的也可以，通常用官方预训练更好）
    model = YOLO(r'D:\GitHub\detection-tracking-benchmark\yolov8n.pt')
    
    # 2. 正式训练
    model.train(
        data=r'D:\GitHub\detection-tracking-benchmark\configs\detection\data.yaml',
        epochs=100,                         # 正式跑 100 轮
        imgsz=640,
        batch=4,                            # 3050显卡 4GB 显存，保持 4 最稳
        workers=4,                          # 可以开多线程加载数据了
        device=0,
        seed=42,                            # 固定种子
        # 下面这行非常重要，防止它跑到默认的 runs 文件夹里，这里强制存到 experiments 里
        project=r'D:\GitHub\detection-tracking-benchmark\experiments\detection',
        name='yolov8n',
        exist_ok=True
    )