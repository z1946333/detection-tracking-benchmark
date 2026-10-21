from ultralytics import YOLO

if __name__ == '__main__':
    # 加载本地权重
    model = YOLO(r'D:\GitHub\detection-tracking-benchmark\yolov8n.pt')
    
    model.train(
        # 绝对路径指向刚刚新建的 yaml
        data=r'D:\GitHub\detection-tracking-benchmark\configs\detection\data.yaml',
        epochs=1,
        imgsz=640,
        batch=4,
        workers=0,
        device=0,
        seed=42,
        project='experiments/smoke',
        name='yolov8n_smoke'
    )