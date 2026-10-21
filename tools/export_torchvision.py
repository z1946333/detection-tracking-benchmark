import ssl
import certifi
import os
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import torch
import torchvision
import json
import cv2
import argparse
from pathlib import Path
import torchvision.transforms.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.ssd import SSDClassificationHead, SSDRegressionHead

def get_model(model_name, num_classes):
    if model_name == 'fasterrcnn':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    elif model_name == 'retinanet':
        model = torchvision.models.detection.retinanet_resnet50_fpn(weights=None)
        num_anchors = model.anchor_generator.num_anchors_per_location()[0]
        model.head.classification_head = RetinaNetClassificationHead(model.backbone.out_channels, num_anchors, num_classes)
    elif model_name == 'ssd300':
        model = torchvision.models.detection.ssd300_vgg16(weights=None)
        in_channels = [512, 1024, 512, 256, 256, 256]
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
        model.head.regression_head = SSDRegressionHead(in_channels, num_anchors)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    DATA_DIR = Path(r"D:\Datasets\processed_detection")
    TEST_JSON = DATA_DIR / "annotations" / "instances_test.json"
    OUT_DIR = Path(r"D:\GitHub\detection-tracking-benchmark\evaluation\detection")
    
    with open(TEST_JSON, 'r') as f:
        coco_data = json.load(f)
    img_name_to_id = {img['file_name']: img['id'] for img in coco_data['images']}

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(args.model, num_classes=2)
    model.load_state_dict(torch.load(f"experiments/detection/{args.model}/best.pt"))
    model.to(device).eval()

    img_dir = DATA_DIR / "images" / "test"
    img_paths = list(img_dir.glob("*.jpg"))
    results_list = []

    print(f"开始推理 {args.model}...")
    with torch.no_grad():
        for img_path in img_paths:
            abs_path = str(img_path.absolute())
            img = cv2.imread(abs_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = F.to_tensor(img_rgb).to(device)
            
            outputs = model([img_tensor])[0]
            image_id = img_name_to_id.get(abs_path)
            
            for box, score, label in zip(outputs['boxes'], outputs['scores'], outputs['labels']):
                if score < 0.001: continue
                x1, y1, x2, y2 = box.tolist()
                results_list.append({
                    "image_id": image_id,
                    "category_id": int(label.item()),
                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                    "score": round(score.item(), 4)
                })

    with open(OUT_DIR / f"{args.model}_predictions.json", 'w') as f:
        json.dump(results_list, f)
    print("✅ 导出完成！")