# ==============================================================================
# 👇 必须放在最顶端：解决 Windows 系统证书导致的 SSL 报错
import ssl
import certifi
import os

os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
# ==============================================================================

import torch
import torchvision
import json
import cv2
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

# 导入模型适配相关的类
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.ssd import SSDClassificationHead, SSDRegressionHead
from torchvision.models.detection import (
    SSD300_VGG16_Weights, 
    FasterRCNN_ResNet50_FPN_Weights, 
    RetinaNet_ResNet50_FPN_Weights
)

class PigDataset(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        ann_file = self.root / "annotations" / f"instances_{split}.json"
        with open(ann_file, 'r') as f:
            coco = json.load(f)
        self.imgs = coco["images"]
        self.anns = coco["annotations"]

    def __getitem__(self, idx):
        img_info = self.imgs[idx]
        img_path = img_info["file_name"]
        img_w, img_h = img_info["width"], img_info["height"]
        
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {img_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        boxes, labels = [], []
        for ann in self.anns:
            if ann["image_id"] == img_info["id"]:
                x, y, w, h = ann["bbox"]
                
                # 👇 核心修复 1：严格截断越界坐标，防止计算 Loss 时出现 nan
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(img_w, x + w), min(img_h, y + h)
                
                # 确保有效区域大于 1 个像素
                if x2 - x1 > 1 and y2 - y1 > 1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(ann["category_id"])

        # 如果该图片没有有效框，返回空 Tensor
        if len(boxes) == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64)
            }
        else:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64)
            }
            
        return img, target

    def __len__(self):
        return len(self.imgs)

def get_model(model_name, num_classes):
    if model_name == 'fasterrcnn':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
    elif model_name == 'retinanet':
        model = torchvision.models.detection.retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)
        num_anchors = model.anchor_generator.num_anchors_per_location()[0]
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=model.backbone.out_channels,
            num_anchors=num_anchors,
            num_classes=num_classes
        )
        
    elif model_name == 'ssd300':
        model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
        in_channels = [512, 1024, 512, 256, 256, 256]
        num_anchors = model.anchor_generator.num_anchors_per_location()
        
        model.head.classification_head = SSDClassificationHead(
            in_channels=in_channels, 
            num_anchors=num_anchors, 
            num_classes=num_classes
        )
        model.head.regression_head = SSDRegressionHead(
            in_channels=in_channels, 
            num_anchors=num_anchors
        )
        
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['fasterrcnn', 'retinanet', 'ssd300'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"正在使用设备: {device}")

    dataset = PigDataset(r"D:\Datasets\processed_detection", "train")
    loader = DataLoader(
        dataset, 
        batch_size=args.batch, 
        shuffle=True, 
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=0
    )

    model = get_model(args.model, num_classes=2).to(device)
    
    # 👇 核心修复 2：降低学习率，防止第一步就走飞
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    save_dir = Path(f"experiments/detection/{args.model}")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始正式训练 {args.model}，共 {args.epochs} 轮...")
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for i, (images, targets) in enumerate(loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # 检查是否出现 NaN 值
            if torch.isnan(losses):
                print(f"⚠️ 警告: 在第 {i} 步检测到 NaN Loss，跳过该批次！")
                optimizer.zero_grad()
                continue # 跳过这个坏样本
            
            optimizer.zero_grad()
            losses.backward()
            
            # 👇 核心修复 3：梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            
            epoch_loss += losses.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i}/{len(loader)}], Loss: {losses.item():.4f}")
        
        if len(loader) > 0 and epoch_loss > 0:
            avg_loss = epoch_loss / len(loader)
            print(f"✅ Epoch {epoch+1} 完成, 平均 Loss: {avg_loss:.4f}")
            
            torch.save(model.state_dict(), save_dir / "last.pt")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), save_dir / "best.pt")
                print(f"🌟 发现更低 Loss ({best_loss:.4f})，已保存 best.pt")

    print(f"训练彻底完成！模型保存在 {save_dir}")