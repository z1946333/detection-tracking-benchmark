import json
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib
import io

DATA_DIR = Path(r"D:\Datasets\processed_detection")
GT_JSON = DATA_DIR / "annotations" / "instances_test.json"
PRED_DIR = Path(r"D:\GitHub\detection-tracking-benchmark\evaluation\detection")

def evaluate_model(model_name, pred_file):
    print(f"\n{'='*50}")
    print(f"正在评估模型: {model_name}")
    print(f"{'='*50}")
    
    # 1. 加载真实标签
    coco_gt = COCO(str(GT_JSON))
    
    # 2. 加载预测结果
    with open(pred_file, 'r') as f:
        preds = json.load(f)
        
    if not preds:
        print(f"⚠️ {model_name} 没有预测结果，跳过评估。")
        return
        
    # 将预测结果加载为 COCO 对象
    coco_dt = coco_gt.loadRes(preds)
    
    # 3. 初始化 COCOeval
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # 运行评估
    coco_eval.evaluate()
    coco_eval.accumulate()
    
    # 为了只打印结果，不打印大段废话，我们重定向一下输出
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.summarize()
    
    # 提取关键指标
    stats = coco_eval.stats
    ap50_95 = stats[0] * 100
    ap50 = stats[1] * 100
    ap75 = stats[2] * 100
    ap_s = stats[3] * 100
    ap_m = stats[4] * 100
    ap_l = stats[5] * 100
    
    print(f"✅ AP@[.50:.95] : {ap50_95:.2f}")
    print(f"✅ AP@0.50      : {ap50:.2f}")
    print(f"✅ AP@0.75      : {ap75:.2f}")
    print(f"✅ AP_small     : {ap_s:.2f}" if ap_s != -1 else "✅ AP_small     : N/A")
    print(f"✅ AP_medium    : {ap_m:.2f}" if ap_m != -1 else "✅ AP_medium    : N/A")
    print(f"✅ AP_large     : {ap_l:.2f}" if ap_l != -1 else "✅ AP_large     : N/A")

def main():
    # 查找目录下所有的 *_predictions.json
    pred_files = list(PRED_DIR.glob("*_predictions.json"))
    if not pred_files:
        print("未找到任何预测 JSON 文件！")
        return
        
    for pf in pred_files:
        model_name = pf.name.replace("_predictions.json", "")
        evaluate_model(model_name, pf)

if __name__ == "__main__":
    main()