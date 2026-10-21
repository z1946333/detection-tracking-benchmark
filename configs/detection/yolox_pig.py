import os
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.basename(os.path.realpath(__file__)).split(".")[0]
        
        self.num_classes = 1
        self.max_epoch = 100
        self.data_num_workers = 0
        self.eval_interval = 5
        
        # 数据路径
        self.data_dir = r"D:\Datasets\processed_detection"
        self.train_ann = "instances_train.json"
        self.val_ann = "instances_val.json"