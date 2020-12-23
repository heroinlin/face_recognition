import os
import sys

from yacs.config import CfgNode as CN
workroot = os.path.split(os.path.realpath(__file__))[0]

def get_database(name):
    if name in ['Ms_celeb', 'default']:
        image_root = "/home/heroin/datasets/MS-Celeb-1M"
    elif name in ['lfw']:
        image_root = "/home/heroin/datasets/lfw"
    elif name in ['GlintAsia']:
        image_root = "/home/heroin/datasets/faces_glintasia_images_debug"
    else:
        print(f"The database {name} is not exist!")
        exit(-1)
    return image_root

Cfg = CN()

# 数据部分参数
Cfg.Database = CN()
Cfg.Database.name = 'GlintAsia'
Cfg.Database.image_size = [112, 112]
Cfg.Database.image_root = get_database(Cfg.Database.name)
Cfg.Database.val_image_root = get_database(Cfg.Database.name)
Cfg.Database.pickle_folder = None


# 采样部分参数
Cfg.Sample = CN()
Cfg.Sample.batch_size = 64
Cfg.Sample.num_workers = 16

# 训练部分参数
Cfg.Train = CN()
Cfg.Train.initial_lr = 0.001
Cfg.Train.lr = 0.001
Cfg.Train.weight_decay = 0.0005
Cfg.Train.epochs = 400
Cfg.Train.gpu_list = [0]
Cfg.Train.search_learn_rate = False
Cfg.Train.train = True
Cfg.Train.test = False
Cfg.Train.resume = False
Cfg.Train.checkpoint = os.path.join(workroot, "checkpoints/model_mobilefacenet.pth")
Cfg.Train.finetune = False
Cfg.Train.export = False

# 网络部分参数
Cfg.Net = CN()
Cfg.Net.backbone_name = 'MobileFaceNet'
Cfg.Net.head_name = 'ArcFace'
Cfg.Net.pretrained = True
Cfg.Net.feature_size = 512

# Loss函数参数
Cfg.Loss = CN()
Cfg.Loss.margin = 0.3
Cfg.Loss.list = ['FocalLoss']
Cfg.Loss.weight_list = [1.0]