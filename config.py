import sys

from yacs.config import CfgNode as CN


def get_database(name):
    if name in ['Ms_celeb', 'default']:
        image_root = "/mnt/data/linjian/datasets/MS-Celeb-1M"
    else:
        print(f"The database {name} is not exist!")
        exit(-1)

Cfg = CN()

# 数据部分参数
Cfg.Database = CN()
Cfg.Database.name = 'default'
Cfg.Database.image_size = [112, 112]
Cfg.Database.image_root = get_database(Cfg.Database.name)
# Cfg.Database.pickle_folder = None


# 采样部分参数
Cfg.Sample = CN()
Cfg.Sample.sample_batch_size = 64
Cfg.Sample.sample_num_instance = 4
Cfg.Sample.batch_size = 256
Cfg.Sample.num_workers = 16

# 训练部分参数
Cfg.Train = CN()
Cfg.Train.initial_lr = 0.0002
Cfg.Train.lr = 0.0002
Cfg.Train.weight_decay = 0.0005
Cfg.Train.center_initial_lr = 0.0001
Cfg.Train.center_lr = 0.0001
Cfg.Train.epochs = 400
Cfg.Train.gpu_list = [8, 9]
Cfg.Train.search_learn_rate = False
Cfg.Train.train = True
Cfg.Train.test = False
Cfg.Train.resume = False
Cfg.Train.checkpoint = ''
Cfg.Train.finetune = False
Cfg.Train.export = False

# 网络部分参数
Cfg.Net = CN()
Cfg.Net.name = 'MobileFaceNet'
Cfg.Net.pretrained = True
Cfg.Net.feature_size = 512

# Loss函数参数
Cfg.Loss = CN()
Cfg.Loss.margin = 0.3
Cfg.Loss.list = ['FocalLoss']
Cfg.Loss.weight_list = [1.0]