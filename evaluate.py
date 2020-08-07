from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
sys.path.append(r'/home/heroin/workspace/envs/bcolz')
sys.path.append(r'/home/heroin/workspace/envs/mxnet')
import bcolz
import mxnet as mx
from PIL import Image
from torchvision import transforms as trans

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.eval import evaluate
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})


def load_bin(path, rootdir, transform, image_size=[112,112]):
    if not rootdir.exists():
        rootdir.mkdir()
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = transform(img)
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data.shape)
    np.save(str(rootdir)+'_list', np.array(issame_list))
    return data, issame_list


class TorchInference(object):
    def __init__(self, model_path=None, device=None):
        """
        对TorchInference进行初始化

        Parameters
        ----------
        model_path : str
            pytorch模型的路径，推荐使用绝对路径
        """
        super().__init__()
        self.model_path = model_path
        self.device = device
        if self.model_path is None:
            print("please set pytorch model path!\n")
            exit(-1)
        self.session = None
        self.model_loader()

    def model_loader(self):
        if torch.__version__ < "1.0.0":
            print("Pytorch version is not  1.0.0, please check it!")
            exit(-1)
        if self.model_path is None:
            print("Please set model path!!")
            exit(-1)
        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # check_point = torch.load(self.checkpoint_file_path, map_location=self.device)
        # self.model = check_point['net'].to(self.device)
        self.session = torch.jit.load(self.model_path, map_location=self.device)
        # 如果模型为pytorch0.3.1版本，需要以下代码添加BN内的参数
        # for _, module in self.model._modules.items():
        #     recursion_change_bn(self.model)
        self.session.eval()

    def inference(self, x: torch.Tensor):
        """
        pytorch的推理
        Parameters
        ----------
        x : torch.Tensor
            pytorch模型输入

        Returns
        -------
        torch.Tensor
            pytorch模型推理结果
        """
        x = x.to(self.device)
        self.session = self.session.to(self.device)
        outputs = self.session(x)
        return outputs


class Evaluation(object):
    def __init__(self, root='', data_name=None, batch_inference=None):
        self.embeddings = None
        self.actual_issame = list()
        self.image_list = list()
        self.root = root
        self.data_name = data_name
        self.batch_inference = batch_inference
        self.config = {
            'width': 112,
            'height': 112,
            'color_format': 'RGB',
            'mean': [0.5, 0.5, 0.5],
            'stddev': [0.5, 0.5, 0.5],
            'divisor': 255.0,
            "feature_size": 512,
            "batch_size": 32,
            "hflip": True
        }

    def set_config(self, key: str, value):
        if key not in self.config:
            print("config key error! please check it!")
            exit()
        self.config[key] = value

    @staticmethod
    def get_val_pair(path, name):
        carray = bcolz.carray(rootdir=os.path.join(path, name), mode='r')
        issame = np.load('{}/{}_list.npy'.format(path, name))
        return carray, issame

    @staticmethod
    def normalize(nparray, order=2, axis=0):
        """Normalize a N-D numpy array along the specified axis."""
        norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
        return nparray / (norm + np.finfo(np.float32).eps)

    @staticmethod
    def l2_norm(input, axis=1):
        norm = torch.norm(input, 2, axis, True)
        output = torch.div(input, norm)
        return output

    def _pre_process(self, image: np.ndarray, hflip=False) -> np.ndarray:
        """对图像进行预处理

        Parameters
        ----------
        image : np.ndarray
            输入的原始图像，BGR格式，通常使用cv2.imread读取得到

        Returns
        -------
        np.ndarray
            原始图像经过预处理后得到的数组
        """
        # deprocess image
        image = np.transpose(image, [0, 2, 3, 1])
        image = image * 0.5 + 0.5

        if self.config['color_format'] == "RGB":
            image = image[:, :, :, ::-1]
        if hflip:
            image = image[:, :-1, :, :]
        # if self.config['width'] > 0 and self.config['height'] > 0:
        #     # image = cv2.resize(image, (self.config['width'], self.config['height']))
        #     image = cv2.resize(image, (128, 128))
        #     image = self.center_crop(image, self.config['width'], self.config['height'])
        input_image = (np.array(image, dtype=np.float32) / self.config['divisor'] - self.config['mean']) / self.config[
            'stddev']
        input_image = np.transpose(input_image, [0, 3, 1, 2])
        return input_image

    def _post_process(self, features):
        features = self.l2_norm(features)
        features = torch.squeeze(features)
        features_array = features.data.cpu().numpy()
        # features_array = self.normalize(features_array, axis=1)
        return features_array

    def batch_feature_extract(self, images):
        # features = model(batch_images)
        batch_images = self._pre_process(images)
        batch_images = torch.from_numpy(batch_images).float()
        output = self.batch_inference(batch_images)
        features_array = self._post_process(output)
        return features_array

    def feature_extract(self):
        person_batch = len(self.image_list) // self.config['batch_size']
        person_id_features = np.zeros([0, self.config['feature_size']], np.float)
        print("start feature extract...")
        for index in range(person_batch):
            if index % (person_batch // 10) == 0:
                print(f"process {index}/{person_batch}...")
            batch_images = np.stack(self.image_list[index * self.config['batch_size']:
                                                    (index + 1) * self.config['batch_size']])
            features_array = self.batch_feature_extract(batch_images)
            person_id_features = np.vstack((person_id_features, features_array))
        if len(self.image_list) % self.config['batch_size']:
            batch_images = np.stack(self.image_list[person_batch * self.config['batch_size']::])
            features_array = self.batch_feature_extract(batch_images)
            person_id_features = np.vstack((person_id_features, features_array))
        self.embeddings = person_id_features

    def evaluate(self):
        self.image_list, self.actual_issame = self.get_val_pair(self.root, self.data_name)
        self.feature_extract()
        tpr, fpr, accuracy, best_thresholds = evaluate(self.embeddings, self.actual_issame)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = torch.as_tensor(roc_curve/255.0)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor

def buffer_val(writer, db_name, acc, best_threshold, roc_curve_tensor, epoch):
    writer.add_scalar('{}_Accuracy'.format(db_name), acc, epoch)
    writer.add_scalar('{}_Best_Threshold'.format(db_name), best_threshold, epoch)
    writer.add_image('{}_ROC_Curve'.format(db_name), roc_curve_tensor, epoch)


def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    # plt.show()



def generate_data():
    """
    提取开源的lfw.bin等类型数据，生成lfw_list.npy等文件
    Returns
    -------

    """
    from pathlib import Path
    test_transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    path = Path(r"/mnt/data/FaceReid/faces_glintasia/lfw.bin")
    rootdir = Path(r"/home/heroin/dataset/lfw")
    load_bin(path, rootdir, test_transform)


if __name__ == '__main__':
    root = r"/mnt/data/datasets"
    data_name = r"lfw"
    root_working = os.path.split(os.path.realpath(__file__))[0]
    model_path = os.path.join(os.path.dirname(os.path.dirname(root_working)),
                              # r"checkpoints/face_reid/plr_osnet_246_2.1345_jit.pth")
                              # r"checkpoints/face_reid/backbone_ir50_ms1m_epoch120_jit.pth")
                              r"checkpoints/face_reid/backbone_ir50_asia_jit.pth")
    # r"checkpoints/face_reid/model_mobilefacenet_jit.pth")
    torch_inference = TorchInference(model_path)
    print("load model to inference success!")
    evalution = Evaluation(root, data_name, torch_inference.inference)
    # evalution.set_config('mean', [0.4914, 0.4822, 0.4465])
    # evalution.set_config('stddev',  [0.247, 0.243, 0.261])
    # evalution.set_config('divisor', 255.0)
    # evalution.set_config('feature_size', 512)
    tpr, fpr, accuracy, best_thresholds = evalution.evaluate()
    print(tpr, fpr, accuracy, best_thresholds)
    gen_plot(fpr, tpr)
