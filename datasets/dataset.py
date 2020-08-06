from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os

import cv2
import numpy as np
import torch.utils.data as data
from .transform import opencv_transforms as cv_transforms


class Dataset(data.Dataset):
    def __init__(self, image_root='data', cur_id=0):
        super(Dataset, self).__init__()
        self.image_root = image_root
        self.database_name = "dataset"
        self.cur_id = cur_id
        self.person_id_container = list()
        self.person_id2label = {}
        self.data_list = list()
        self.transforms = cv_transforms.Compose([
            cv_transforms.RandomRotation(30),
            cv_transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0),
            cv_transforms.ToTensor(),
            cv_transforms.ColorAugmentation(),
            cv_transforms.RandomErasing(sl=0.02, sh=0.2),
            cv_transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self._prepare(self.image_root )
        self.person_id2label = {idx: label + self.cur_id for label, idx in enumerate(self.person_id_container)}

    def __len__(self):
        return len(self.data_list)

    def class_num(self):
        return len(self.person_id_container)

    def _prepare(self, image_dir, direction=0):
        for image_path in glob.glob(image_dir + "/*/*.jpg"):
            person_id = os.path.basename(os.path.dirname(image_path))
            if int(person_id) == -1:
                continue  # junk images are just ignored
            person_id = '_'.join([self.database_name, person_id])
            if person_id not in self.person_id_container:
                self.person_id_container.append(person_id)
            data_dict = {
                "image_path": image_path,
                "person_id": person_id,
                "direction": direction,  # 标记数据用于训练还是测试的gallery或query
                "database_name": self.database_name
            }
            self.data_list.append(data_dict)

    def __getitem__(self, index):
        data_index = self.data_list[index]
        image_path = data_index['image_path']
        person_id = data_index['person_id']
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8()), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(112, 112))
        img = self.transforms(img)
        target = self.person_id2label[person_id]
        return img, target


def test():
    image_root = r"/mnt/data/public/FaceReid/face_reid"
    train_dataset = Dataset(image_root)
    train_data_loader = data.DataLoader(dataset=train_dataset,
                                        batch_size=64,
                                        shuffle=False,
                                        num_workers=0,
                                        drop_last=True
                                        )
    print(train_dataset.person_id_container)
    for images, targets in train_data_loader:
        # print(images.data.cpu().numpy()[:, :, 0:10, 0:10])
        print(targets)
        exit(1)


if __name__ == '__main__':
    test()
