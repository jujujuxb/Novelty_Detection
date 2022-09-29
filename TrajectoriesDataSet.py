import torch
import torch.nn.functional as F
import glob
import os
from collections import OrderedDict
from sortedcontainers import SortedDict, SortedSet
import cv2
import numpy as np
import tqdm


class TrajectoryDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_dir="/home/juxiaobing/code/GraduationProject/CNN-VAE/data/T15/T15_images", labels: SortedSet = {1}):

        self.extract_labels = labels
        self.dataset_dir = dataset_dir
        self.length = 0
        self.img_labels_ = self.get_labels()
        self.img_dicts = {}
        self.read_imgs()

    def get_labels(self):
        self.length = 0
        class_dirs = os.path.join(self.dataset_dir, '*')
        paths = glob.glob(class_dirs)
        labels = []
        for path in paths:
            label = (int)(path.split('/')[-1])
            if label in self.extract_labels:
                img_paths = glob.glob(os.path.join(path, '*.jpg'))
                for img_path in img_paths:
                    labels.append([img_path, label])
                self.length += len(img_path)
        return labels

    def read_imgs(self):

        print("Load images .... ")

        for idx in tqdm.tqdm(range(self.length)):
            self.img_dicts[idx] = self.read_img(self.img_labels_[idx][0])

    def read_img(self, img_path, need_trasform=True):

        cvimg = cv2.imread(img_path)

        if need_trasform:
            cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)

        cvimg = cv2.resize(cvimg, dsize=(64, 64))

        cvimg = cv2.normalize(cvimg, 0, 1, norm_type=cv2.NORM_MINMAX)

        cvimg = np.array(cvimg, dtype=np.float32).transpose([2, 0, 1])

        return cvimg

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        if idx not in self.img_dicts:
            self.img_dicts[idx] = self.read_img(self.img_labels_[idx][0])
        return self.img_dicts[idx], self.img_labels_[idx][1]


if __name__ == '__main__':
    data = TrajectoryDataset(labels={1, 2, 3, 4})

    pass
