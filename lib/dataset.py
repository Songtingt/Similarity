import numpy as np
from PIL import Image
from PIL import ImageFilter
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time
import random
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
from lib.utils import get_root_logger


class MyDataset(Dataset):
    def __init__(
            self,
            label_train='/data/home/songtt/work/map2optical/train_random.txt',
            label_valid='/data/home/songtt/work/map2optical/valid_random.txt',
            label_test='/data/home/songtt/work/map2optical/test.txt',
            domainA_small_file='/data/home/songtt/work/map2optical/map_512',  # small
            domainB_big_file='/data/home/songtt/work/map2optical/optical',  # big
            mode='Train'
    ):

        self.mode = mode
        self.logger = get_root_logger()

        if self.mode == 'Train':
            self.label_path = label_train
            self.transform_domainA = transforms.Compose([
                transforms.RandomChoice([
                    transforms.ColorJitter(brightness=0.1),  # 加一些色彩抖动 不要翻转的
                    transforms.ColorJitter(contrast=0.1),
                    transforms.ColorJitter(saturation=0.1)
                ]),

                transforms.ToTensor(),  # /255 归一化到01之间
                transforms.Normalize(mean=0.4, std=0.5)  # 标准化，使得分布在-1,1之间
            ])
            self.transform_domainB = transforms.Compose([
                transforms.RandomChoice([
                    transforms.ColorJitter(brightness=0.1),  # 加一些色彩抖动 不要翻转的
                    transforms.ColorJitter(contrast=0.1),
                    transforms.ColorJitter(saturation=0.1)
                ]),

                transforms.ToTensor(),
                transforms.Normalize(mean=0.4, std=0.5)
            ])

        else:
            self.label_path = label_valid
            self.transform_domainA = transforms.Compose([
                transforms.ToTensor()]
            )
            self.transform_domainB = transforms.Compose([
                transforms.ToTensor()]
            )

        self.img_domainA_small_path = []
        self.img_domainB_big_path = []
        self.label = []
        self.class_id = []
        self.file_id = []
        f = open(self.label_path, 'r')
        try:
            lines = f.readlines()
            for line in lines:
                lineData = line.split()
                domainA_small_path = os.path.join(domainA_small_file, lineData[1])
                domainB_big_path = os.path.join(domainB_big_file, lineData[0])
                
                self.img_domainA_small_path.append(domainA_small_path)
                self.img_domainB_big_path.append(domainB_big_path)
                self.class_id.append(int(lineData[0][0]))
                self.file_id.append(int(lineData[0][:-4].split("_")[-1]))
                self.label.append([int(lineData[2]), int(lineData[3])])
        finally:
            f.close()
        self.logger.info("dataset load ok, mode:{} len:{}".format(self.mode, len(self.img_domainA_small_path)))

    def __getitem__(self, item):
        domainA_small_path = self.img_domainA_small_path[item]
        domainB_big_path = self.img_domainB_big_path[item]

        location = self.label[item]
        class_id = self.class_id[item]
        file_id = self.file_id[item]

        domainA_small = Image.open(domainA_small_path).convert('L')  # 512,512 change to single channel
        domainB_big = Image.open(domainB_big_path).convert('L')  # 600,600

        pair1 = self.transform_domainA(domainA_small)  # (1,1,512, 512)
        pair2 = self.transform_domainB(domainB_big)  # (1,1,600, 600)
        loc = torch.tensor((location[0], location[1]))

        return {'pair_1': pair1, 'pair_2': pair2, 'label': loc, 'class_id': class_id, 'file_id': file_id}

    def __len__(self):
        return len(self.label)



if __name__ == "__main__":
    ds = MyDataset()
    # ds[0]
    training_dataloader = DataLoader(  # 应该是这里面对training_dataset[idx] 会默认调用_getitem_() 和_len_
        ds,
        batch_size=16,  # batch_size=1  将数据集划分成batch块，每块batch_size个数据
        shuffle=False
    )
    ds.evaluate(3)
    progress_bar = tqdm(enumerate(training_dataloader), total=len(training_dataloader))
    for batch_idx, (batch) in progress_bar:  # 数据集被划分成了batch块
        # print(batch['pair_1'].shape) #b,3,256,256

        print(batch['pair_2'].shape)  # b,1,800,800
        print(batch['label'].shape)  # b,2  torch.Size([16, 2])
        print(batch['class_id'].shape)  # b,2  torch.Size([16, 2])
