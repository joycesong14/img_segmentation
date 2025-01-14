import pdb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import pandas
import random
from typing import Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import kaiming_normal_ as he_normal

#first split the data into train and test data
def read_data(path):
    data_df = pandas.read_csv(path)
    data = data_df.to_dict("index") #index -> {column -> value}
    indexes = list(data.keys())
    random.shuffle(indexes)
    train_idx = indexes[:100]
    eval_idx = indexes[100:]
    train_data = [data[idx] for idx in train_idx]
    eval_data = [data[idx] for idx in eval_idx]
    return train_data, eval_data

#start preparing to get the data
class MontData(Dataset):
    def __init__(self, data, parent_path, img_size = (512,512)):
        self.data = data
        self.parent_path = parent_path
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def read_img(self, path):
        #resize the image
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.img_size)
        #rescale the image
        img = np.float32(img/255)
        #expand the dimension: add a channel
        img = np.expand_dims(img, axis = 0)
        return img


    def __getitem__(self, index):
        dict_instance = self.data[index]
        img_path = os.path.join(self.parent_path, dict_instance['image'])
        left_path = os.path.join(self.parent_path, dict_instance['mask_left'])
        right_path = os.path.join(self.parent_path, dict_instance['mask_right'])
        # read the images
        img = self.read_img(img_path)
        left_mask = self.read_img(left_path)
        right_mask = self.read_img(right_path)
        mask = np.concatenate([left_mask, right_mask], axis = 0)
        return img, mask
