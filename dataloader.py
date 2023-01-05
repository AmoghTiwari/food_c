# REFERENCE: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
import os
import cv2
import json
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.nn.functional import one_hot

from options import Options

def change_key_type(in_dict):
    """JSON doesn't allow storing int values as keys.
    This function changes data type of the keys of the dict from int to str
    """
    out_dict = {}
    for k,v in in_dict.items():
        out_dict[int(k)] = v
    return out_dict

class FoodDatasetTrain(Dataset):
    def __init__(self, args, transform):
        self.transform = transform
        self.data_dir = args.dataroot
        self.phase = args.phase
        self.target_h = args.target_h
        self.target_w = args.target_w
        self.df = pd.read_csv(os.path.join(self.data_dir, f"{self.phase}.csv"))
        self.data = self.df.to_numpy()
        self.to_tensor = ToTensor()
        self.num_classes = 0
        self.labels2idx_file = args.labels2idx_file
        self.idx2labels_file = args.idx2labels_file

        with open(self.labels2idx_file) as f:
            self.labels2idx = json.load(f)
            self.num_classes = args.num_labels
        with open(self.idx2labels_file) as f:
            self.idx2labels = change_key_type(json.load(f))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, f"{self.phase}_images", self.data[idx,0])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.target_w, self.target_h)) # Remember, this is supposed to be (w,h) and not (h,w)
        img = self.to_tensor(img)
        img = self.transform(img)
        img_label = self.labels2idx[self.data[idx,1]]
        img_label = one_hot(torch.tensor(img_label), self.num_classes).to(torch.float32)
        return {"image": img, "label": img_label, "path": img_path}
        
    
class FoodDatasetTest(Dataset):
    def __init__(self, data_dir, phase, target_image_size=(224,224)):
        args = Options().parse()
        self.data_dir = data_dir
        self.phase = phase
        self.target_h, self.target_w = target_image_size
        self.df = pd.read_csv(os.path.join(data_dir, f"{self.phase}.csv"))
        self.data = self.df.to_numpy()
        self.to_tensor = ToTensor()
        self.labels2idx_file = args.labels2idx_file
        self.idx2labels_file = args.idx2labels_file

        with open(self.labels2idx_file) as f:
            self.labels2idx = json.load(f)
        with open(self.idx2labels_file) as f:
            self.idx2labels = change_key_type(json.load(f))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, f"{self.phase}_images", self.data[idx,0])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.target_w, self.target_h)) # Remember, this is supposed to be (w,h) and not (h,w)
        img = self.to_tensor(img)
        return {"image": img, "path": img_path}
    
    def idx2label(self, pred_idx):
        labels = []
        for i in range(len(pred_idx)):
            labels.append(self.idx2labels[pred_idx[i]])
        return labels
