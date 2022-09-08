# REFERENCE: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision.transforms import ToTensor
import cv2
import json

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
    def __init__(self, data_dir, phase, target_image_size=(224,224)):
        args = Options().parse()
        self.data_dir = data_dir
        self.phase = phase
        self.target_h, self.target_w = target_image_size
        df = pd.read_csv(os.path.join(data_dir, f"{self.phase}.csv"))
        self.data = df.to_numpy()
        self.to_tensor = ToTensor()
        self.labels2idx_file = args.labels2idx_file
        self.idx2labels_file = args.idx2labels_file

        with open(self.labels2idx_file) as f:
            self.labels2idx = json.load(f)
        with open(self.idx2labels_file) as f:
            self.idx2labels = change_key_type(json.load(f))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, f"{self.phase}_images", self.data[idx,0])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.target_w, self.target_h)) # Remember, this is supposed to be (w,h) and not (h,w)
        img = self.to_tensor(img)
        img_label = self.labels2idx[self.data[idx,1]]
        return {"image": img, "label": img_label, "path": img_path}
        
    
class FoodDatasetTest(Dataset):
    def __init__(self, data_dir, phase, target_image_size=(224,224)):
        args = Options().parse()
        self.data_dir = data_dir
        self.phase = phase
        self.target_h, self.target_w = target_image_size
        df = pd.read_csv(os.path.join(data_dir, f"{self.phase}.csv"))
        self.data = df.to_numpy()
        self.to_tensor = ToTensor()
        self.labels2idx_file = args.labels2idx_file
        self.idx2labels_file = args.idx2labels_file

        with open(self.labels2idx_file) as f:
            self.labels2idx = json.load(f)
        with open(self.idx2labels_file) as f:
            self.idx2labels = change_key_type(json.load(f))

    def __len__(self):
        return len(self.data)

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