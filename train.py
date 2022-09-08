import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader import FoodDatasetTrain
from models.resnet18_bb_model import Resnet18Model
from models.vanilla_model import VanillaModel
from options import Options

import numpy as np

""" NOTES
- Dataloader needs an input of type Dataset
- To create an instance of type Dataset, we inherit from the in-built dataset class and then override its __getitem__
and __len__ functions
- Then we pass this to 

1. Inherit from Dataset class
2. getitem will take self and idx as input

"""
""" LAST MODIFIED 
-> 9 Aug 13:21
-> 26 Aug 04:48
-> 26 Aug 22:30
-> 6 Sep 16:32
-> 6 Sep 19:15
-> 7 Sep ~8 AM
-> 8 Sep 14:43
"""

""" NEXT STEPS
    6 Sep:
        1. Do the one-hot encoding thing: Done (6 Sep)
    6 Sep: 
        1. Add backprop step: Done (7 sep)
    7 Sep:
        1. Loss is diverging. Fix that

"""

if __name__ == "__main__":
    args = Options().parse()

    target_image_size = (args.target_h, args.target_w)
    food_dataset_train = FoodDatasetTrain(args.dataroot, phase=args.phase, target_image_size=target_image_size)

    if args.phase != 'train':
        print(f"Mismatch!! argparser has phase={args.phase}, but you are running file train.py")
        print("Exitting ... ")
        exit()

    dataloader = DataLoader(food_dataset_train, batch_size=args.batch_size, shuffle=True)
    if args.model_name == "resnet18_bb":
        model = Resnet18Model(args)
    elif args.model_name == "vanilla":
        model = VanillaModel(args)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(args.num_epoch):
        running_loss = 0
        num_samples = len(dataloader)
        for idx, sample in enumerate(dataloader):
            img_path = sample['path']
            imgs = sample['image']
            gt_labels = sample['label']
            pred = model(imgs)
            pred_labels=torch.argmax(pred, axis=1)
            # print(f"gt: {gt_labels}, pred:{pred_labels}")
            loss = criterion(pred, gt_labels)
            loss.backward()
            optimizer.step()
            running_loss+= loss.item()

        print(f"Epoch: {epoch+1}, running loss: {running_loss}")
