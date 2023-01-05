import os
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from dataloader import FoodDatasetTrain
from models.resnet18_bb_model import Resnet18Model
from models.vanilla_model import VanillaModel
from options import Options

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    options = Options()
    args = options.parse()
    transform  = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    food_dataset_train = FoodDatasetTrain(args, transform=transform)

    if args.phase != 'train':
        print(f"Mismatch!! argparser has phase={args.phase}, but you are running file train.py")
        print("Exitting ... ")
        exit(0)

    dataloader = DataLoader(food_dataset_train, batch_size=args.batch_size, shuffle=False)
    if args.model_name == "resnet18":
        model = Resnet18Model(args)
    elif args.model_name == "vanilla":
        model = VanillaModel(args)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    now = datetime.datetime.now()
    timestamp=now.strftime("%Y_%m_%d_%H_%M_%S")

    LOGS_DIR = "logs"
    if not os.path.isdir(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    log_file = open(f"{LOGS_DIR}/{timestamp}_log.txt", "w")
    print("--------------------------------------------------------")
    print("Printing chosen argument values:")
    print("--------------------------------------------------------")
    options.print_options(args)
    print("--------------------------------------------------------")
    options.save_options(args, f"{LOGS_DIR}/{timestamp}_train_options.txt")

    for epoch in range(args.num_epoch):
        epoch_loss = 0
        num_samples = len(food_dataset_train)
        print("Num samples", num_samples)
        print(f"Starting epoch: {epoch+1}")
        log_file.write(f"Starting epoch: {epoch+1}\n")
        for idx, sample in enumerate(dataloader):
            img_path = sample['path']
            imgs = sample['image'].to(device)
            gt_labels = sample['label'].to(device)
            preds = model(imgs)
            loss = criterion(preds, gt_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss+= loss.item()
            # print(f"preds: {torch.arg max(preds, dim=1)}; gt: {gt_labels}, loss: {loss.item()}")

        print(f"Finished epoch: {epoch+1}, epoch loss: {epoch_loss}, average epoch loss: {epoch_loss/num_samples}")
        log_file.write(f"Finished epoch: {epoch+1}, epoch loss: {epoch_loss}, average epoch loss: {epoch_loss/num_samples}\n")
        torch.save(model.state_dict(), os.path.join(args.ckpts_dir, f"model_{epoch}.ckpt"))
    log_file.close()
