import torch
import numpy as np
from torch.utils.data import DataLoader
from dataloader import FoodDatasetTest
from options import Options
from models.resnet18_bb_model import Resnet18Model
from models.vanilla_model import VanillaModel

if __name__ == "__main__":
    args = Options().parse()
    target_image_size = (args.target_h, args.target_w)
    food_dataset_test = FoodDatasetTest(args.dataroot, phase=args.phase, target_image_size=target_image_size)    
    print("phase", args.phase)
    if args.phase != 'test':
        print(f"Mismatch!! argparser has phase={args.phase}, but you are running file test.py")
        print("Exitting ...")
        exit(0)
    
    dataloader = DataLoader(food_dataset_test, batch_size=args.batch_size, shuffle=True)
    if args.model_name == "resnet18_bb":
        model = Resnet18Model(args)
    elif args.model_name == "vanilla":
        model = VanillaModel(args)
    model.load_state_dict(torch.load("ckpts/model_5.ckpt"))
    model.eval()
    num_samples = len(food_dataset_test)
    for idx, sample in enumerate(dataloader):
        img_path = sample['path']
        imgs = sample['image']
        pred = model(imgs)
        pred_labels=np.argmax(pred.detach().cpu().numpy(), axis=1)
        print("pred idxs", pred_labels)
        print("pred_labels", food_dataset_test.idx2label(pred_labels))
