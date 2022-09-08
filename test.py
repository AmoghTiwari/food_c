# REFERENCE: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

import numpy as np
from torch.utils.data import DataLoader
from dataloader import FoodDatasetTest
from options import Options
from models.resnet18_bb_model import Model

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
"""

""" NEXT STEPS
    1. Resize all to same size: Done
    2. Apply transforms
    3. Write model
        3.1 Use an available model
"""

if __name__ == "__main__":
    args = Options().parse()

    target_image_size = (args.target_h, args.target_w)
    food_dataset_test = FoodDatasetTest(args.dataroot, phase=args.phase, target_image_size=target_image_size)

    if args.phase != 'test':
        print(f"Mismatch!! argparser has phase={args.phase}, but you are running file test.py")
        print("Exitting ...")
        exit(0)

    dataloader = DataLoader(food_dataset_test, batch_size=1, shuffle=True)
    model = Model(args.num_labels)

    num_samples = len(food_dataset_test)
    for idx, sample in enumerate(dataloader):
        img_path = sample['path']
        imgs = sample['image']
        pred = model(imgs)
        pred_labels=np.argmax(pred.detach().cpu().numpy(), axis=1)
        print("pred idxs", pred_labels)
        print("pred_labels", food_dataset_test.idx2label(pred_labels))
