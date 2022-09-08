import sys
sys.path.append("./")

from dataloader import FoodDatasetTrain
from options import Options

LABELS_TXT_FILE_PATH = "../data/labels.txt"
args = Options().parse()
target_image_size = (args.target_h, args.target_w)
food_dataset = FoodDatasetTrain(args.dataroot, phase='train', target_image_size=target_image_size)
num_samples = len(food_dataset)
img_label_list = []

for idx, sample in enumerate(food_dataset):
    print(f"Processing sample: {idx+1}/{num_samples}")
    img_label = sample['label']
    img_label_list.append(img_label)
    print(img_label)
    img_label_list.append(img_label)

img_label_set = set(img_label_list)
unique_labels = list(img_label_set)

with open(LABELS_TXT_FILE_PATH, "w") as f:
    for label in unique_labels:
        f.write(label)
        f.write("\n")