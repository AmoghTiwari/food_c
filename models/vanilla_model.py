import torch
import torch.nn as nn

class VanillaModel(nn.Module):
    def __init__(self, args):
        super(VanillaModel, self).__init__()
        h,w = args.target_h, args.target_w

        self.conv1 = nn.Conv2d(3,16,kernel_size=(3,3),padding='same')
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2))
        h,w = h//2, w//2

        self.conv2 = nn.Conv2d(16,64,kernel_size=(3,3),padding='same')
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2))
        h,w = h//2, w//2

        self.fc1 = nn.Linear(in_features=(h*w*64), out_features=args.num_labels)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        return x
