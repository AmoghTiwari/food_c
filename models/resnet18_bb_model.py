import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18

class Resnet18Model(nn.Module):
    def __init__(self, args):
        super(Resnet18Model, self).__init__()
        self.model = resnet18()
        # self.model = self.freeze_layers()
        self.model.fc = nn.Linear(self.model.fc.in_features, args.num_labels)
        # self.fc = nn.Linear(self.model.fc.in_features, args.num_labels)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def freeze_layers(self):
        for name, params in self.model.named_parameters():
            params.requires_grad=False
        return self.model

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        x = torch.softmax(x, axis=1)
        return x
