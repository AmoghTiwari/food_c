import torch.nn as nn
from torchvision.models.resnet import resnet18

class Resnet18Model(nn.Module):
    def __init__(self, NUM_LABELS) -> None:
        super(Resnet18Model, self).__init__()
        self.model = resnet18()
        self.model = self.freeze_layers()
        self.model.fc = nn.Linear(self.model.fc.in_features, NUM_LABELS)

    def freeze_layers(self):
        for name, params in self.model.named_parameters():
            params.requires_grad=False
        return self.model

    def forward(self, x):
        return self.model(x)
