
import torch.nn as nn
# NN Model
class SimpleModel(nn.Module):
    def __init__(self, num_classes=10,convLayerSize = 30):
        super(SimpleModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1,convLayerSize,5,padding=2,stride=2),
            nn.LeakyReLU(0.001),
            # nn.Conv2d(40,60,3,padding=1,stride=1),
            # nn.LeakyReLU(0.001),
            nn.BatchNorm2d(convLayerSize),
            nn.MaxPool2d(2,2),
            # nn.ReLU(),
            nn.Flatten(),
            nn.Linear(convLayerSize*7*7,100),
            nn.LeakyReLU(0.001),
            nn.Linear(100,num_classes),
            nn.Softmax(1)
        )

    def forward(self, x):
        return self.layers(x)