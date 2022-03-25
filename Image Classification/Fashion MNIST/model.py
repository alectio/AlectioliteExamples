import torch.nn as nn
import torch.nn.functional as F


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 3, kernel_size=(3, 3), stride=1, padding=0)

        self.conv2 = nn.Conv2d(3, 6, kernel_size=(4, 4), stride=1, padding=0)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0)

        self.fullCon1 = nn.Linear(in_features=6 * 11 * 11, out_features=360)

        self.fullCon2 = nn.Linear(in_features=360, out_features=100)

        self.fullCon3 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(F.relu(self.conv2(x)))
        x = x.view(-1, 6 * 11 * 11)
        x = F.relu(self.fullCon1(x))
        x = F.relu(self.fullCon2(x))
        x = self.fullCon3(x)
        return x