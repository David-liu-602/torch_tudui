import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader


class total_demo(nn.Module):

    def __init__(self) -> None:
        super(total_demo, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# if __name__ == '__main__':
#     CIFRA10_net = total_demo()
#     input = torch.ones((64, 3, 32, 32))
#     output = CIFRA10_net(input)
#     print(output.shape)
