import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter



class LCJ_linear(nn.Module):
    # ctrl + O : 重写方法         ctrl+I ：实现方法
    def __init__(self) :
        super(LCJ_linear, self).__init__()
        self.linear = nn.Linear(196608, 10)

    def forward(self, x):
        x = self.linear(x)
        return x





writer = SummaryWriter("./tensorboard/nn1")



dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)


Lcj_linear = LCJ_linear()
step = 0
for data in dataloader:
    imgs, targets =data

    # 等效的展平操作
    imgs = torch.reshape(imgs,(1, 1, 1, -1))
    # imgs = torch.flatten(imgs)

    output = Lcj_linear(imgs)
    writer.add_images("linear", imgs, step)
    writer.add_images("output", output, step)
    step += 1













writer.close()