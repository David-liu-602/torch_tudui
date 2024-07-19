import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter



class LCJ_conv(nn.Module):
    # ctrl + O : 重写方法         ctrl+I ：实现方法
    def __init__(self) :
        super(LCJ_conv, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


class LCJ_maxPool(nn.Module):

    def __init__(self) -> None:
        super(LCJ_maxPool, self).__init__()
        self.maxPool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxPool1(input)
        return output


class LCJ_Nonlinear(nn.Module):

    def __init__(self) -> None:
        super(LCJ_Nonlinear, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.sigmoid(input)
        return output




writer = SummaryWriter("./tensorboard/nn")



# ######     Conv卷积
# input = torch.tensor([
#     [1,2,0,3,1],
#     [0,1,2,3,1],
#     [1,2,1,0,0],
#     [5,2,3,1,1],
#     [2,1,0,1,1]
# ], dtype=torch.float32)

# kernel = torch.tensor([
#     [1,2,1],
#     [0,1,0],
#     [2,1,0]
# ])
#
# # torch.reshape(input, (channel, batch, H, W))
# input = torch.reshape(input, (1, 1, 5, 5))
# kernel = torch.reshape(kernel, (1, 1, 3, 3))
#
# # 参数：input   weight   bias   stride  padding   dilation    groups
# # padding是在输入数据上填充空0，由5*5变成6*6
# output = F.conv2d(input, kernel, stride=1, padding=0, dilation=1, groups=1)       # input有格式要求，所以要reshape
# print(output)


dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)

LCJ_c = LCJ_conv()
# print(LCJ_c)

# dataloader不支持整数索引,转成列表查看
# dataloader_list = list(dataloader)
# print(dataloader_list[1][0].shape)
# print(dataloader_list[1][1].shape)

# for data in dataloader:
#     imgs, targets =data
#     output = LCJ_c(imgs)
    # print(imgs.shape)
    # print(output.shape)





# pool：池化         降低特征图的空间维度、增加不变性和减少计算量, 控制过拟合
# 常见类型包括最大池化（Max Pooling）、平均池化（Average Pooling）、L2池化（L2 Pooling）、随机池化（Stochastic Pooling）和全局池化（Global Pooling）。
LCJ_Pool = LCJ_maxPool()
# output = LCJ_Pool(input)
# print(output)


# step = 0
# for data in dataloader:
#     imgs, targets =data
#     output = LCJ_Pool(imgs)
#     writer.add_images("max_Pool", imgs, step)
#     writer.add_images("output", output, step)
#     step += 1



# 非线性激活——ReLu,sigmoid
LCJ_Nonlinear_sigmoid = LCJ_Nonlinear()
step = 0
for data in dataloader:
    imgs, targets =data
    output = LCJ_Nonlinear_sigmoid(imgs)
    writer.add_images("sigmoid", imgs, step)
    writer.add_images("output_sigmoid", output, step)
    step += 1


# 写到此处开始卡了,另起文件

writer.close()