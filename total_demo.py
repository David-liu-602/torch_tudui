import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model_CIFRA10 import *

# 添加tensorboard                                     加载数据集
writer = SummaryWriter("./tensorboard/total_net")
train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集长度为{}".format(train_data_size))          # ctrl+D：复制上一行
print("测试数据集长度为{}".format(test_data_size))

train_dataloader = DataLoader(train_data, 16)
test_dataloader = DataLoader(test_data,16)

# 实例化网络
lcj_CIFRA10 = total_demo()

if torch.cuda.is_available():
    lcj_CIFRA10 = lcj_CIFRA10.cuda()

# 定义损失函数
loss_func = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_func = loss_func.cuda()

# 定义优化器
learning_rate = 0.01
# learning_rate = 1e-2
opti = torch.optim.SGD(lcj_CIFRA10.parameters(), lr=learning_rate)


total_train_step = 0
total_test_step = 0
epoch = 10

# 预训练
for i in range(epoch):
    print("------------第{}轮训练-------------".format(i + 1))

    # 训练步骤开始
    lcj_CIFRA10.train()
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = lcj_CIFRA10(imgs)
        loss = loss_func(outputs, targets)

        # 优化器
        opti.zero_grad()
        loss.backward()
        opti.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
#         加了item不会输出tensor

    # 测试部分
    lcj_CIFRA10.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()         # .cuda加速
            outputs = lcj_CIFRA10(imgs)
            loss = loss_func(outputs, targets)
            total_test_loss += loss

            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy


    print("整体测试集的loss：{}".format(total_test_loss))
    print("整体测试集的accuracy：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(lcj_CIFRA10, "lcj_{}.pth".format(i))
    print("模型已保存")



writer.close()