import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter



dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataLoader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)


class LCJ_seq(nn.Module):

    def __init__(self) -> None:
        super(LCJ_seq, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


lcj_seq = LCJ_seq()
# print(lcj_seq)
# input = torch.ones((64, 3, 32, 32))
# output = lcj_seq(input)
# print(output.shape)


# loss + backward + optimizer
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(lcj_seq.parameters(), lr=0.01)
for epochs in range(20):
    running_loss = 0.0
    for data in dataLoader:
        imgs, targets = data
        outputs = lcj_seq(imgs)
        result_loss = loss(outputs, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss += result_loss
    print(running_loss)








# writer = SummaryWriter("./tensorboard/sequential")
# writer.add_graph(lcj_seq, input)
# writer.close()