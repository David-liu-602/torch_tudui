import torch.nn
import torchvision
import os
# 更改预训练模型的下载路径
os.environ['TORCH_HOME'] = 'D://torch_home'


vgg16_False = torchvision.models.vgg16(weights=None)
vgg16_True = torchvision.models.vgg16(weights = 'DEFAULT')
# print(vgg16_False)
# print(vgg16_True)

train_data = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)

# "迁移学习"——在别人的基础上加层
# vgg16_True.classifier.add_module('add_linear', torch.nn.Linear(1000, 10))
# print(vgg16_True)


# 在别人的基础上修改
# vgg16_False.classifier[6] = torch.nn.Linear(4096, 10)
# print(vgg16_False)


# 保存预训练
vgg16 = torchvision.models.vgg16(weights=None)
torch.save(vgg16, "vgg16_method1.pth")       # 保存方式1：模型结构+模型参数
torch.save(vgg16.state_dict(), "vgg16_method2.pth")           # 保存方式2：模型参数（官方推荐）

# 加载预训练
model1 = torch.load("vgg16_method1.pth")
print(model1)

print("============================================================")

torchvision.models.vgg16(weights=None)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model2 = torch.load("vgg16_method2.pth")
print(vgg16)


