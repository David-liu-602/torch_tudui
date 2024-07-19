import torch

input = torch.tensor([1, 8, 3], dtype=torch.float32)
target = torch.tensor([7, 2, 4], dtype=torch.float32)

# loss对输入形式任意，这里只是习惯性改一下
input = torch.reshape(input, (1, 1, 1, 3))
target = torch.reshape(target, (1, 1, 1, 3))

# 绝对值误差
loss1 = torch.nn.L1Loss(reduction='sum')
result = loss1(input, target)
print(result)

# 平方差
loss2 = torch.nn.MSELoss(reduction='sum')
result2 = loss2(input, target)
print(result2)

# 交叉熵
x = torch.tensor((0.1, 0.2, 0.3))
y = torch.tensor(1.)
x = torch.reshape(x, (1, 3))
loss3 = torch.nn.CrossEntropyLoss()
result3 = loss3(x, y)
print(result3)

