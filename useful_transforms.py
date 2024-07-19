from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("D://Desktop//png//png//1.png")
print(img)


# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)


# Normalize
# output[channel] = (input[channel] - mean[channel]) / std[channel]
# (input-0.5)/0.5 = 2*input - 1                  input[0,1]   =>   result[-1,1]
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)

print(img_tensor[0][0][0])
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm,1)
writer.add_image("Normalize", img_tensor,2)       # 对比


# Resize        给两个参数就是缩放成正方形，给一个参数就是等比例缩放
print(img.size)
trans_resize = transforms.Resize((128,128))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)


# Compose      将多个变换组合组成一个序列
trans_resize_2 = transforms.Resize(128)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize",img_resize_2,1)


# RandomCrop
trans_random = transforms.RandomCrop(128)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("random_crop", img_crop, i)


writer.close()





