from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

img_path = "D://Desktop//dataset//hymenoptera_data//train//ants//0013035.jpg"
img = Image.open(img_path)

# ctrl+p：显示函数所需参数
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)


writer = SummaryWriter("logs")

writer.add_image("Tensor_img", tensor_img)
writer.close()
