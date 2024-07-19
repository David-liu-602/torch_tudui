from torch.utils.data import Dataset
import cv2
from PIL import Image
import os


root_dir = "D:\\Desktop\\dataset\\hymenoptera_data\\train"

class MyData(Dataset):

    def __int__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


ants_label_dir = "ants"
ants_dataset = MyData(root_dir, ants_label_dir)


