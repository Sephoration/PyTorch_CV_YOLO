import os

import torch
import torchvision.transforms
from PIL import Image
from torch.utils.data import Dataset

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(32),
    torchvision.transforms.CenterCrop((32,32)),
    torchvision.transforms.ToTensor()
])
class myData(Dataset):
    def __init__(self, root_dir, label_dir, transform = None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.img_list = os.listdir(self.path)
        self.transform = transform
        # print(self.img_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        # print(img_item_path)
        img = Image.open(img_item_path)
        # img.show()
        img = self.transform(img)
        label = torch.tensor(int(self.label_dir))

        return img, label



train_root_dir = "./datasets/ant_bee/train"
train_ants_label_dir = "0"
train_bees_label_dir = "1"
train_ants_dataset = myData(train_root_dir, train_ants_label_dir, data_transform)
train_bees_dataset = myData(train_root_dir, train_bees_label_dir, data_transform)
train_data = train_ants_dataset + train_bees_dataset

val_root_dir = "./datasets/ant_bee/val"
val_ants_label_dir = "0"
val_bees_label_dir = "1"
val_ants_dataset = myData(val_root_dir, val_ants_label_dir, data_transform)
val_bees_dataset = myData(val_root_dir, val_bees_label_dir, data_transform)
test_data = val_ants_dataset + val_bees_dataset

if __name__ == '__main__':
    # img, label = train_ants_dataset[2]
    # img.show()
    # print(img)
    # print(label)
    print("train_dataset:{}".format(len(train_data)))
    print("val_dataset:{}".format(len(val_dataset)))