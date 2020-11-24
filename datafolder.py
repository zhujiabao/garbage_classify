import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


random.seed(1)

#data_root = '/media/jiabao/D928233F6F20498B/Project/garbage_classify/garbage_classify_orgin/dataset/garbage_classify/train_data'

class garbageData(Dataset):
    def __init__(self, img_root,data_dir='train.txt'):

        self.data_dir = data_dir
        self.img_root = img_root
        self.data_info = self.get_img_info()

    def __getitem__(self, item):
        path_img, label = self.data_info[item]
        #print(path_img, label)
        ##img = Image.open(path_img).convert('RGB')

        return path_img, label

    def __len__(self):
        return len(self.data_info)

    #@staticmethod
    def get_img_info(self):

        data_info = []
        #data_root2 = self.img_root
        img_dirs = open(self.data_dir, 'r')
        for line in img_dirs.readlines():
            txt_info = open(line.strip(), 'r')
            for x in txt_info:
                #print(x.split()[0].rstrip(','), x.split()[1])
                img_path = os.path.join(self.img_root, x.split()[0].rstrip(','))
                label = x.split()[1]

                data_info.append((img_path, int(label)))

        return data_info

def default_loader(path):
    try:
        img = Image.open(path)
        #img = img.resize((456,456))

        #print(img)
        return img.convert('RGB')
    except:
        print("Can not open {0}".format(path))


class load_dataset(Dataset):
  def __init__(self, data_set, transform=None,loader=default_loader):
    img_list = []
    img_label = []

    self.loader = loader
    for i in range(len(data_set)):
      img_list.append(data_set[i][0])
      img_label.append(data_set[i][1])

    self.img_list =[file for file in img_list]
    self.label = img_label
    if transform is None:
        self.transform = T.Compose([
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
            T.Resize((456, 456)),
            T.RandomRotation(degrees=20, resample=False, expand=False, center=None),
            T.ToTensor(),
            T.Normalize(mean=[0.544, 0.506, 0.460],
                        std=[0.207, 0.213, 0.220])
        ])

  def __len__(self):
    return len(self.img_list)

  def __getitem__(self, index):
    img_path = self.img_list[index]
    label = self.label[index]
    img = self.loader(img_path)
    if self.transform is not None:
      try:
        img = self.transform(img)
      except:
        print('Cannot transform image: {}'.format(img_path))
    return img, label


if __name__ == '__main__':
    train_data = garbageData()
    #print(train_data[0])