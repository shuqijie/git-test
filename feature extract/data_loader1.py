import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


class ResizeImage():
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size
    def __call__(self, img):
      th, tw = self.size
      return img.resize((th, tw))

class PlaceCrop(object):

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))

def get_train_valid_loader(root_path, batch_size, kwargs):

    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path , transform=transform)
    num_train = len(data)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=train_sampler, drop_last=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size,sampler=valid_sampler, drop_last=True, **kwargs
    )


    return train_loader,valid_loader

def get_test_loader(root_path, dir, batch_size, kwargs):
    start_center = (256 - 224 - 1) / 2
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         PlaceCrop(224, start_center, start_center),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
    return test_loader

