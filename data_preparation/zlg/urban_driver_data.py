import os
import torch
from torch.utils.data import Dataset
import numpy as np

class NuplanDataset(Dataset):
    def __init__(self, data_path, phase='train'):
        
        if phase == 'train':
            print('train')
        if phase == 'val':
            print('val')
        if phase == 'test':
            print('test')

        self.file_paths = os.listdir(data_path)
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = np.random.random((3, 512, 512)).astype(np.float32)
        image = torch.from_numpy(image)

        return image, image
    

class Level5Dataset(Dataset):
    def __init__(self):
        self

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        return
    
class NuscenesDataset(Dataset):
    def __init__(self, data_path, phase='train'):

        file_paths = os.listdir(data_path)
        
        if phase == 'train':
            print(data_path)
        if phase == 'val':
            print(data_path)
        if phase == 'test':
            print('test')
        self
        
    def __len__(self):
        return

    def __getitem__(self, idx):
        return