import os
import cv2
import glob
import torch
from torch.utils.data import Dataset
import numpy as np

def letterbox(image, a=640):
    h, w, c = image.shape

    sw = 0
    sh = 0

    if h==w:
        image = cv2.resize(image, (a, a))
        return image, a, a, 0, 0
    else:
        letter = np.zeros((a,a,c), image.dtype)
        if h>w:
            w = w*a//h
            h = a
            image = cv2.resize(image, (w, h))

            sw = (a - w)//2
            letter[:,sw:sw+w] = image
        else:
            h = h*a//w
            w = a
            image = cv2.resize(image, (w, h))

            sh = (a - h)//2
            letter[sh:sh+h] = image

        return letter, w, h, sw, sh
        

class CocoDataset(Dataset):
    def __init__(self, data_path, size = 640, phase='train'):
        image_paths_tmp = glob.glob(data_path+'/*.jpg', recursive=True)

        self.image_paths = []
        self.label_paths = []

        for ipath in image_paths_tmp:
            lpath = ipath.replace('images', 'labels').replace('jpg', 'txt')
            if os.path.isfile(lpath):
                self.image_paths.append(ipath)
                self.label_paths.append(lpath)

        del image_paths_tmp

        self.size = size
        if phase == 'train':
            print('train')
        if phase == 'val':
            print('val')
        if phase == 'test':
            print('test')
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        #self.image_paths[idx] = 'D:/datasets/coco/images/train2017/000000111435.jpg'
        #self.label_paths[idx] = 'D:/datasets/coco/labels/train2017/000000111435.txt'
        image = cv2.imread(self.image_paths[idx])
        label = np.loadtxt(self.label_paths[idx])
        print(self.image_paths[idx])
        if len(label.shape) == 1:
            label = label[None,:]
        image, w, h, sw, sh = letterbox(image, self.size) # w h 代表原始图像在letterbox里的尺寸，sw sh 代表原始图像在letterbox里的起始位置
        label[:,1] = label[:,1]*w + sw - label[:,3]*w/2
        label[:,2] = label[:,2]*h + sh - label[:,4]*h/2
        label[:,3] = label[:,1] + label[:,3]*w
        label[:,4] = label[:,2] + label[:,4]*h
        for l in label:
            #l[1] = l[1]*w + sw - l[3]*w/2
            #l[2] = l[2]*h + sh - l[4]*h/2
            #l[3] = l[1] + l[3]*w
            #l[4] = l[2] + l[4]*h
            cv2.rectangle(image, (int(l[1]), int(l[2])), (int(l[3]), int(l[4])), (255,0,0), 2)
        cv2.imshow(self.image_paths[idx], image)
        cv2.waitKey(1)
        cv2.destroyWindow(self.image_paths[idx])
        print(self.image_paths[idx])

        image = image[..., ::-1]    # BGR to RGB
        image = np.transpose(image, [2,0,1])    # whc to cwh
        image = np.ascontiguousarray(image)    # At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported.
        image = torch.from_numpy(image)

        return image, image
    
# test
if __name__ == '__main__':
    from torch.utils.data import DataLoader

    train_data = CocoDataset(data_path = os.path.join('D:/datasets/coco/images', 'train2017'), phase='train')

    train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True)

    for train_i, (input, target) in enumerate(train_dataloader):
        input, target