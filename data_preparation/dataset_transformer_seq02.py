from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import glob
import cv2
import random
import argparse
import torchvision.transforms as transforms


class ResNetDataset(Dataset):
    def __init__(self, cfg, phase='train'):
        # image_n: 每一帧环视相机图像有六张图，image_n代表使用其中几张
        # input_n: 是否使用序列图像: 0代表不使用序列图像, 仅使用当前帧; 其他数值n代表使用当前帧和之前的n帧作为时序图像输入
        # predict_n: 预测未来路径点个数
        
        # 初始化参数
        self.phase = phase

        data_path = cfg.data_path+phase+'/'

        self.data_root = cfg.data_root
        self.image_n = cfg.image_n
        self.input_n = cfg.input_n
        self.predict_n = cfg.predict_n

        self.image_size = cfg.image_size
        self.crop_size = cfg.crop_size

        self.crop_ij_max = [self.image_size[0] - self.crop_size[0], self.image_size[1] - self.crop_size[1]]  # 
        self.crop_ij_val = [int((self.image_size[0] - self.crop_size[0])/2), int((self.image_size[1] - self.crop_size[1])/2)]

        filepaths = glob.glob(data_path + '*.csv')
        scene_n = len(filepaths) # scene_n = 700
        sample_n = np.zeros(scene_n, np.int32)

        f = open(filepaths[0], 'r')
        self.imagepaths = f.readlines()

        sample_n[0] = len(self.imagepaths) # sample_n[0]=240

        for i, filepath in enumerate(filepaths[1:]):
            f = open(filepath, 'r')
            self.imagepaths = self.imagepaths+f.readlines()

            sample_n[i+1] = len(self.imagepaths)

        sample_n = ((sample_n)/(6)).astype(np.int32)
        
        self.sample_index = []
        for i in range(scene_n):
            start_index = cfg.input_n if i == 0 else sample_n[i-1] + cfg.input_n + 1
            end_index = sample_n[i] - cfg.predict_n
            # print(f"Scene {i}: start_index={start_index}, end_index={end_index}")
            self.sample_index.extend(range(start_index, end_index))
        # print(len(self.sample_index)) # 21831
        if cfg.image_n == 1:
            self.imagepaths = self.imagepaths[::6] # 取出1scene中的一系列图片路径（CAM_FRONT)
        if cfg.image_n == 3:
            self.imagepaths = [self.imagepaths[5::6], self.imagepaths[0::6], self.imagepaths[1::6]]

        
    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        frame_idx = self.sample_index[idx]

        imagepath, x, y, z, w, wx, wy, wz = self.imagepaths[frame_idx].split(',')

        x = float(x)
        y = float(y)

        # 提取当前时间图片
        image = cv2.imread(self.data_root+imagepath)
        image = cv2.resize(image, self.image_size)
        
        if self.phase == 'train' or self.phase == 'mini_train':
            i = random.randint(0, self.crop_ij_max[0])
            j = random.randint(0, self.crop_ij_max[1])
            image = image[j:j+self.crop_size[1], i:i+self.crop_size[0]]
        else:
            image = image[self.crop_ij_val[1]:self.crop_ij_val[1]+self.crop_size[1], self.crop_ij_val[0]:self.crop_ij_val[0]+self.crop_size[0]]

        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB (height, width, channels)->(channels, height, width)
        image = image.astype(np.float32)/255.0

        # 得到过去一系列坐标和图片
        images_list = []
        images_list.append(image)
        if self.input_n != 0:
            coor = np.zeros(2*(self.input_n+1), np.float32)
            coor[0], coor[1] = x, y

            for i in range(self.input_n):
                imagepath_past, x_past, y_past, z, w, wx, wy, wz = self.imagepaths[frame_idx-i-1].split(',')
                coor[(i+1)*2], coor[(i+1)*2 + 1] = float(x_past), float(y_past)

                image_past = cv2.imread(self.data_root+imagepath_past)
                image_past = cv2.resize(image_past, self.image_size)

                if self.phase == 'train' or self.phase == 'mini_train':
                    i = random.randint(0, self.crop_ij_max[0])
                    j = random.randint(0, self.crop_ij_max[1])
                    image_past = image_past[j:j+self.crop_size[1], i:i+self.crop_size[0]]
                else:
                    image_past = image_past[self.crop_ij_val[1]:self.crop_ij_val[1]+self.crop_size[1], self.crop_ij_val[0]:self.crop_ij_val[0]+self.crop_size[0]]

                image_past = image_past.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB (height, width, channels)->(channels, height, width)
                image_past = image_past.astype(np.float32)/255.0

                images_list.append(image_past)
            
        # 计算相对坐标
        coor = coor[::-1]
        deta_coor = coor[2:] - coor[:-2]
        # 得到target：预测的predict_n个坐标
        target = np.zeros((self.predict_n, 2), np.float32)

        for i in range(self.predict_n):
            x_future, y_future = self.imagepaths[frame_idx+i+1].split(',')[1:3]
            target[i, 0] = float(x_future) - x
            target[i, 1] = float(y_future) - y

        # image_list[im_tn,im_tn-1, ... im_t0]
        # array[deta(cox_tn - cox_tn-1), deta(coy_tn - coy_tn-1), ... ]

        # deta_coor = torch.arange(-self.input_n, 1.0, 1)

        return images_list[::-1], deta_coor, target
    
class NuscenesDatasetCoor(Dataset):
    def __init__(self, cfg, phase='train'):
        # image_n: 每一帧环视相机图像有六张图，image_n代表使用其中几张
        # input_n: 是否使用序列图像: 0代表不使用序列图像, 仅使用当前帧; 其他数值n代表使用当前帧和之前的n帧作为时序图像输入
        # predict_n: 预测未来路径点个数
        
        # 初始化参数
        self.phase = phase

        data_path = cfg.data_path+phase+'/'

        self.data_root = cfg.data_root
        self.image_n = cfg.image_n
        self.input_n = cfg.input_n
        self.predict_n = cfg.predict_n

        self.image_size = cfg.image_size
        self.crop_size = cfg.crop_size

        self.crop_ij_max = [self.image_size[0] - self.crop_size[0], self.image_size[1] - self.crop_size[1]]  # 
        self.crop_ij_val = [int((self.image_size[0] - self.crop_size[0])/2), int((self.image_size[1] - self.crop_size[1])/2)]

        filepaths = glob.glob(data_path + '*.csv')
        scene_n = len(filepaths) # scene_n = 700
        sample_n = np.zeros(scene_n, np.int32)

        f = open(filepaths[0], 'r')
        self.imagepaths = f.readlines()

        sample_n[0] = len(self.imagepaths) # sample_n[0]=240

        for i, filepath in enumerate(filepaths[1:]):
            f = open(filepath, 'r')
            self.imagepaths = self.imagepaths+f.readlines()

            sample_n[i+1] = len(self.imagepaths)

        sample_n = ((sample_n)/(6)).astype(np.int32)
        
        self.sample_index = []
        for i in range(scene_n):
            start_index = cfg.input_n if i == 0 else sample_n[i-1] + cfg.input_n + 1
            end_index = sample_n[i] - cfg.predict_n
            # print(f"Scene {i}: start_index={start_index}, end_index={end_index}")
            self.sample_index.extend(range(start_index, end_index))
        # print(len(self.sample_index)) # 21831
        if cfg.image_n == 1:
            self.imagepaths = self.imagepaths[::6] # 取出1scene中的一系列图片路径（CAM_FRONT)
        if cfg.image_n == 3:
            self.imagepaths = [self.imagepaths[5::6], self.imagepaths[0::6], self.imagepaths[1::6]]

        
    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        frame_idx = self.sample_index[idx]

        imagepath, x, y, z, w, wx, wy, wz = self.imagepaths[frame_idx].split(',')

        x = float(x)
        y = float(y)

        # 得到过去一系列坐标
        if self.input_n != 0:
            coor = np.zeros((self.input_n, 2), np.float32)

            for i in range(self.input_n):
                imagepath_past, x_past, y_past, z, w, wx, wy, wz = self.imagepaths[frame_idx-i-1].split(',')
                coor[i, 0] = x - float(x_past)
                coor[i, 1] = y - float(y_past)
            
        # 得到target：预测的predict_n个坐标
        target = np.zeros((self.predict_n, 2), np.float32)

        for i in range(self.predict_n):
            x_future, y_future = self.imagepaths[frame_idx+i+1].split(',')[1:3]
            target[i, 0] = float(x_future) - x
            target[i, 1] = float(y_future) - y

        # array[deta(cox_tn - cox_tn-1), deta(coy_tn - coy_tn-1), ... ]

        # deta_coor = torch.arange(-self.input_n, 1.0, 1)

        return coor, target
    
class NuscenesDatasetImage(Dataset):
    def __init__(self, cfg, phase='train'):
        # image_n: 每一帧环视相机图像有六张图，image_n代表使用其中几张
        # input_n: 是否使用序列图像: 0代表不使用序列图像, 仅使用当前帧; 其他数值n代表使用当前帧和之前的n帧作为时序图像输入
        # predict_n: 预测未来路径点个数
        
        # 初始化参数
        self.phase = phase

        data_path = cfg.data_path+phase+'/'

        self.data_root = cfg.data_root
        self.image_n = cfg.image_n
        self.input_n = cfg.input_n
        self.predict_n = cfg.predict_n

        self.image_size = cfg.image_size
        self.crop_size = cfg.crop_size

        self.crop_ij_max = [self.image_size[0] - self.crop_size[0], self.image_size[1] - self.crop_size[1]]  # 
        self.crop_ij_val = [int((self.image_size[0] - self.crop_size[0])/2), int((self.image_size[1] - self.crop_size[1])/2)]

        filepaths = glob.glob(data_path + '*.csv')
        scene_n = len(filepaths) # scene_n = 700
        sample_n = np.zeros(scene_n, np.int32)

        f = open(filepaths[0], 'r')
        self.imagepaths = f.readlines()

        sample_n[0] = len(self.imagepaths) # sample_n[0]=240

        for i, filepath in enumerate(filepaths[1:]):
            f = open(filepath, 'r')
            self.imagepaths = self.imagepaths+f.readlines()

            sample_n[i+1] = len(self.imagepaths)

        sample_n = ((sample_n)/(6)).astype(np.int32)
        
        self.sample_index = []
        for i in range(scene_n):
            start_index = cfg.input_n if i == 0 else sample_n[i-1] + cfg.input_n + 1
            end_index = sample_n[i] - cfg.predict_n
            # print(f"Scene {i}: start_index={start_index}, end_index={end_index}")
            self.sample_index.extend(range(start_index, end_index))
        # print(len(self.sample_index)) # 21831
        if cfg.image_n == 1:
            self.imagepaths = self.imagepaths[::6] # 取出1scene中的一系列图片路径（CAM_FRONT)
        if cfg.image_n == 3:
            self.imagepaths = [self.imagepaths[5::6], self.imagepaths[0::6], self.imagepaths[1::6]]

        
    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        frame_idx = self.sample_index[idx]

        imagepath, x, y, z, w, wx, wy, wz = self.imagepaths[frame_idx].split(',')

        x = float(x)
        y = float(y)

        # 提取当前时间图片
        image = cv2.imread(self.data_root+imagepath)
        image = cv2.resize(image, self.image_size)
        
        if self.phase == 'train' or self.phase == 'mini_train':
            i = random.randint(0, self.crop_ij_max[0])
            j = random.randint(0, self.crop_ij_max[1])
            image = image[j:j+self.crop_size[1], i:i+self.crop_size[0]]
        else:
            image = image[self.crop_ij_val[1]:self.crop_ij_val[1]+self.crop_size[1], self.crop_ij_val[0]:self.crop_ij_val[0]+self.crop_size[0]]

        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB (height, width, channels)->(channels, height, width)
        image = image.astype(np.float32)/255.0

        # 得到过去一系列坐标和图片
        images_list = []
        images_list.append(image)
        if self.input_n != 0:

            for i in range(self.input_n):
                imagepath_past, x_past, y_past, z, w, wx, wy, wz = self.imagepaths[frame_idx-i-1].split(',')

                image_past = cv2.imread(self.data_root+imagepath_past)
                image_past = cv2.resize(image_past, self.image_size)

                if self.phase == 'train' or self.phase == 'mini_train':
                    i = random.randint(0, self.crop_ij_max[0])
                    j = random.randint(0, self.crop_ij_max[1])
                    image_past = image_past[j:j+self.crop_size[1], i:i+self.crop_size[0]]
                else:
                    image_past = image_past[self.crop_ij_val[1]:self.crop_ij_val[1]+self.crop_size[1], self.crop_ij_val[0]:self.crop_ij_val[0]+self.crop_size[0]]

                image_past = image_past.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB (height, width, channels)->(channels, height, width)
                image_past = image_past.astype(np.float32)/255.0

                images_list.append(image_past)
            

        # 得到target：预测的predict_n个坐标
        # target = np.zeros(2*self.predict_n, np.float32)
        target = np.zeros((self.predict_n, 2), np.float32)

        for i in range(self.predict_n):
            x_future, y_future = self.imagepaths[frame_idx+i+1].split(',')[1:3]
            # target[i*2], target[i*2 + 1] = float(x_futrure) - x, float(y_future) - y
            target[i, 0] = float(x_future) - x
            target[i, 1] = float(y_future) - y

        # image_list[im_tn,im_tn-1, ... im_t0]
        # array[deta(cox_tn - cox_tn-1), deta(coy_tn - coy_tn-1), ... ]

        # deta_coor = torch.arange(-self.input_n, 1.0, 1)
        images_list = images_list[::-1]
        # 打包
        # packed_tensor = torch.stack((tensor1, tensor2, tensor3))
        images_packed = np.stack(images_list) # (6, 3, 384, 704)

        return images_packed, target
  
def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='total number of training epochs')
    parser.add_argument('--device', type=str, default='0', help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--learning-rate', type=int, default=0.01, help='initial learning rate')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')

    # parser.add_argument('--data-root', type=str, default='D:/ECHO/4_2_WS2324_TUM/autonomousDriving/code_project/dataset/v1.0-mini/', help='')
    parser.add_argument('--data-root', type=str, default='D:/ECHO/4_2_WS2324_TUM/autonomousDriving/data/samples/', help='')
    parser.add_argument('--data-path', type=str, default='data/nuscenes/image_scenes/', help='')
    parser.add_argument('--image-n', type=int, default=1, help='每一帧环视相机图像有六张图, image_n代表使用其中几张')
    parser.add_argument('--input-n', type=int, default=4, help='是否使用序列图像, 0代表不使用, 其他数值n代表使用n张')
    parser.add_argument('--predict-n', type=int, default=4, help='')
    parser.add_argument('--image-size', type=int, default=[800, 450], help='')
    parser.add_argument('--crop-size', type=int, default=[704, 384], help='')

    return parser.parse_args()
    
if __name__=='__main__':
    cfg = parse_cfg()
    # train_data = NuscenesDatasetCoor(cfg, phase='train')
    train_data = ResNetDataset(cfg, phase='train')
    # train_dataloader = DataLoader(train_data, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    train_dataloader = DataLoader(train_data, batch_size=10, shuffle=True, num_workers=0, pin_memory=True)
    for i, (coor, target) in enumerate(train_dataloader):
        if i == 3:
            print(i,coor.shape,target) # torch.Size([16, 6, 384, 704])
        #cv2.imshow('image', image.numpy()[0].transpose((1, 2, 0)))
        #cv2.waitKey(1)
        #print(target)  #'samples/CAM_FRONT/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915628412465.jpg' scene-0170
