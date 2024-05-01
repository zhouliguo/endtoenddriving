from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import glob
import cv2
import random
import argparse
import torchvision.transforms as transforms

Aufgabe = 2

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
        scene_n = len(filepaths)
        sample_n = np.zeros(scene_n, np.int32)

        f = open(filepaths[0], 'r')
        self.imagepaths = f.readlines()

        sample_n[0] = len(self.imagepaths)

        for i, filepath in enumerate(filepaths[1:]):
            f = open(filepath, 'r')
            self.imagepaths = self.imagepaths+f.readlines()

            sample_n[i+1] = len(self.imagepaths)

        sample_n = ((sample_n-self.input_n)/(self.predict_n)).astype(np.int32)
        
        self.sample_index = []
        for i in range(scene_n):
            start_index = cfg.input_n if i == 0 else sample_n[i-1] + cfg.input_n + 1
            end_index = sample_n[i] - cfg.predict_n
            # print(f"Scene {i}: start_index={start_index}, end_index={end_index}")
            self.sample_index.extend(range(start_index, end_index))

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

        
        #cv2.imshow('image', image)
        #cv2.waitKey()
        
        if self.phase == 'train' or self.phase == 'mini_train':
            i = random.randint(0, self.crop_ij_max[0])
            j = random.randint(0, self.crop_ij_max[1])
            image = image[j:j+self.crop_size[1], i:i+self.crop_size[0]]
        else:
            image = image[self.crop_ij_val[1]:self.crop_ij_val[1]+self.crop_size[1], self.crop_ij_val[0]:self.crop_ij_val[0]+self.crop_size[0]]


        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB (height, width, channels)->(channels, height, width)
        image = image.astype(np.float32)/255.0

        '''
        # 得到过去一系列坐标
        if self.input_n != 0:
            coor = np.zeros(2*self.input_n, np.float32)

            for i in range(self.input_n):
                x_past, y_past = self.imagepaths[frame_idx-i-1].split(',')[1:3]
                x1 = x - float(x_past)
                y1 = y - float(y_past)
                coor[i*2], coor[i*2 + 1] = x1, y1

                # if abs(x1) > 20 or abs(y1) > 20:
                #    print("Problematic coordinates:", x1, y1, i-1, frame_idx)

        '''
        images_past_list = []
        # 得到过去一系列坐标和图片
        if self.input_n != 0:
            coor = np.zeros(2*self.input_n, np.float32)

            for i in range(self.input_n):
                imagepath_past, x_past, y_past, z, w, wx, wy, wz = self.imagepaths[frame_idx-i-1].split(',')
                x1 = x - float(x_past)
                y1 = y - float(y_past)
                coor[i*2], coor[i*2 + 1] = x1, y1

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

                images_past_list.append(image_past)
                '''
                if abs(x1) > 20 or abs(y1) > 20:
                    print("Problematic coordinates:", x1, y1, i-1, frame_idx)
                '''
            images_past = np.concatenate(images_past_list, axis=0)
            
        # print(images_past.shape)
            
        '''
        # 归一化坐标到[=1,1]
        min_value = np.min(coor)
        max_value = np.max(coor)
        normalized_coordinates = (coor - min_value) / (max_value - min_value)
        normalized_coordinates = normalized_coordinates * 2 - 1
        # normalized_coordinates = coor
        '''

        if self.input_n != 0:
            # 归一化坐标到[-1, 1]
            sigmoid_normalized_coordinates = 1 / (1 + np.exp(-coor))
            normalized_coordinates = 2 * sigmoid_normalized_coordinates - 1

        # 得到target：预测的predict_n个坐标
        target = np.zeros(2*self.predict_n, np.float32)

        for i in range(self.predict_n):
            x_futrure, y_future = self.imagepaths[frame_idx+i+1].split(',')[1:3]
            target[i*2], target[i*2 + 1] = float(x_futrure) - x, float(y_future) - y
            '''
            if abs(float(x_futrure) - x) > 500 or abs(float(y_future) - y) > 500:
                    print("Problematic coordinates target:", float(x_futrure) - x, float(y_future) - y)
            '''
        
        if self.input_n != 0:
            output = self.catchImageWithPos(image, normalized_coordinates)
            output = self.catchImageWithPosImagePast(image, images_past, normalized_coordinates)
        else:
            output = image
        return output, target
    
    def catchImageWithPosImagePast(self, images, image_past, coordinates):
        '''
        添加了过去的image_past
        把每一个坐标值放在 image 的新通道上
        '''

        num_channels, height, width = images.shape
        num_channels_past, height, width = image_past.shape
        coord_channel = np.zeros((num_channels + num_channels_past + len(coordinates), height, width), 
                                 dtype=images.dtype)
        coord_channel[:num_channels, :, :] = images
        coord_channel[num_channels:num_channels + num_channels_past, :, :] = image_past

        for i in range(len(coordinates)):
            coord_channel[num_channels + num_channels_past + i, :, :] = float(coordinates[i].item())

        return coord_channel
    
    def catchImageWithPos(self, images, coordinates):
        '''
        把每一个坐标值放在 image 的新通道上
        '''

        num_channels, height, width = images.shape
        coord_channel = np.zeros((num_channels + len(coordinates), height, width), dtype=images.dtype)
        coord_channel[:num_channels, :, :] = images

        for i in range(len(coordinates)):
            coord_channel[num_channels + i, :, :] = float(coordinates[i].item())

        return coord_channel

  
def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='total number of training epochs')
    parser.add_argument('--device', type=str, default='0', help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--learning-rate', type=int, default=0.01, help='initial learning rate')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')

    parser.add_argument('--data-root', type=str, default='D:\ECHO/4_2_WS2324_TUM/autonomousDriving/data/samples/', help='')
    parser.add_argument('--data-path', type=str, default='data/nuscenes/image_scenes/', help='')
    parser.add_argument('--image-n', type=int, default=1, help='每一帧环视相机图像有六张图, image_n代表使用其中几张')
    parser.add_argument('--input-n', type=int, default=3, help='是否使用序列图像, 0代表不使用, 其他数值n代表使用n张')
    parser.add_argument('--predict-n', type=int, default=6, help='')
    parser.add_argument('--image-size', type=int, default=[800, 450], help='')
    parser.add_argument('--crop-size', type=int, default=[704, 384], help='')

    return parser.parse_args()
    
if __name__=='__main__':
    cfg = parse_cfg()
    train_data = ResNetDataset(cfg, phase='train')
    # train_dataloader = DataLoader(train_data, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    for i, (image, target) in enumerate(train_dataloader):
        if i == 3:
            print(i, image.shape)
        #cv2.imshow('image', image.numpy()[0].transpose((1, 2, 0)))
        #cv2.waitKey(1)
        #print(target)  #'samples/CAM_FRONT/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915628412465.jpg' scene-0170
