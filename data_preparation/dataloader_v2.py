from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import glob
import cv2
import random
import argparse
import torch
import csv
import os
from pyquaternion import Quaternion
import PIL
from PIL import Image
import torchvision.transforms as tfs
import sys
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.eval.common.utils import quaternion_yaw
# sys.path.append('/mnt/ssd/hao/')
# from utils import preprocess, preprocess_all

'''
# 用于数据增强的随机裁剪参数。在训练过程中, 在尺寸为image_size的图像中随机裁剪出一个crop_size大小的区域。在测试/验证过程中, 从图像的中心位置裁剪出crop_size大小的区域
self.crop_ij_max = [self.image_size[0] - self.crop_size[0], self.image_size[1] - self.crop_size[1]] # 训练过程中，裁剪的区域的左上角坐标的最大值
self.crop_ij_val = [int((self.image_size[0] - self.crop_size[0])/2), int((self.image_size[1] - self.crop_size[1])/2)]   # 测试/验证过程中，裁剪区域的左上角坐标值
'''
def random_crop(imagepath, image_size, crop_size, crop_ij_max, crop_ij_val, phase):
    image = cv2.imread(imagepath)
    image = cv2.resize(image, image_size)

    if phase == 'train' or phase == 'mini_train':
        xi = np.random.randint(0, crop_ij_max[0])
        yj = np.random.randint(0, crop_ij_max[1])
        image = image[yj:yj+crop_size[1], xi:xi+crop_size[0]]
    else:
        image = image[crop_ij_val[1]:crop_ij_val[1]+crop_size[1], crop_ij_val[0]:crop_ij_val[0]+crop_size[0]]

    image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    image = image.astype(np.float32)/255.0

def resize_and_crop_image(img, resize_dims, crop):
    # Bilinear resizing followed by cropping
    img = img.resize(resize_dims, resample=PIL.Image.BILINEAR)
    img = img.crop(crop)
    return img

def normalize(coors):
    coors = 1 / (1 + np.exp(-coors))
    coors = 2 * coors - 1
    return coors

def get_global_pose(ep_t, ep_r, cs_t, cs_r, inverse=False):
    if inverse is False:
        # inverse为False时，返回从当前sensor坐标系到global坐标系的变换矩阵
        global_from_ego = transform_matrix(ep_t, Quaternion(ep_r), inverse=False)
        ego_from_sensor = transform_matrix(cs_t, Quaternion(cs_r), inverse=False)
        pose = global_from_ego.dot(ego_from_sensor)
    else:
        # inverse为True时，返回从global坐标系到当前sensor坐标系的变换矩阵
        sensor_from_ego = transform_matrix(cs_t, Quaternion(cs_r), inverse=True)
        ego_from_global = transform_matrix(ep_t, Quaternion(ep_r), inverse=True)
        pose = sensor_from_ego.dot(ego_from_global)
    return pose

class DT_Img(Dataset):
    def __init__(self, cfg, phase='train'):
        # image_n: 每一帧环视相机图像有六张图，image_n代表使用其中几张
        # input_n: 输入序列帧数
        # predict_n: 预测未来路径点个数
        self.phase = phase  # train/val

        self.image_root = cfg.data_root
        self.image_n = cfg.image_n
        self.input_n = cfg.input_n
        self.predict_n = cfg.predict_n
        self.original_height = cfg.IMAGE_ORIGINAL_HEIGHT
        self.original_weight = cfg.IMAGE_ORIGINAL_WIDTH
        self.final_dim = cfg.IMAGE_FINAL_DIM
        self.resize_scale = cfg.IMAGE_RESIZE_SCALE
        self.crop_h = cfg.IMAGE_TOP_CROP
        self.coord_mode = cfg.coord_mode
        assert self.coord_mode in ['relative', 'delta']

        self.augmentation_parameters = self.get_resizing_and_cropping_parameters()

        self.normalise_image = tfs.Compose(
            [tfs.ToTensor(),
             tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        data_path = cfg.csv_root + phase + '/'
        filepaths = glob.glob(data_path + '*.csv')

        # if phase == 'train':
        #     data_path = cfg.data_path+'val/'
        #     filepaths = filepaths+glob.glob(data_path + '*.csv')

        scene_n = len(filepaths)
        self.frame_n = np.zeros(scene_n, np.int32) # 每个场景中front image的数量

        self.records = []

        for i, filepath in enumerate(filepaths):
            f = open(filepath, 'r')
            self.records += f.readlines()
            self.frame_n[i] = len(self.records)

        self.frame_n = (self.frame_n/6).astype(np.int32)
        self.frame_idxs = []

        # 留出past的4个帧，以及未来的6个帧。因此，对于一个长度为40的序列，有效的idx为4-33 (40-1-6)
        for i in range(scene_n):
            if i == 0:
                self.frame_idxs = [j for j in range(4, self.frame_n[i] - cfg.predict_n)]
            else:
                self.frame_idxs = self.frame_idxs + [j for j in range(self.frame_n[i-1] + 4, self.frame_n[i] - cfg.predict_n)]
        
        self.records = self.records[::6] # 总共28130张front iamge

        # if cfg.image_n == 1:
        #     self.records = self.records[::6]
        # if cfg.image_n == 3:
        #     self.records = [self.records[5::6], self.records[0::6], self.records[1::6]]
        ''''
        self.tfs_train = tfs.Compose([tfs.Resize(self.image_size),
                                      tfs.RandomCrop(self.crop_size),
                                      #tfs.RandomHorizontalFlip(0.5),
                                      tfs.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.02, 0.02)),
                                      #tfs.RandomRotation(30),
                                      #tfs.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                      tfs.ToTensor()
                                      ])
        self.tfs_val = tfs.Compose([tfs.Resize(self.image_size), tfs.CenterCrop(self.crop_size), tfs.ToTensor()])

        '''
    
    def get_resizing_and_cropping_parameters(self):
        original_height, original_width = self.original_height, self.original_weight
        final_height, final_width = self.final_dim

        resize_scale = self.resize_scale
        resize_dims = (int(original_width * resize_scale), int(original_height * resize_scale))
        resized_width, resized_height = resize_dims

        crop_h = self.crop_h
        crop_w = int(max(0, (resized_width - final_width) / 2))
        # Left, top, right, bottom crops.
        crop = (crop_w, crop_h, crop_w + final_width, crop_h + final_height)

        if resized_width != final_width:
            print('Zero padding left and right parts of the image.')
        if crop_h + final_height != resized_height:
            print('Zero padding bottom part of the image.')

        return {'scale_width': resize_scale,
                'scale_height': resize_scale,
                'resize_dims': resize_dims,
                'crop': crop,
                }

    def __len__(self):
        return len(self.frame_idxs)

    def __getitem__(self, idx):
        # 当前帧的idx，我们需要获取过去的4帧和未来的6帧
        frame_idx = self.frame_idxs[idx]
        imgs = []

        # coords加上yaw，变成3维坐标
        # 将过去的4帧和当前帧的坐标进行拼接，得到长度为5的序列
        egopose_past = np.zeros((self.input_n + 1, 3), dtype=np.float32)
        egopose_future = np.zeros((self.predict_n, 3), dtype=np.float32)

        # 当前帧的坐标
        imagepath_cur, x, y, z, w, wx, wy, wz, cx, cy, cz, cw, cwx, cwy, cwz = self.records[frame_idx].split(',')    #max(frame_idx-i, 0)
        ep_t_cur = np.array([float(x), float(y), float(z)])
        cs_t_cur = np.array([float(cx), float(cy), float(cz)])
        ep_r_cur = np.array([float(w), float(wx), float(wy), float(wz)])
        cs_r_cur = np.array([float(cw), float(cwx), float(cwy), float(cwz)])
        egopose_cur = get_global_pose(ep_t_cur, ep_r_cur, cs_t_cur, cs_r_cur, inverse=True)
        theta_cur = quaternion_yaw(Quaternion(matrix=egopose_cur))
        origin_cur = np.array(egopose_cur[:3, 3])
        # egopose_past[-1, :] = [origin_cur[0], origin_cur[1], theta_cur]
        # 当前帧的图像
        image_cur = Image.open(self.image_root+imagepath_cur)
        image_cur = resize_and_crop_image(
            image_cur, resize_dims=self.augmentation_parameters['resize_dims'], crop=self.augmentation_parameters['crop']
            )
        image_cur = self.normalise_image(image_cur)

        # 过去4帧的坐标及图像
        for i in range(self.input_n):
            imagepath, x, y, z, w, wx, wy, wz, cx, cy, cz, cw, cwx, cwy, cwz = self.records[frame_idx -4 + i].split(',')    #max(frame_idx-i, 0)
            ep_t_past = np.array([float(x), float(y), float(z)])
            cs_t_past = np.array([float(cx), float(cy), float(cz)])
            ep_r_past = np.array([float(w), float(wx), float(wy), float(wz)])
            cs_r_past = np.array([float(cw), float(cwx), float(cwy), float(cwz)])
            egopose_past_i = get_global_pose(ep_t_past, ep_r_past, cs_t_past, cs_r_past, inverse=False)
            egopose_past_i = egopose_cur.dot(egopose_past_i)
            theta = quaternion_yaw(Quaternion(matrix=egopose_past_i))
            origin = np.array(egopose_past_i[:3, 3])
            egopose_past[i, :] = [origin[0], origin[1], theta]

            image = Image.open(self.image_root+imagepath)
                
            # image = Image.open(self.image_root+imagepath).convert('RGB')
            image = resize_and_crop_image(
                image, resize_dims=self.augmentation_parameters['resize_dims'], crop=self.augmentation_parameters['crop']
                )
            # Normalise image
            image = self.normalise_image(image)
            imgs.append(image)
        imgs.append(image_cur) 
        imgs = torch.stack(imgs) # (5, 3, 224, 480)

        for j in range(1, self.predict_n + 1):
            x, y, z, w, wx, wy, wz, cx, cy, cz, cw, cwx, cwy, cwz = self.records[frame_idx + j].split(',')[1:]
            #coors_future[j*2], coors_future[j*2 + 1] = float(x), float(y)
            ep_t_future = np.array([float(x), float(y), float(z)])
            cs_t_future = np.array([float(cx), float(cy), float(cz)])
            ep_r_future = np.array([float(w), float(wx), float(wy), float(wz)])
            cs_r_future = np.array([float(cw), float(cwx), float(cwy), float(cwz)])
            yaw = Quaternion([w, wx, wy, wz]).yaw_pitch_roll[0]

            egopose_future_j = get_global_pose(ep_t_future, ep_r_future, cs_t_future, cs_r_future)
            egopose_future_j = egopose_cur.dot(egopose_future_j)
            theta = quaternion_yaw(Quaternion(matrix=egopose_future_j))
            origin = np.array(egopose_future_j[:3, 3])
            egopose_future[j - 1, :] = [origin[0], origin[1], theta]

        # set current pose as the origin
        egopose_past[-1, :] = [0.0, 0.0, theta_cur]
        if self.coord_mode == 'delta':
            for i in range(self.input_n, 0, -1):
                egopose_past[i, :2] = egopose_past[i, :2] - egopose_past[i - 1, :2]
            egopose_past = egopose_past[1:]

            for i in range(self.predict_n - 1, 0, -1):
                egopose_future[i, :2] = egopose_future[i, :2] - egopose_future[i - 1, :2]

        imagepath_cur = imagepath_cur.split('_')[-1][:-4]

        return imgs, egopose_past, egopose_future, imagepath_cur # torch.stack((coors_future, coors_future, coors_future), dim=0) 



def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, help='total number of training epochs')
    parser.add_argument('--device', type=str, default='1', help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--learning-rate', type=int, default=0.01, help='initial learning rate')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')

    parser.add_argument('--image-root', type=str, default='/data/tumdriving/hao/gr/dataset/Nuscenes/nuscenes/trainval/', help='图像路径的根路径')
    parser.add_argument('--data-path', type=str, default='/data/tumdriving/hao/gr/dataset/Nuscenes/image_t_r/', help='用于保存图像路径和标签的csv文件的路径')
    parser.add_argument('--image-n', type=int, default=1, help='每一帧环视相机图像有六张图, image_n代表使用其中几张')
    parser.add_argument('--input-n', type=int, default=4, help='输入序列帧数')
    parser.add_argument('--predict-n', type=int, default=6, help='')
    parser.add_argument('--IMAGE-FINAL_DIM', type=int, default=(224, 480), help='')
    parser.add_argument('--IMAGE-RESIZE_SCALE', type=int, default=0.3, help='')
    parser.add_argument('--IMAGE_TOP_CROP', type=int, default=46, help='')
    parser.add_argument('--IMAGE-ORIGINAL-HEIGHT', type=int, default=900, help='')
    parser.add_argument('--IMAGE-ORIGINAL-WIDTH', type=int, default=1600, help='[h, w]')


    return parser.parse_args()
    
if __name__=='__main__':
    cfg = parse_cfg()
    train_data = DT_Img(cfg, phase='train')
    train_data1 = DT_Noimg(cfg, phase='train')
    train_dataloader = DataLoader(train_data, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    train_dataloader1 = DataLoader(train_data1, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    i = 0
    j = 0
    for (image, xy_in, target1) in train_dataloader:
        i += 1
        print('with image')
        print('img', image.size())
        print('coo past', xy_in.size(), xy_in)
        print('coo future', target1.size(), target1)

        if i > 0:
            break


    for (image, xy_in, target1) in train_dataloader1:
        j += 1
        print('without image')
        print('img', image.size())
        print('coo past', xy_in.size(), xy_in)
        print('coo future', target1.size(), target1)

        if j > 0:
            break
