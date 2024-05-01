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

def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm

def get_global_pose(ep_t, ep_r, cs_t, cs_r, inverse=False):
    if inverse is False:
        global_from_ego = transform_matrix(ep_t, Quaternion(ep_r), inverse=False)
        ego_from_sensor = transform_matrix(cs_t, Quaternion(cs_r), inverse=False)
        pose = global_from_ego.dot(ego_from_sensor)
    else:
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
        self.frame_n = np.zeros(scene_n, np.int32)

        self.records = []

        for i, filepath in enumerate(filepaths):
            f = open(filepath, 'r')
            self.records += f.readlines()
            self.frame_n[i] = len(self.records)

        self.frame_n = (self.frame_n/6).astype(np.int32)
        
        self.frame_idxs = []
        for i in range(scene_n):
            if i == 0:
                self.frame_idxs = [j for j in range(0, self.frame_n[i] - cfg.predict_n - cfg.input_n + 1)]
            else:
                self.frame_idxs = self.frame_idxs + [j for j in range(self.frame_n[i-1], self.frame_n[i] - cfg.predict_n - cfg.input_n + 1)]

        self.records = self.records[::6]

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
        frame_idx = self.frame_idxs[idx]

        # images = torch.zeros((3, self.crop_size[0], self.crop_size[1]), dtype=torch.float32)
        coors_past = torch.zeros(2*self.input_n, dtype=torch.float32)
        yaws = torch.zeros(self.input_n, dtype=torch.float32)
        coors_future = torch.zeros(2*self.predict_n, dtype=torch.float32)

        ep_t_past = np.zeros((self.input_n, 3))
        cs_t_past = np.zeros((self.input_n, 3))

        ep_t_future = np.zeros((self.predict_n, 3))
        cs_t_future = np.zeros((self.predict_n, 3))

        ep_r_past = np.zeros((self.input_n, 4))
        cs_r_past = np.zeros((self.input_n, 4))

        ep_r_future = np.zeros((self.predict_n, 4))
        cs_r_future = np.zeros((self.predict_n, 4))

        for i in range(self.input_n):
            imagepath, x, y, z, w, wx, wy, wz, cx, cy, cz, cw, cwx, cwy, cwz = self.records[frame_idx + i].split(',')    #max(frame_idx-i, 0)

            ep_t_past[i] = np.array([float(x), float(y), float(z)])
            cs_t_past[i] = np.array([float(cx), float(cy), float(cz)])
            ep_r_past[i] = np.array([float(w), float(wx), float(wy), float(wz)])
            cs_r_past[i] = np.array([float(cw), float(cwx), float(cwy), float(cwz)])

            yaws[i] = Quaternion([w, wx, wy, wz]).yaw_pitch_roll[0]
         
            coors_past[2 * i] = float(x)
            coors_past[2 * i + 1] = float(y)

            if i == self.input_n - 1:
                image = Image.open(self.image_root+imagepath)
                # image = Image.open(self.image_root+imagepath).convert('RGB')
                image = resize_and_crop_image(
                    image, resize_dims=self.augmentation_parameters['resize_dims'], crop=self.augmentation_parameters['crop']
                    )
            # Normalise image
                image = self.normalise_image(image)

                # if self.phase == 'train' or self.phase == 'mini_train':
                #     image = self.tfs_train(image)
                # else:
                #     image = self.tfs_val(image)

            # images[i,:3] = image # * yaws[i]
            # images[i,3] = yaws[i]
        
        egopose_cur = get_global_pose(ep_t_past[-1], ep_r_past[-1], cs_t_past[-1], cs_r_past[-1], inverse=True)

        for j in range(self.predict_n):
            x, y, z, w, wx, wy, wz, cx, cy, cz, cw, cwx, cwy, cwz = self.records[frame_idx+self.input_n+j].split(',')[1:]
            #coors_future[j*2], coors_future[j*2 + 1] = float(x), float(y)
            ep_t_future[j] = np.array([float(x), float(y), float(z)])
            cs_t_future[j] = np.array([float(cx), float(cy), float(cz)])
            ep_r_future[j] = np.array([float(w), float(wx), float(wy), float(wz)])
            cs_r_future[j] = np.array([float(cw), float(cwx), float(cwy), float(cwz)])

            egopose_future = get_global_pose(ep_t_future[j], ep_r_future[j], cs_t_future[j], cs_r_future[j])
            egopose_future = egopose_cur.dot(egopose_future)
            origin = np.array(egopose_future[:3, 3])
            coors_future[j*2], coors_future[j*2 + 1] = origin[0], origin[1]

        if self.input_n > 1:
            coors_past[0:-2:2] = coors_past[2::2]-coors_past[0:-2:2]  # 增量: 过去一帧的坐标 — 前一帧的坐标 T(-n)-T(-n-1)
            coors_past[1:-2:2] = coors_past[3::2]-coors_past[1:-2:2]
        
            #coors_past[0:-2:2] = coors_past[0:-2:2]-coors_past[-2]
            #coors_past[1:-2:2] = coors_past[1:-2:2]-coors_past[-1]

        # (coors_past[-2], coors_past[-1])是车辆当前位置的坐标,coors_future被处理成相对当前位置的增量
        #coors_future[0::2] = coors_future[0::2]#-coors_past[-2]
        #coors_future[1::2] = coors_future[1::2]#-coors_past[-1]

        #images = images[:,::-1]
        #images = np.ascontiguousarray(images)

        #for i in range(1, 5):
        #    images[i,4] = coors_past[2*i]
        #    images[i,5] = coors_past[2*i+1]

        return image, coors_past, coors_future # torch.stack((coors_future, coors_future, coors_future), dim=0) 

class DT_Noimg(Dataset):
    def __init__(self, cfg, phase='train'):
        # image_n: 每一帧环视相机图像有六张图，image_n代表使用其中几张
        # input_n: 输入序列帧数
        # predict_n: 预测未来路径点个数
        
        self.phase = phase  # train/val

        self.image_root = cfg.image_root
        self.image_n = cfg.image_n
        self.input_n = cfg.input_n
        self.predict_n = cfg.predict_n


        data_path = cfg.data_path + phase + '/'
        filepaths = glob.glob(data_path + '*.csv')
        # if phase == 'train':
        #     data_path = cfg.data_path+'val/'
        #     filepaths = filepaths+glob.glob(data_path + '*.csv')

        scene_n = len(filepaths)
        self.frame_n = np.zeros(scene_n, np.int32)

        self.records = []

        for i, filepath in enumerate(filepaths):
            f = open(filepath, 'r')
            self.records += f.readlines()
            self.frame_n[i] = len(self.records)

        self.frame_n = (self.frame_n/6).astype(np.int32)
        
        self.frame_idxs = []
        for i in range(scene_n):
            if i == 0:
                self.frame_idxs = [j for j in range(0, self.frame_n[i] - cfg.predict_n - cfg.input_n + 1)]
            else:
                self.frame_idxs = self.frame_idxs + [j for j in range(self.frame_n[i-1], self.frame_n[i] - cfg.predict_n - cfg.input_n + 1)]

        self.records = self.records[::6]

        # if cfg.image_n == 1:
        #     self.records = self.records[::6]
        # if cfg.image_n == 3:
        #     self.records = [self.records[5::6], self.records[0::6], self.records[1::6]]

     
    def __len__(self):
        return len(self.frame_idxs)

    def __getitem__(self, idx):
        frame_idx = self.frame_idxs[idx]

        # images = torch.zeros((self.input_n, 3, self.crop_size[0], self.crop_size[1]), dtype=torch.float32)
        coors_past = torch.zeros(2*self.input_n, dtype=torch.float32)
        yaws = torch.zeros(self.input_n, dtype=torch.float32)
        coors_future = torch.zeros(2*self.predict_n, dtype=torch.float32)

        ep_t_past = np.zeros((self.input_n, 3))
        cs_t_past = np.zeros((self.input_n, 3))

        ep_t_future = np.zeros((self.predict_n, 3))
        cs_t_future = np.zeros((self.predict_n, 3))

        ep_r_past = np.zeros((self.input_n, 4))
        cs_r_past = np.zeros((self.input_n, 4))

        ep_r_future = np.zeros((self.predict_n, 4))
        cs_r_future = np.zeros((self.predict_n, 4))

        for i in range(self.input_n):
            imagepath, x, y, z, w, wx, wy, wz, cx, cy, cz, cw, cwx, cwy, cwz = self.records[frame_idx + i].split(',')    #max(frame_idx-i, 0)

            ep_t_past[i] = np.array([float(x), float(y), float(z)])
            cs_t_past[i] = np.array([float(cx), float(cy), float(cz)])
            ep_r_past[i] = np.array([float(w), float(wx), float(wy), float(wz)])
            cs_r_past[i] = np.array([float(cw), float(cwx), float(cwy), float(cwz)])

            yaws[i] = Quaternion([w, wx, wy, wz]).yaw_pitch_roll[0]
         
            coors_past[2*i] = float(x)
            coors_past[2*i+1] = float(y)
            
            # image = Image.open(self.image_root+imagepath).convert('RGB')

            # if self.phase == 'train' or self.phase == 'mini_train':
            #     image = self.tfs_train(image)
            # else:
            #     image = self.tfs_val(image)

            # images[i,:3] = image # * yaws[i]
        
        egopose_cur = get_global_pose(ep_t_past[-1], ep_r_past[-1], cs_t_past[-1], cs_r_past[-1], inverse=True)

        for j in range(self.predict_n):
            x, y, z, w, wx, wy, wz, cx, cy, cz, cw, cwx, cwy, cwz = self.records[frame_idx+self.input_n+j].split(',')[1:]
            #coors_future[j*2], coors_future[j*2 + 1] = float(x), float(y)
            ep_t_future[j] = np.array([float(x), float(y), float(z)])
            cs_t_future[j] = np.array([float(cx), float(cy), float(cz)])
            ep_r_future[j] = np.array([float(w), float(wx), float(wy), float(wz)])
            cs_r_future[j] = np.array([float(cw), float(cwx), float(cwy), float(cwz)])

            egopose_future = get_global_pose(ep_t_future[j], ep_r_future[j], cs_t_future[j], cs_r_future[j])
            egopose_future = egopose_cur.dot(egopose_future)
            origin = np.array(egopose_future[:3, 3])
            coors_future[j*2], coors_future[j*2 + 1] = origin[0], origin[1]

        if self.input_n > 1:
            coors_past[0:-2:2] = coors_past[2::2]-coors_past[0:-2:2]  # 增量: 过去一帧的坐标 — 前一帧的坐标 T(-n)-T(-n-1)
            coors_past[1:-2:2] = coors_past[3::2]-coors_past[1:-2:2]

        # (coors_past[-2], coors_past[-1])是车辆当前位置的坐标,coors_future被处理成相对当前位置的增量
        # coors_future[0::2] = coors_future[0::2]#-coors_past[-2]
        # coors_future[1::2] = coors_future[1::2]#-coors_past[-1]
        image = torch.zeros((1,1,1,1))

        return image, coors_past, coors_future # torch.stack((coors_future, coors_future, coors_future), dim=0) 



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
