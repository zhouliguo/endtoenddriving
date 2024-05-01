import cv2
import glob
import torch
import argparse
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
import torchvision.transforms as tfs
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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
            
class NuScenesDataset(Dataset):
    def __init__(self, cfg, phase='train'):
        # image_n: 每一帧环视相机图像有六张图，image_n代表使用其中几张
        # input_n: 输入序列帧数
        # predict_n: 预测未来路径点个数
        
        self.phase = phase  # train/val

        self.res_chain = cfg.res_chain
        self.image_root = cfg.image_root
        self.image_n = cfg.image_n
        self.input_n = cfg.input_n
        self.predict_n = cfg.predict_n

        self.image_size = cfg.image_size
        self.crop_size = cfg.crop_size

        data_path = cfg.data_path+phase+'/'
        filepaths = glob.glob(data_path + '*.csv')
        #if phase == 'train':
        #    data_path = cfg.data_path+'val/'
        #    filepaths = filepaths+glob.glob(data_path + '*.csv')

        scene_n = len(filepaths)
        self.frame_n = np.zeros(scene_n, np.int32)

        self.records = []

        for i, filepath in enumerate(filepaths):
            f = open(filepath, 'r')
            self.records += f.readlines()
            self.frame_n[i] = len(self.records)

        #self.frame_n = (self.frame_n/6).astype(np.int32)
        
        self.frame_idxs = []
        for i in range(scene_n):
            if i == 0:
                self.frame_idxs = [j for j in range(0, self.frame_n[i] - cfg.predict_n - cfg.input_n + 1)]
            else:
                self.frame_idxs = self.frame_idxs + [j for j in range(self.frame_n[i-1], self.frame_n[i] - cfg.predict_n - cfg.input_n + 1)]

        #if cfg.image_n == 1:
        #    self.records = self.records[::6]
        #if cfg.image_n == 3:
        #    self.records = [self.records[5::6], self.records[0::6], self.records[1::6]]

        self.tfs_train = tfs.Compose([tfs.Resize(self.image_size),
                                      tfs.RandomCrop(self.crop_size),
                                      #tfs.RandomHorizontalFlip(0.5),
                                      tfs.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.02, 0.02)),
                                      #tfs.RandomRotation(30),
                                      #tfs.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                      tfs.ToTensor()
                                      ])
        self.tfs_val = tfs.Compose([tfs.Resize(self.image_size), tfs.CenterCrop(self.crop_size), tfs.ToTensor()])
        
    def __len__(self):
        return len(self.frame_idxs)

    def __getitem__(self, idx):
        frame_idx = self.frame_idxs[idx]

        images = torch.zeros((self.input_n, 3, self.crop_size[0], self.crop_size[1]), dtype=torch.float32)
        coors_past = torch.zeros(2*self.input_n, dtype=torch.float32)
        yaws = torch.zeros(self.input_n, dtype=torch.float32)
        coors_future = torch.zeros(2*self.predict_n, dtype=torch.float32)

        for i in range(self.input_n):
            record = self.records[frame_idx+i].split(',')    #max(frame_idx-i, 0)
            imagepath = record[:6]
            x, y, z, w, wx, wy, wz = record[6:13]

            yaws[i] = Quaternion([w, wx, wy, wz]).yaw_pitch_roll[0]
         
            coors_past[2*i] = float(x)
            coors_past[2*i+1] = float(y)
            
            image = Image.open(self.image_root+imagepath[0]).convert('RGB')

            if self.phase == 'train' or self.phase == 'mini_train':
                image = self.tfs_train(image)
            else:
                image = self.tfs_val(image)

            images[i,:3] = image# * yaws[i]
            #images[i,3] = yaws[i]
            
        for j in range(self.predict_n):
            x, y, z, w, wx, wy, wz = self.records[frame_idx+self.input_n+j].split(',')[6:13]
            coors_future[j*2], coors_future[j*2 + 1] = float(x), float(y)

        if self.input_n > 1:
            if self.res_chain:
                coors_past[0:-2:2] = coors_past[2::2]-coors_past[0:-2:2]  # 增量: 过去一帧的坐标 — 前一帧的坐标 T(-n)-T(-n-1)
                coors_past[1:-2:2] = coors_past[3::2]-coors_past[1:-2:2]
            else:
                coors_past[0:-2:2] = coors_past[0:-2:2]-coors_past[-2]
                coors_past[1:-2:2] = coors_past[1:-2:2]-coors_past[-1]

        coors_future[0::2] = coors_future[0::2]-coors_past[-2]
        coors_future[1::2] = coors_future[1::2]-coors_past[-1]

        
        # (coors_past[-2], coors_past[-1])是车辆当前位置的坐标,coors_future被处理成相对当前位置的增量
        #coors_future[0::2] = coors_future[0::2]-coors_past[-2]
        #coors_future[1::2] = coors_future[1::2]-coors_past[-1]

        #images = images[:,::-1]
        #images = np.ascontiguousarray(images)

        #for i in range(1, 5):
        #    images[i,4] = coors_past[2*i]
        #    images[i,5] = coors_past[2*i+1]

        return images, coors_past, coors_future # torch.stack((coors_future, coors_future, coors_future), dim=0) 

class NuScenesDataset_T(Dataset):
    def __init__(self, cfg, phase='train'):
        # image_n: 每一帧环视相机图像有六张图，image_n代表使用其中几张
        # input_n: 输入序列帧数
        # predict_n: 预测未来路径点个数
        
        self.phase = phase  # train/val
        self.res_chain = cfg.res_chain

        self.image_root = cfg.image_root
        self.image_n = cfg.image_n
        self.input_n = cfg.input_n
        self.predict_n = cfg.predict_n

        self.image_size = cfg.image_size
        self.crop_size = cfg.crop_size

        data_path = cfg.data_path+phase+'/'
        filepaths = glob.glob(data_path + '*.csv')
        #if phase == 'train':
        #    data_path = cfg.data_path+'val/'
        #    filepaths = filepaths+glob.glob(data_path + '*.csv')

        scene_n = len(filepaths)
        self.frame_n = np.zeros(scene_n, np.int32)

        self.records = []

        for i, filepath in enumerate(filepaths):
            f = open(filepath, 'r')
            self.records += f.readlines()
            self.frame_n[i] = len(self.records)

        #self.frame_n = (self.frame_n/6).astype(np.int32)
        
        self.frame_idxs = []
        for i in range(scene_n):
            if i == 0:
                self.frame_idxs = [j for j in range(0, self.frame_n[i] - cfg.predict_n - cfg.input_n + 1)]
            else:
                self.frame_idxs = self.frame_idxs + [j for j in range(self.frame_n[i-1], self.frame_n[i] - cfg.predict_n - cfg.input_n + 1)]

        #if cfg.image_n == 1:
        #    self.records = self.records[::6]
        #if cfg.image_n == 3:
        #    self.records = [self.records[5::6], self.records[0::6], self.records[1::6]]

        self.tfs_train = tfs.Compose([tfs.Resize(self.image_size),
                                      tfs.RandomCrop(self.crop_size),
                                      #tfs.RandomHorizontalFlip(0.5),
                                      tfs.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.02, 0.02)),
                                      #tfs.RandomRotation(30),
                                      #tfs.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                      tfs.ToTensor()
                                      ])
        self.tfs_val = tfs.Compose([tfs.Resize(self.image_size), tfs.CenterCrop(self.crop_size), tfs.ToTensor()])
        
    def __len__(self):
        return len(self.frame_idxs)

    def __getitem__(self, idx):
        frame_idx = self.frame_idxs[idx]
        
        if self.image_n == 1:
            images = torch.zeros((self.input_n, 3, self.crop_size[0], self.crop_size[1]), dtype=torch.float32)
        if self.image_n == 3:
            images = torch.zeros((self.input_n, 3, self.crop_size[0], self.crop_size[1]*3), dtype=torch.float32)
        if self.image_n == 6:
            images = torch.zeros((self.input_n, 3, self.crop_size[0]*2, self.crop_size[1]*3), dtype=torch.float32)
        
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
            record = self.records[frame_idx+i].split(',')
            imagepath = record[:6]
            x, y, z, w, wx, wy, wz, cx, cy, cz, cw, cwx, cwy, cwz = record[6:]

            ep_t_past[i] = np.array([float(x), float(y), float(z)])
            cs_t_past[i] = np.array([float(cx), float(cy), float(cz)])
            ep_r_past[i] = np.array([float(w), float(wx), float(wy), float(wz)])
            cs_r_past[i] = np.array([float(cw), float(cwx), float(cwy), float(cwz)])

            yaws[i] = Quaternion([w, wx, wy, wz]).yaw_pitch_roll[0]
         
            coors_past[2*i] = float(x)
            coors_past[2*i+1] = float(y)
            
            if self.image_n >= 1:
                image_f = Image.open(self.image_root+imagepath[0]).convert('RGB')
            if self.image_n >= 3:
                image_fr = Image.open(self.image_root+imagepath[1]).convert('RGB')
                image_fl = Image.open(self.image_root+imagepath[5]).convert('RGB')
            if self.image_n == 6:
                image_br = Image.open(self.image_root+imagepath[2]).convert('RGB')
                image_b = Image.open(self.image_root+imagepath[3]).convert('RGB')
                image_bl = Image.open(self.image_root+imagepath[4]).convert('RGB')
            

            if self.phase == 'train' or self.phase == 'mini_train':
                if self.image_n >= 1:
                    image_f = self.tfs_train(image_f)
                if self.image_n >= 3:
                    image_fl = self.tfs_train(image_fl)
                    image_fr = self.tfs_train(image_fr)
                if self.image_n == 6:
                    image_bl = self.tfs_train(image_bl)
                    image_b = self.tfs_train(image_b)
                    image_br = self.tfs_train(image_br)
            else:
                if self.image_n >= 1:
                    image_f = self.tfs_val(image_f)
                if self.image_n >= 3:
                    image_fl = self.tfs_val(image_fl)
                    image_fr = self.tfs_val(image_fr)
                if self.image_n == 6:
                    image_bl = self.tfs_val(image_bl)
                    image_b = self.tfs_val(image_b)
                    image_br = self.tfs_val(image_br)

            #images[i,:3] = image # * yaws[i]
            #images[i,3] = yaws[i]
            if self.image_n == 1:
                images[i] = image_f
            if self.image_n == 3:
                images[i] = torch.concat((image_fl, image_f, image_fr), 2)
            if self.image_n == 6:                
                images[i] = torch.concat((torch.concat((image_fl, image_f, image_fr), 2), torch.flipud(torch.concat((image_bl, image_b, image_br), 2))),1)
                #cv2.imshow('1', np.transpose(images[i].numpy(),(1,2,0)))
                #cv2.waitKey()
            
        egopose_cur = get_global_pose(ep_t_past[-1], ep_r_past[-1], cs_t_past[-1], cs_r_past[-1], inverse=True)

        for j in range(self.predict_n):
            x, y, z, w, wx, wy, wz, cx, cy, cz, cw, cwx, cwy, cwz = self.records[frame_idx+self.input_n+j].strip('\n').split(',')[6:]
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
            if self.res_chain:
                coors_past[0:-2:2] = coors_past[2::2]-coors_past[0:-2:2]  # 增量: 过去一帧的坐标 — 前一帧的坐标 T(-n)-T(-n-1)
                coors_past[1:-2:2] = coors_past[3::2]-coors_past[1:-2:2]
            else:
                coors_past[0:-2:2] = coors_past[0:-2:2]-coors_past[-2]
                coors_past[1:-2:2] = coors_past[1:-2:2]-coors_past[-1]
        
            #coors_past[0:-2:2] = coors_past[0:-2:2]-coors_past[-2]
            #coors_past[1:-2:2] = coors_past[1:-2:2]-coors_past[-1]

        for k in range(self.input_n):
            egopose_past = get_global_pose(ep_t_past[k], ep_r_past[k], cs_t_past[k], cs_r_past[k])
            egopose_past = egopose_cur.dot(egopose_past)
            origin = np.array(egopose_past[:3, 3])
            coors_past[k*2], coors_past[k*2 + 1] = origin[0], origin[1]

        if self.input_n > 1:
            if self.res_chain:
                coors_past[0:-2:2] = coors_past[2::2]-coors_past[0:-2:2]  # 增量: 过去一帧的坐标 — 前一帧的坐标 T(-n)-T(-n-1)
                coors_past[1:-2:2] = coors_past[3::2]-coors_past[1:-2:2]
            else:
                coors_past[0:-2:2] = coors_past[0:-2:2]-coors_past[-2]
                coors_past[1:-2:2] = coors_past[1:-2:2]-coors_past[-1]
        
        #coors_future[2::2] = coors_future[2::2]-coors_future[0:-2:2]
        #coors_future[3::2] = coors_future[3::2]-coors_future[1:-2:2]

        #images = images[:,::-1]
        #images = np.ascontiguousarray(images)

        #for i in range(1, 5):
        #    images[i,4] = coors_past[2*i]
        #    images[i,5] = coors_past[2*i+1]

        return images, coors_past, coors_future # torch.stack((coors_future, coors_future, coors_future), dim=0) 
    
class RNNDataset(Dataset):
    def __init__(self, cfg, phase='train'):
        # image_n: 每一帧环视相机图像有六张图，image_n代表使用其中几张
        # input_n: 输入序列帧数
        # predict_n: 预测未来路径点个数
        
        self.phase = phase  # train/val

        self.image_root = cfg.image_root
        self.image_n = cfg.image_n
        self.input_n = cfg.input_n
        self.predict_n = cfg.predict_n

        self.image_size = cfg.image_size
        self.crop_size = cfg.crop_size

        data_path = cfg.data_path+phase+'/'
        filepaths = glob.glob(data_path + '*.csv')
        #if phase == 'train':
        #    data_path = cfg.data_path+'val/'
        #    filepaths = filepaths+glob.glob(data_path + '*.csv')

        scene_n = len(filepaths)
        self.frame_n = np.zeros(scene_n, np.int32)

        self.records = []

        for i, filepath in enumerate(filepaths):
            f = open(filepath, 'r')
            self.records += f.readlines()
            self.frame_n[i] = len(self.records)

        #self.frame_n = (self.frame_n/6).astype(np.int32)
        
        self.frame_idxs = []
        self.frame_idxs_b = []
        for i in range(scene_n):
            if i == 0:
                self.frame_idxs = [j for j in range(0, self.frame_n[i] - cfg.predict_n - cfg.input_n + 1)]
                self.frame_idxs_b = [0 for j in range(0, self.frame_n[i] - cfg.predict_n - cfg.input_n + 1)]
            else:
                self.frame_idxs = self.frame_idxs + [j for j in range(self.frame_n[i-1], self.frame_n[i] - cfg.predict_n - cfg.input_n + 1)]
                self.frame_idxs_b = self.frame_idxs_b + [self.frame_n[i-1] for j in range(self.frame_n[i-1], self.frame_n[i] - cfg.predict_n - cfg.input_n + 1)]

        #if cfg.image_n == 1:
        #    self.records = self.records[::6]
        #if cfg.image_n == 3:
        #    self.records = [self.records[5::6], self.records[0::6], self.records[1::6]]

        self.tfs_train = tfs.Compose([tfs.Resize(self.image_size),
                                      tfs.RandomCrop(self.crop_size),
                                      #tfs.RandomHorizontalFlip(0.5),
                                      tfs.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.02, 0.02)),
                                      #tfs.RandomRotation(30),
                                      #tfs.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                      tfs.ToTensor()
                                      ])
        self.tfs_val = tfs.Compose([tfs.Resize(self.image_size), tfs.CenterCrop(self.crop_size), tfs.ToTensor()])
        
    def __len__(self):
        return len(self.frame_idxs)
    
    def __getitem__(self, idx):
        frame_idx = self.frame_idxs[idx]
        frame_idx_b = self.frame_idxs[idx]
        input_l = frame_idx - frame_idx_b + self.input_n
        images = torch.zeros((input_l, 3, self.crop_size[0], self.crop_size[1]), dtype=torch.float32)
        coors_past = torch.zeros(2*input_l, dtype=torch.float32)
        yaws = torch.zeros(input_l, dtype=torch.float32)
        coors_future = torch.zeros(2*self.predict_n, dtype=torch.float32)

        for i in range(input_l):
            record = self.records[frame_idx_b+i].strip('\n').split(',')
            imagepath = record[:6]
            x, y, z, w, wx, wy, wz, cx, cy, cz, cw, cwx, cwy, cwz = record[6:]    #max(frame_idx-i, 0)

            yaws[i] = Quaternion([w, wx, wy, wz]).yaw_pitch_roll[0]
            
            coors_past[2*i] = float(x)
            coors_past[2*i+1] = float(y)
                
            #image = Image.open(self.image_root+imagepath[0]).convert('RGB')

            #if self.phase == 'train' or self.phase == 'mini_train':
            #    image = self.tfs_train(image)
            #else:
            #    image = self.tfs_val(image)

            #images[i,:3] = image * yaws[i]
            #images[i,3] = yaws[i]
                
        for j in range(self.predict_n):
            x, y, z, w, wx, wy, wz, cx, cy, cz, cw, cwx, cwy, cwz = self.records[frame_idx+self.input_n+j].strip('\n').split(',')[6:]
            coors_future[j*2], coors_future[j*2 + 1] = float(x), float(y)

        if input_l > 1:
            coors_past[0:-2:2] = coors_past[2::2]-coors_past[0:-2:2]  # 增量: 过去一帧的坐标 — 前一帧的坐标 T(-n)-T(-n-1)
            coors_past[1:-2:2] = coors_past[3::2]-coors_past[1:-2:2]
            
            #coors_past[0:-2:2] = coors_past[0:-2:2]-coors_past[-2]
            #coors_past[1:-2:2] = coors_past[1:-2:2]-coors_past[-1]

        coors_future[0::2] = coors_future[0::2]-coors_past[-2]
        coors_future[1::2] = coors_future[1::2]-coors_past[-1]

        return images, coors_past, coors_future

class RNNDataset_T(Dataset):
    def __init__(self, cfg, phase='train'):
        # image_n: 每一帧环视相机图像有六张图，image_n代表使用其中几张
        # input_n: 输入序列帧数
        # predict_n: 预测未来路径点个数
        
        self.phase = phase  # train/val

        self.image_root = cfg.image_root
        self.image_n = cfg.image_n
        self.input_n = cfg.input_n
        self.predict_n = cfg.predict_n

        self.image_size = cfg.image_size
        self.crop_size = cfg.crop_size

        data_path = cfg.data_path+phase+'/'
        filepaths = glob.glob(data_path + '*.csv')
        #if phase == 'train':
        #    data_path = cfg.data_path+'val/'
        #    filepaths = filepaths+glob.glob(data_path + '*.csv')

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

        if cfg.image_n == 1:
            self.records = self.records[::6]
        if cfg.image_n == 3:
            self.records = [self.records[5::6], self.records[0::6], self.records[1::6]]

        self.tfs_train = tfs.Compose([tfs.Resize(self.image_size),
                                      tfs.RandomCrop(self.crop_size),
                                      #tfs.RandomHorizontalFlip(0.5),
                                      tfs.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.02, 0.02)),
                                      #tfs.RandomRotation(30),
                                      #tfs.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                      tfs.ToTensor()
                                      ])
        self.tfs_val = tfs.Compose([tfs.Resize(self.image_size), tfs.CenterCrop(self.crop_size), tfs.ToTensor()])
        
    def __len__(self):
        return len(self.frame_idxs)

    def __getitem__(self, idx):
        frame_idx = self.frame_idxs[idx]

        images = torch.zeros((self.input_n, 3, self.crop_size[0], self.crop_size[1]), dtype=torch.float32)
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
            imagepath, x, y, z, w, wx, wy, wz, cx, cy, cz, cw, cwx, cwy, cwz = self.records[frame_idx+i].strip('\n').split(',')    #max(frame_idx-i, 0)

            ep_t_past[i] = np.array([float(x), float(y), float(z)])
            cs_t_past[i] = np.array([float(cx), float(cy), float(cz)])
            ep_r_past[i] = np.array([float(w), float(wx), float(wy), float(wz)])
            cs_r_past[i] = np.array([float(cw), float(cwx), float(cwy), float(cwz)])

            yaws[i] = Quaternion([w, wx, wy, wz]).yaw_pitch_roll[0]
         
            coors_past[2*i] = float(x)
            coors_past[2*i+1] = float(y)
            
            image = Image.open(self.image_root+imagepath).convert('RGB')

            if self.phase == 'train' or self.phase == 'mini_train':
                image = self.tfs_train(image)
            else:
                image = self.tfs_val(image)

            images[i,:3] = image # * yaws[i]
            #images[i,3] = yaws[i]
        
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

        for k in range(self.input_n):
            egopose_past = get_global_pose(ep_t_past[k], ep_r_past[k], cs_t_past[k], cs_r_past[k])
            egopose_past = egopose_cur.dot(egopose_past)
            origin = np.array(egopose_past[:3, 3])
            coors_past[k*2], coors_past[k*2 + 1] = origin[0], origin[1]

        #if self.input_n > 1:
        #    coors_past[0:-2:2] = coors_past[2::2]-coors_past[0:-2:2]  # 增量: 过去一帧的坐标 — 前一帧的坐标 T(-n)-T(-n-1)
        #    coors_past[1:-2:2] = coors_past[3::2]-coors_past[1:-2:2]
        
            #coors_past[0:-2:2] = coors_past[0:-2:2]-coors_past[-2]
            #coors_past[1:-2:2] = coors_past[1:-2:2]-coors_past[-1]

        #images = images[:,::-1]
        #images = np.ascontiguousarray(images)

        #for i in range(1, 5):
        #    images[i,4] = coors_past[2*i]
        #    images[i,5] = coors_past[2*i+1]

        return images, coors_past, coors_future # torch.stack((coors_future, coors_future, coors_future), dim=0) 

class OpenSceneDataset(Dataset):
    def __init__(self, cfg, phase='train'):
        # image_n: 每一帧环视相机图像有六张图，image_n代表使用其中几张
        # input_n: 输入序列帧数
        # predict_n: 预测未来路径点个数
        
        self.phase = phase  # train/val
        self.res_chain = cfg.res_chain
        self.image_root = cfg.image_root
        self.image_n = cfg.image_n
        self.input_n = cfg.input_n
        self.predict_n = cfg.predict_n

        self.image_size = cfg.image_size
        self.crop_size = cfg.crop_size

        data_path = cfg.data_path+phase+'/'
        filepaths = glob.glob(data_path + '*.csv')
        #if phase == 'train':
        #    data_path = cfg.data_path+'val/'
        #    filepaths = filepaths+glob.glob(data_path + '*.csv')

        scene_n = len(filepaths)
        self.frame_n = np.zeros(scene_n, np.int32)

        self.records = []

        for i, filepath in enumerate(filepaths):
            f = open(filepath, 'r')
            self.records += f.readlines()
            self.frame_n[i] = len(self.records)
        
        self.frame_idxs = []
        for i in range(scene_n):
            if i == 0:
                self.frame_idxs = [j for j in range(0, self.frame_n[i] - cfg.predict_n - cfg.input_n + 1)]
            else:
                self.frame_idxs = self.frame_idxs + [j for j in range(self.frame_n[i-1], self.frame_n[i] - cfg.predict_n - cfg.input_n + 1)]

        #if cfg.image_n == 1:
        #    self.records = self.records[::6]
        #if cfg.image_n == 3:
        #    self.records = [self.records[5::6], self.records[0::6], self.records[1::6]]

        self.tfs_train = tfs.Compose([tfs.Resize(self.image_size),
                                      tfs.RandomCrop(self.crop_size),
                                      #tfs.RandomHorizontalFlip(0.5),
                                      tfs.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.02, 0.02)),
                                      #tfs.RandomRotation(30),
                                      #tfs.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                      tfs.ToTensor()
                                      ])
        self.tfs_val = tfs.Compose([tfs.Resize(self.image_size), tfs.CenterCrop(self.crop_size), tfs.ToTensor()])
        
    def __len__(self):
        return len(self.frame_idxs)

    def __getitem__(self, idx):
        frame_idx = self.frame_idxs[idx]

        images = torch.zeros((self.input_n, 3, self.crop_size[0], self.crop_size[1]), dtype=torch.float32)
        coors_past = torch.zeros(2*self.input_n, dtype=torch.float32)
        yaws = torch.zeros(self.input_n, dtype=torch.float32)
        coors_future = torch.zeros(2*self.predict_n, dtype=torch.float32)

        for i in range(self.input_n):
            record = self.records[frame_idx+i].split(',')
            x, y, z, w, wx, wy, wz = record[8:15]    #max(frame_idx-i, 0)

            yaws[i] = Quaternion([w, wx, wy, wz]).yaw_pitch_roll[0]
         
            coors_past[2*i] = float(x)
            coors_past[2*i+1] = float(y)
            
            #image = Image.open(self.image_root+imagepath).convert('RGB')

            #if self.phase == 'train' or self.phase == 'mini_train':
            #    image = self.tfs_train(image)
            #else:
            #    image = self.tfs_val(image)

            #images[i,:3] = image * yaws[i]
            #images[i,3] = yaws[i]
            
        for j in range(self.predict_n):
            record = self.records[frame_idx+self.input_n+j].split(',')
            x, y, z, w, wx, wy, wz = record[8:15]    #max(frame_idx-i, 0)
            coors_future[j*2], coors_future[j*2 + 1] = float(x), float(y)

        
        if self.input_n > 1:
            if self.res_chain:
                coors_past[0:-2:2] = coors_past[2::2]-coors_past[0:-2:2]  # 增量: 过去一帧的坐标 — 前一帧的坐标 T(-n)-T(-n-1)
                coors_past[1:-2:2] = coors_past[3::2]-coors_past[1:-2:2]
            else:
                coors_past[0:-2:2] = coors_past[0:-2:2]-coors_past[-2]
                coors_past[1:-2:2] = coors_past[1:-2:2]-coors_past[-1]

        coors_future[0::2] = coors_future[0::2]-coors_past[-2]
        coors_future[1::2] = coors_future[1::2]-coors_past[-1]

        
        # (coors_past[-2], coors_past[-1])是车辆当前位置的坐标,coors_future被处理成相对当前位置的增量
        #coors_future[0::2] = coors_future[0::2]-coors_past[-2]
        #coors_future[1::2] = coors_future[1::2]-coors_past[-1]

        #images = images[:,::-1]
        #images = np.ascontiguousarray(images)

        #for i in range(1, 5):
        #    images[i,4] = coors_past[2*i]
        #    images[i,5] = coors_past[2*i+1]

        return images, coors_past, coors_future # torch.stack((coors_future, coors_future, coors_future), dim=0) 
    
def collate_fn(data):
    b = len(data)
    c,w,h = data[0][0].size(1), data[0][0].size(2), data[0][0].size(3)
    l_min = data[0][0].size(0)
    for i in range(b):
        l_min = min(l_min, data[i][0].size(0))
    images = torch.zeros((b,l_min,c,w,h))
    coors_past = torch.zeros((b,l_min*2))
    coors_future = torch.zeros(b, data[0][2].size(0))
    for i in range(b):
        images[i] = data[i][0][(data[i][0].size(0)-l_min):]
        coors_past[i] = data[i][1][(data[i][1].size(0)-2*l_min):]
        coors_future[i] = data[i][2]
    return images, coors_past, coors_future

def collate_fn1(data):
    l_max = 0
    b = len(data)
    c,w,h = data[0][0].size(1), data[0][0].size(2), data[0][0].size(3)
    for i in range(b):
        l_max = max(l_max, data[i][0].size(0))
    images = torch.zeros((b,l_max,c,w,h))
    coors_past = torch.zeros((b,l_max*2))
    coors_future = torch.zeros(b, data[0][2].size(0))
    for i in range(b):
        images[i,(l_max - data[i][0].size(0)):] = data[i][0]
        coors_past[i,(l_max*2-data[i][1].size(0)):] = data[i][1]
        coors_future[i] = data[i][2]
    return images, coors_past, coors_future

def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='total number of training epochs')
    parser.add_argument('--device', type=str, default='0', help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--learning-rate', type=int, default=0.01, help='initial learning rate')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')

    parser.add_argument('--image-root', type=str, default='D:/datasets/nuscenes/trainval/', help='')
    parser.add_argument('--data-path', type=str, default='data/nuscenes/', help='')
    parser.add_argument('--image-n', type=int, default=1, help='每一帧环视相机图像有六张图, image_n代表使用其中几张')
    parser.add_argument('--input-n', type=int, default=4, help='输入序列帧数')
    parser.add_argument('--predict-n', type=int, default=6, help='')
    parser.add_argument('--image-size', type=int, default=[800, 450], help='')
    parser.add_argument('--crop-size', type=int, default=[704, 384], help='')

    return parser.parse_args()

# 测试
if __name__=='__main__':
    cfg = parse_cfg()
    train_data = RNNDataset(cfg, phase='train')
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    for i, (image, coor_past, coor_future) in enumerate(train_dataloader):
        print(coor_past)
        #cv2.imshow('image', image.numpy()[0].transpose((1, 2, 0)))
        #cv2.waitKey(1)
        #print(target)  #'samples/CAM_FRONT/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915628412465.jpg' scene-0170
