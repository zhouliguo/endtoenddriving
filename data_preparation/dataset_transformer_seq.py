from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import glob
import cv2
import random
import argparse

from PIL import Image
from pyquaternion import Quaternion
import torchvision.transforms as tfs

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

class NuScenesDataset_T(Dataset):
    def __init__(self, cfg, phase='train'):
        # image_n: 每一帧环视相机图像有六张图，image_n代表使用其中几张
        # input_n: 输入序列帧数
        # predict_n: 预测未来路径点个数
        
        self.phase = phase  # train/val

        self.image_root = cfg.data_root
        self.image_n = cfg.image_n
        self.input_n = cfg.input_n
        self.predict_n = cfg.predict_n

        self.image_size = cfg.image_size
        self.crop_size = cfg.crop_size

        data_path = cfg.data_path+phase+'/'
        if cfg.model_type == 'TransformerCoor_T':
            data_path = cfg.data_path_T+phase+'/'
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

        return coors_past, coors_future # images, coors_past, coors_future

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

        return images_list[::-1], deta_coor, target
  
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
    train_data = ResNetDataset(cfg, phase='train')
    # train_dataloader = DataLoader(train_data, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    train_dataloader = DataLoader(train_data, batch_size=10, shuffle=True, num_workers=0, pin_memory=True)
    for i, (output, coor, target) in enumerate(train_dataloader):
        if i == 3:
            print(i, output[0].shape, len(output),coor,target) # torch.Size([16, 6, 384, 704])
        #cv2.imshow('image', image.numpy()[0].transpose((1, 2, 0)))
        #cv2.waitKey(1)
        #print(target)  #'samples/CAM_FRONT/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915628412465.jpg' scene-0170
