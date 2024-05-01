from torch.utils.data import Dataset
import os
import csv
from pyquaternion import Quaternion
import numpy as np
import torch
from PIL import Image

class ResNetDataset(Dataset):
    # phase means the phase of the dataset, e.g. "mini_train", "mini_val"
    def __init__(self, csv_root=None,data_root=None,phase="mini_train", transform=None):
        self.data_dict = {}
        self.data_root = data_root
        # self.img_shape = img_shape
        self.transform = transform
        idx = 0 # idx is the valid index as the key of data_dict, e.g.0,1,2...33,40,41,42...73,80
        for phase_csv in os.listdir(os.path.join(csv_root, phase)):
            filename = os.path.join(csv_root, phase, phase_csv)
            with open(filename,'r') as csvfile:
                # select all "CAM_FRONT" rows
                csv_list = []
                for row in csv.reader(csvfile):
                    if "CAM_FRONT" in row[0].split('/'):
                        csv_list.append(row)    

                for i, row in enumerate(csv_list): 
                    img_path, x, y = row[0], float(row[1]), float(row[2])
                    w, wx, wy, wz = float(row[4]), float(row[5]), float(row[6]), float(row[7])
                    yaw = Quaternion([w, wx, wy, wz]).yaw_pitch_roll[0]
                    target = []
                    # in each scene with only "CAM_FRONT", the last 6 rows cannot be used as input
                    if i <= len(csv_list) - 7:  
                        for t in range(1,7):
                            x_t, y_t = float(csv_list[i+t][1]), float(csv_list[i+t][2])
                            target.append([x_t, y_t])
                    self.data_dict[str(idx)] = {"img_path":img_path, "coord":(x,y), "target":target, "yaw":yaw}
                    idx += 1 
        # print(self.data_dict)

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        key_list = list(self.data_dict.keys())
        
        key = key_list[idx]
        img_path, coord, target, yaw = self.data_dict[key]["img_path"], self.data_dict[key]["coord"], \
            self.data_dict[key]["target"], self.data_dict[key]["yaw"]
        
        img = Image.open(os.path.join(self.data_root, img_path))
        
        if(img.mode!='RGB'):
            img = img.convert("RGB")
        img = self.transform(img)
        target = np.array(target)
            
        return img, torch.tensor(coord[0]), torch.tensor(coord[1]), torch.tensor(target), torch.tensor(yaw).float()

    
class SeqResNetDataset(Dataset):
    # phase means the phase of the dataset, e.g. "mini_train", "mini_val"
    def __init__(self, csv_root=None,data_root=None,phase="mini_train", transform=None):
        self.data_dict = {}
        self.data_root = data_root
        # self.img_shape = img_shape
        self.transform = transform
        idx = 0 # idx is the valid index as the key of data_dict, e.g. 0...33,40,41...73,80...
        for phase_csv in os.listdir(os.path.join(csv_root, phase)):
            filename = os.path.join(csv_root, phase, phase_csv)
            with open(filename,'r') as csvfile:
                # select all "CAM_FRONT" rows
                csv_list = []
                for row in csv.reader(csvfile):
                    if "CAM_FRONT" in row[0].split('/'):
                        csv_list.append(row)    

                for i, row in enumerate(csv_list): 
                    # in each scene with only "CAM_FRONT", the first 4 rows and the last 6 rows cannot be used as input
                    if i < 4 or i > len(csv_list) - 7:
                        continue

                    img_path, x, y = row[0], float(row[1]), float(row[2])
                    # w, wx, wy, wz = float(row[4]), float(row[5]), float(row[6]), float(row[7])
                    # yaw = Quaternion([w, wx, wy, wz]).yaw_pitch_roll[0]
                    img_paths, coords, target = [], [], [] 

                    for t in range(4, 0, -1): # coordinates and images of 5 input frames 
                        x_t, y_t = float(csv_list[i-t][1]), float(csv_list[i-t][2])
                        coords.append([x_t, y_t])
                        img_paths.append(csv_list[i-t][0])
                    coords.append([x, y])
                    img_paths.append(img_path)

                    for t in range(1,7):
                        x_t, y_t = float(csv_list[i+t][1]), float(csv_list[i+t][2])
                        target.append([x_t, y_t])
                    self.data_dict[str(idx)] = {"img_path":img_paths, "coord":coords, "target":target} # "yaw":yaw
                    idx += 1 
        # print(self.data_dict)

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        key_list = list(self.data_dict.keys())
        
        key = key_list[idx]
        img_paths, coords, target = self.data_dict[key]["img_path"], self.data_dict[key]["coord"], self.data_dict[key]["target"] # yaw
        
        imgs = []
        for img_path in img_paths:
            img = Image.open(os.path.join(self.data_root, img_path))
            if(img.mode!='RGB'):
                img = img.convert("RGB")
            img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)
        coords = np.array(coords)
        target = np.array(target)

        # normalize coords
        sigmoid_normalized_coordinates = 1 / (1 + np.exp(-coords))
        normalized_coordinates = 2 * sigmoid_normalized_coordinates - 1
        return_dict = {"imgs":imgs, "coordinates":torch.tensor(normalized_coordinates), "target":torch.tensor(target)}
        
        return return_dict  #torch.tensor(yaw).float()
                    


# val_set = ResNetDataset(csv_root="./unified-driving/data/nuscenes/image_scenes/",
#                         data_root = "./Data/v1.0-mini/v1.0-mini/",
#                         phase="mini_val",
#                         transform=None)
