import sys 
sys.path.append('.')

import os
import argparse
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_preparation.zlg.nuscenes_data import NuScenesDataset_T as Dataset

from networks.zlg.resnet_nuscenes import ResNetModel_C, FCModel

# 训练与验证
def test(cfg):

    if cfg.device == 'cpu':
        device = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device
        device = torch.device('cuda:0')

    # 第1步：构建数据读取迭代器

    #train_data = Dataset(cfg=cfg, phase='train')
    #train_dataloader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    test_data = Dataset(cfg=cfg, phase='val')
    test_dataloader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    loss_function = torch.nn.MSELoss()

    model = torch.load(cfg.weights).to(device)
    model.eval()
    with torch.no_grad():
        loss_sum = 0
        l = len(test_dataloader)
        mse = 0
        f_val_loss = open(cfg.save_path+'/rnn5.csv', 'w')
        f_val_loss.write('Val Loss\n')
        for train_i, (image, coor_past, coor_future) in enumerate(test_dataloader):
            coor_current = coor_past[:,-2:]
            image = image.to(device)
            coor_past = coor_past[:,:-2].to(device)
            coor_future = coor_future.to(device)
            delta_sum_xy = torch.zeros((coor_past.size(0), 2)).to(device)

            output = model(coor_past)

            if cfg.res_chain:
                for i in range(1, cfg.predict_n):
                    output[:,2*i] = output[:,2*i] + output[:,2*(i-1)]
                    output[:,2*i+1] = output[:,2*i+1] + output[:,2*(i-1)+1]

            loss = loss_function(output, coor_future)

            diff = (output-coor_future).cpu().numpy()
            diff = diff*diff
            diff = np.mean(diff)
            print('mse', diff)

            #print('output', output.cpu().numpy())
            #print('coor_future', coor_future.cpu().numpy())
            
            mse = mse+diff
            f_val_loss.write(str(loss.item())+'\n')
        print(mse/l)
        f_val_loss.close()
        torch.cuda.empty_cache()


def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/25.pt', help='total number of training epochs')
    parser.add_argument('--epochs', type=int, default=100, help='total number of training epochs')
    parser.add_argument('--device', type=str, default='0', help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')
    parser.add_argument('--save-path', type=str, default='weights/', help='path to save checkpoint')
    parser.add_argument('--res-chain', type=int, default=1, help='1或0, 表示使用或不使用residual chain loss')

    parser.add_argument('--image-root', type=str, default='D:/datasets/nuscenes/trainval/', help='图像路径的根路径')
    parser.add_argument('--data-path', type=str, default='data/nuscenes/', help='用于保存图像路径和标签的csv文件的路径')
    parser.add_argument('--image-n', type=int, default=1, help='每一帧环视相机图像有六张图, image_n代表使用其中几张')
    parser.add_argument('--input-n', type=int, default=4, help='输入序列帧数')
    parser.add_argument('--predict-n', type=int, default=6, help='')
    parser.add_argument('--image-size', type=int, default=[450, 800], help='[h, w]')
    parser.add_argument('--crop-size', type=int, default=[384, 704], help='[h, w]')

    return parser.parse_args()

if __name__ == '__main__':
    cfg = parse_cfg()
    
    test(cfg)

# python test/resnet_nuscenes_test.py --epochs 300 --save-path weights/fc_delta_nuscenes_l2