import sys 
sys.path.append('.')

import os
import argparse
import numpy as np
import random
import math

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_preparation.zlg.nuscenes_data import NuScenesDataset_T as Dataset

from networks.zlg.rnn_nuscenes import LSTMCellModel1 as Model
from networks.zlg.resnet_nuscenes import FCModel

# 自定义损失函数
class CustomLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self):
        return

# 训练与验证
def train_val(cfg):

    if cfg.device == 'cpu':
        device = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device
        device = torch.device('cuda:0')

    # 第1步：构建数据读取迭代器

    train_data = Dataset(cfg=cfg, phase='train')
    val_data = Dataset(cfg=cfg, phase='val')

    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # 第2步：构建网络，设置训练参数：学习率、学习率衰减策略、优化函数（SDG、Adam、……）、损失函数、……

    #model = FCModel(pretrained=True, input_n=cfg.input_n, out_features=2*cfg.predict_n).to(device)
    model = Model(pretrained=True, input_n=cfg.input_n, in_feature=256, out_features=2*cfg.predict_n).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr_init)

    # 学习率衰减策略
    lf = lambda x: (1 - x / cfg.epochs) * (1.0 - cfg.lr_final) + cfg.lr_final
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    loss_function = torch.nn.MSELoss()

    # 第3步：循环读取数据训练网络
    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)
    f_val_loss = open(cfg.save_path+'/val_loss.csv', 'w')
    
    loss_min = float('inf')
    epoch_best = 0

    tl = len(train_dataloader)
    tl_h = int(tl/3)
    losses = np.zeros((tl))
    coor_pasts = torch.zeros((tl, cfg.input_n*2))
    coor_futures = torch.zeros((tl, cfg.predict_n*2))

    tl_v = len(val_dataloader)
    coor_pasts_v = torch.zeros((tl_v, cfg.input_n*2))
    coor_futures_v = torch.zeros((tl_v, cfg.predict_n*2))

    for train_i, (image, coor_past, coor_future) in enumerate(train_dataloader):
        coor_pasts[train_i] = coor_past
        coor_futures[train_i] = coor_future
        #break

    for val_i, (image, coor_past, coor_future) in enumerate(val_dataloader):
        coor_pasts_v[val_i] = coor_past
        coor_futures_v[val_i] = coor_future
        #break

    for epoch_i in range(0, cfg.epochs):
        tl = len(coor_pasts)
        index_shuffle = [i for i in range(tl)]
        random.shuffle(index_shuffle)
        coor_pasts = coor_pasts[index_shuffle]
        coor_futures = coor_futures[index_shuffle]
    
        model.train()

        loss_sum = 0
        
        #for train_i, (image, coor_past, coor_future) in enumerate(train_dataloader):
        for train_i in range(0, tl, cfg.batch_size):
            if tl - train_i >= cfg.batch_size:
                coor_past = coor_pasts[train_i:train_i+cfg.batch_size][:,:-2].to(device)
                coor_future = coor_futures[train_i:train_i+cfg.batch_size].to(device)
            else:
                coor_past = coor_pasts[train_i:][:,:-2].to(device)
                coor_future = coor_futures[train_i:].to(device)
            #image = image.to(device)
            #coor_future = coor_future.to(device)
            #coor_past = coor_past[:,:-2].to(device)

            delta_sum_xy = torch.zeros((coor_future.size(0), 2)).to(device)

            optimizer.zero_grad()

            output = model(coor_past)
            #output = model([image, coor_past])
            #k = torch.zeros((coor_future.size(0), 12)).to(device)
            #for i in range(6):
            #    k[:,i*2:i*2+2] = output[:,i,-2:]
            
            if cfg.res_chain:
                '''
                coor_future[:,2::2] = coor_future[:,2::2] - coor_future[:,0:-2:2]
                coor_future[:,3::2] = coor_future[:,3::2] - coor_future[:,1:-2:2]
                '''
                for i in range(1, cfg.predict_n):
                    delta_sum_xy[:,0] = delta_sum_xy[:,0] + output[:,2*(i-1)]
                    delta_sum_xy[:,1] = delta_sum_xy[:,1] + output[:,2*(i-1)+1]
                    coor_future[:,2*i] = coor_future[:,2*i] - delta_sum_xy[:,0]
                    coor_future[:,2*i+1] = coor_future[:,2*i+1] - delta_sum_xy[:,1]
                
                '''
                for i in range(1, cfg.predict_n):
                    delta_sum_xy[:,0] = delta_sum_xy[:,0] + k[:,2*(i-1)]
                    delta_sum_xy[:,1] = delta_sum_xy[:,1] + k[:,2*(i-1)+1]
                    coor_future[:,2*i] = coor_future[:,2*i] - delta_sum_xy[:,0]
                    coor_future[:,2*i+1] = coor_future[:,2*i+1] - delta_sum_xy[:,1]
                '''

            #coor_future = torch.cat((coor_past, coor_future), dim=1)

            #target = torch.zeros((coor_future.size(0), 6, 8)).to(device)

            #for i in range(6):
            #    target[:,i] = coor_future[:,(i+1)*2:(i+1)*2+8]

            #coor_future = torch.cat((coor_past[:,2:], coor_future), dim=1)
            loss = loss_function(output, coor_future)
            loss_sum = loss_sum + loss.item()
            if train_i/cfg.batch_size%100 == 0 and train_i>0:
                lr = [x['lr'] for x in optimizer.param_groups]
                print('Epoch:', epoch_i, 'Step:', train_i/cfg.batch_size, 'Train Loss:', loss_sum/100, 'Learning Rate:', lr)
                loss_sum = 0

            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # 训练完每个epoch进行验证
        model.eval()
        with torch.no_grad():
            loss_sum = 0
            l = math.ceil(len(val_dataloader)/cfg.batch_size)
            #for val_i, (image, coor_past, coor_future) in enumerate(val_dataloader):
            for val_i in range(0, tl_v, cfg.batch_size):
                if tl - val_i >= cfg.batch_size:
                    coor_past = coor_pasts_v[val_i:val_i+cfg.batch_size][:,:-2].to(device)
                    coor_future = coor_futures_v[val_i:val_i+cfg.batch_size].to(device)
                else:
                    coor_past = coor_pasts_v[val_i:][:,:-2].to(device)
                    coor_future = coor_futures_v[val_i:].to(device)
                #image = image.to(device)
                #coor_past = coor_past[:,:-2].to(device)
                #coor_future = coor_future.to(device)

                output = model(coor_past)
                #output = model([image, coor_past])

                #output = output[:,6:]

                #k = torch.zeros((coor_future.size(0), 12)).to(device)
                #for i in range(6):
                #    k[:,i*2:i*2+2] = output[:,i,-2:]

                if cfg.res_chain:
                    for i in range(1, cfg.predict_n):
                        output[:,2*i] = output[:,2*i] + output[:,2*(i-1)]
                        output[:,2*i+1] = output[:,2*i+1] + output[:,2*(i-1)+1]
                

                #for i in range(1, cfg.predict_n):
                #    k[:,2*i] = k[:,2*i] + k[:,2*(i-1)]
                #    k[:,2*i+1] = k[:,2*i+1] + k[:,2*(i-1)+1]

                loss = loss_function(output, coor_future)
                loss_sum = loss_sum + loss.item()
            if loss_sum<loss_min:
                loss_min=loss_sum
                epoch_best = epoch_i
            print('Val Epoch:', epoch_i, 'Loss:', loss_sum/l, 'Best Epoch:', epoch_best, 'Loss', loss_min/l)
            f_val_loss.write('Val Epoch:, ' + str(epoch_i) + ', Loss:, ' + str(loss_sum/l) + ', Best Epoch:, ' + str(epoch_best) + ', Loss:, ' + str(loss_min/l)+'\n')
            
            f_val_loss.flush()

        torch.save(model, cfg.save_path+'/'+str(epoch_i)+'.pt')
        torch.cuda.empty_cache()
    f_val_loss.close()

def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='total number of training epochs')
    parser.add_argument('--device', type=str, default='0', help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
    parser.add_argument('--save-path', type=str, default='weights/rnn_nuscenes4', help='path to save checkpoint')
    parser.add_argument('--lr-init', type=float, default=0.01, help='path to save checkpoint')
    parser.add_argument('--lr-final', type=float, default=0.0001, help='path to save checkpoint')
    parser.add_argument('--res-chain', type=int, default=0, help='1或0, 表示使用或不使用residual chain loss')

    parser.add_argument('--image-root', type=str, default='D:/datasets/nuscenes/trainval/', help='图像路径的根路径')
    parser.add_argument('--data-path', type=str, default='data/nuscenes/', help='用于保存图像路径和标签的csv文件的路径')
    parser.add_argument('--image-n', type=int, default=1, help='每一帧环视相机图像有六张图, image_n代表使用其中几张')
    parser.add_argument('--input-n', type=int, default=4, help='输入序列帧数')
    parser.add_argument('--predict-n', type=int, default=6, help='')
    parser.add_argument('--image-size', type=int, default=[450, 800], help='')
    parser.add_argument('--crop-size', type=int, default=[384, 704], help='')

    return parser.parse_args()

if __name__ == '__main__':
    cfg = parse_cfg()
    
    train_val(cfg)

# python train/rnn_nuscenes_train.py --image-root /opt/data/private/nuscenes/trainval/ --save-path weights/rnn_nuscenes --device 3 --batch-size 36 --num-workers 8 --input-n 5

# shi yan ji lu
# yi zhang tu
# train loss: 37.32 22.46 14.26 5.24 3.51
# val loss: 87.16 58.61 70.77 52.98 55.39
# yi zhang tu, cheng yi jiao du:
# train loss: 10.71 5.10 3.59 2.52 1.70
# val loss: 16.06 16.22 16.46 14.10 13.67
# san zhang tu he bing, ge zi cheng yi jiao du: 
# train loss: 20.67 13.35 10.50 7.37
# val loss: 30.72 37.00 19.82 45.98