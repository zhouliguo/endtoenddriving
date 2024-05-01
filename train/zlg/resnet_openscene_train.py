import sys 
sys.path.append('.')

import os
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_preparation.zlg.nuscenes_data import OpenSceneDataset as Dataset

from networks.zlg.resnet_nuscenes import FCModel, ResNetModel, ResNetModel_C

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

    train_dataloader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # 第2步：构建网络，设置训练参数：学习率、学习率衰减策略、优化函数（SDG、Adam、……）、损失函数、……

    #model = ResNetModel(type=50, pretrained=True, input_n=cfg.input_n, out_features=12).to(device)
    #model = ResNetModel_C(type=50, pretrained=True, input_n=cfg.input_n, out_features=12).to(device)
    model = FCModel(pretrained=True, input_n=cfg.input_n, out_features=16).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr_init)

    # 学习率衰减策略
    lf = lambda x: (1 - x / cfg.epochs) * (1.0 - cfg.lr_final) + cfg.lr_final
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    #loss_function = torch.nn.L1Loss()
    loss_function = torch.nn.MSELoss()

    # 第3步：循环读取数据训练网络
    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)
    f_val_loss = open(cfg.save_path+'/val_loss.csv', 'w')
    
    loss_min = float('inf')
    epoch_best = 0

    for epoch_i in range(0, cfg.epochs):
        model.train()

        loss_sum = 0
        
        for train_i, (image, coor_past, coor_future) in enumerate(train_dataloader):
            image = image.to(device)
            coor_future = coor_future.to(device)
            coor_past = coor_past[:,:-2].to(device)  # coor_past[:,0:-2]是后一个点相对于前一个点的增量, coor_past[:,-2:]是当前点的绝对坐标，
            delta_sum_xy = torch.zeros((coor_past.size(0), 2)).to(device)
            #delta_sum_xy = torch.zeros((coor_past.size(0), 3, 2)).to(device)

            optimizer.zero_grad()

            output = model(coor_past)
            #output = model([image, coor_past])

            if cfg.res_chain:
                for i in range(1, cfg.predict_n):
                    delta_sum_xy[:,0] = delta_sum_xy[:,0] + output[:,2*(i-1)]
                    delta_sum_xy[:,1] = delta_sum_xy[:,1] + output[:,2*(i-1)+1]
                    coor_future[:,2*i] = coor_future[:,2*i] - delta_sum_xy[:,0]
                    coor_future[:,2*i+1] = coor_future[:,2*i+1] - delta_sum_xy[:,1]

                #for i in range(1, cfg.predict_n):
                #    delta_sum_xy[:,:,0] = delta_sum_xy[:,:,0] + output[:,:,2*(i-1)]
                #    delta_sum_xy[:,:,1] = delta_sum_xy[:,:,1] + output[:,:,2*(i-1)+1]
                #    coor_future[:,:,2*i] = coor_future[:,:,2*i] - delta_sum_xy[:,:,0]
                #    coor_future[:,:,2*i+1] = coor_future[:,:,2*i+1] - delta_sum_xy[:,:,1]

            loss = loss_function(output, coor_future)
            loss_sum = loss_sum + loss.item()
            if (train_i+1)%100 == 0:
                lr = [x['lr'] for x in optimizer.param_groups]
                print('Train Epoch:', epoch_i, 'Step:', train_i, 'Loss:', loss_sum/100, 'Learning Rate:', lr)
                loss_sum = 0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
        
        scheduler.step()
        
        # 训练完每个epoch进行验证
        #model = torch.load('weights/resnet_nuscenes/hao_Noimgae_01.pt')
        model.eval()
        with torch.no_grad():
            loss_sum = 0
            #loss_sum1 = 0
            #loss_sum2 = 0
            l = len(val_dataloader)
            for val_i, (image, coor_past, coor_future) in enumerate(val_dataloader):
                image = image.to(device)
                coor_future = coor_future.to(device)
                coor_past = coor_past[:,:-2].to(device)  # coor_past[:,0:-2]是后一个点相对于前一个点的增量, coor_past[:,-2:]是当前点的绝对坐标，

                output = model(coor_past)
                #output = model([image, coor_past])

                if cfg.res_chain:
                    for i in range(1, cfg.predict_n):
                        output[:,2*i] = output[:,2*i] + output[:,2*(i-1)]
                        output[:,2*i+1] = output[:,2*i+1] + output[:,2*(i-1)+1]

                    #for i in range(1, cfg.predict_n):
                    #    output[:,:,2*i] = output[:,:,2*i] + output[:,:,2*(i-1)]
                    #    output[:,:,2*i+1] = output[:,:,2*i+1] + output[:,:,2*(i-1)+1]

                loss = loss_function(output, coor_future)
                #loss = loss_function(output[:,0,], coor_future[:,0,])
                #loss1 = loss_function(output[:,1,], coor_future[:,1,])
                #loss2 = loss_function(output[:,2,], coor_future[:,2,])
                loss_sum = loss_sum + loss.item()
                #loss_sum1 = loss_sum1 + loss1.item()
                #loss_sum2 = loss_sum2 + loss2.item()
            #loss_sum = min(loss_sum,loss_sum1,loss_sum2)
            if loss_sum < loss_min:
                loss_min = loss_sum
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
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
    parser.add_argument('--save-path', type=str, default='weights/resnet_nuscenes', help='path to save checkpoint')
    parser.add_argument('--lr-init', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--lr-final', type=float, default=0.0001, help='最终学习率')
    parser.add_argument('--res-chain', type=int, default=1, help='1或0, 表示使用或不使用residual chain loss')

    parser.add_argument('--image-root', type=str, default='D:/datasets/nuscenes/trainval/', help='图像路径的根路径')
    parser.add_argument('--data-path', type=str, default='data/openscene/', help='用于保存图像路径和标签的csv文件的路径')
    parser.add_argument('--image-n', type=int, default=1, help='每一帧环视相机图像有六张图, image_n代表使用其中几张')
    parser.add_argument('--input-n', type=int, default=8, help='输入序列帧数')
    parser.add_argument('--predict-n', type=int, default=8, help='')
    parser.add_argument('--image-size', type=int, default=[450, 800], help='[h, w]')
    parser.add_argument('--crop-size', type=int, default=[384, 704], help='[h, w]')

    return parser.parse_args()

if __name__ == '__main__':
    cfg = parse_cfg()
    train_val(cfg)

# python train/resnet_openscene_train.py --image-root /opt/data/private/nuscenes/trainval/ --save-path weights/resnet_openscene1 --device 1 --batch-size 128 --num-workers 8
# python train/resnet_nuscenes_train.py --image-root /mnt/ssd/nuscenes/trainval/ --save-path weights/resnet_nuscenes_1 --device 0 --batch-size 36 --num-workers 8