import sys 
sys.path.append('.')

import os
import argparse

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_preparation.dataset_transformer_seq02 import NuscenesDatasetCoor, NuscenesDatasetImage
from data_preparation.dataset_transformer_seq import ResNetDataset, NuScenesDataset_T

from networks.lstm_rnn import ImagePositionEnhancementResMet50, ImageResNet50_LSTM, Transformer_image
from networks.Models import Transformer

# 训练与验证
def train_val(cfg):

    if cfg.device == 'cpu':
        device = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device
        device = torch.device('cuda:0')

    # 第1步：构建数据读取迭代器
    train_data, val_data = load_nuscenesDataset(cfg)

    train_dataloader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=cfg.batch_size*2, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # 第2步：构建网络，设置训练参数：学习率、学习率衰减策略、优化函数（SDG、Adam、……）、损失函数、……
    model = load_model(cfg)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr_init)

    # 学习率衰减策略
    lf = lambda x: (1 - x / cfg.epochs) * (1.0 - cfg.lr_final) + cfg.lr_final
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    loss_function = torch.nn.MSELoss()

    # 第3步：循环读取数据训练网络
    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)
    f_loss = open(cfg.save_path+'/' + cfg.model_type + '_train_val_loss.csv', 'w')
    f_loss.write('Epoch,' + 'Train Loss,' + 'Val Loss\n')

    loss_min = float('inf')
    epoch_best = 0

    for epoch_i in range(cfg.epochs):
        model.train()

        loss_sum = 0
        loss_sum_print = 0
        l = len(train_dataloader)

        for train_i, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            '''         
            tgt = 0.1*torch.ones(target.shape).to(device) 
            tgt[:,1:,:] = target[:,0:cfg.predict_n-1,:]
            outputs = model(data,tgt)
            '''
            #--------------------------
            tgt = torch.rand(target.size()).to(device)
            tgt[:,0,:] = 0.1*torch.ones(tgt[:,0,:].size()) 
            for t in range(1, cfg.predict_n):   
                output = model(data,tgt.clone())
                tgt[:,t,:] = output[:,t-1,:] 
            outputs = model(data,tgt)
            #---------------------------
            
            loss = loss_function(outputs, target)

            loss_sum = loss_sum + loss.item()
            loss_sum_print = loss_sum_print + loss.item()
            loss.backward()
            optimizer.step()

            if (train_i+1)%cfg.print_count == 0:
                lr = [x['lr'] for x in optimizer.param_groups]
                print('Epoch:', epoch_i,'Step:', train_i, 
                    'Train Loss:', loss_sum_print/cfg.print_count,'Learning Rate:', lr)
                loss_sum_print = 0

        f_loss.write(str(epoch_i) + ', ' + str(loss_sum/l) + ', ')
        loss_sum = 0
        
        scheduler.step()
        
        # 训练完每个epoch进行验证
        model.eval()
        with torch.no_grad():
            loss_sum = 0
            l = len(val_dataloader)

            for val_i, (data, target) in enumerate(val_dataloader):
                data, target = data.to(device), target.to(device)

                #--------------------------
                tgt = torch.rand(target.size()).to(device)
                tgt[:,0,:] = 0.1*torch.ones(tgt[:,0,:].size()) 

                for t in range(1, cfg.predict_n):   
                    output = model(data,tgt)
                    tgt[:,t,:] = output[:,t-1,:]
                
                outputs = model(data,tgt)
                #---------------------------

                '''
                # Create dummy out_H_head                            
                outputs = torch.zeros(out_H.size()).to(device)
                tgt = (torch.rand(out_H.size())*2-1).to(device)
                tgt[:,0,:] = 0.1*torch.ones(tgt[:,0,:].size())                        

                # Compute inference
                for t in range(1, out_H.size()[1]):   
                outputs = net(src=in_B.to(device),tgt=tgt.to(device),var=torch.cat((in_F.to(device), in_T.to(device), in_D.to(device)), dim=1), device=device)                     
                tgt[:,t,:] = outputs[:,t-1,:]     
                outputs = net(in_B.to(device),tgt.to(device),torch.cat((in_F.to(device), in_T.to(device), in_D.to(device)), dim=1), device=device)
                '''

                loss = loss_function(outputs, target)
                loss_sum = loss_sum + loss.item()
            if loss_sum<loss_min:
                loss_min=loss_sum
                epoch_best = epoch_i

            print('Epoch:', epoch_i, 'Val Loss:', loss_sum/l)
            f_loss.write(str(loss_sum/l) + ', Best Epoch:' + str(epoch_best) + ', Loss: '+ str(loss_min/l) +'\n')
            f_loss.flush()
            torch.cuda.empty_cache()

        torch.save(model, cfg.save_path+'/'+str(epoch_i)+'.pt')
        torch.cuda.empty_cache()
    f_loss.close

def train_val_imageWithCoor(cfg):

    if cfg.device == 'cpu':
        device = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device
        device = torch.device('cuda:0')

    # 第1步：构建数据读取迭代器
    train_data, val_data = load_nuscenesDataset(cfg)

    train_dataloader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=cfg.batch_size*2, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # 第2步：构建网络，设置训练参数：学习率、学习率衰减策略、优化函数（SDG、Adam、……）、损失函数、……
    model = load_model(cfg)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr_init)

    # 学习率衰减策略
    lf = lambda x: (1 - x / cfg.epochs) * (1.0 - cfg.lr_final) + cfg.lr_final
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    loss_function = torch.nn.MSELoss()

    # 第3步：循环读取数据训练网络
    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)
    f_loss = open(cfg.save_path+'/' + cfg.model_type + '_train_val_loss.csv', 'w')
    f_loss.write('Epoch,' + 'Train Loss,' + 'Val Loss\n')

    loss_min = float('inf')
    epoch_best = 0

    for epoch_i in range(cfg.epochs):
        model.train()

        loss_sum = 0
        loss_sum_print = 0
        l = len(train_dataloader)

        for train_i, (images, coor, target) in enumerate(train_dataloader):
            images, coor, target = images.to(device), coor.to(device), target.to(device)
            
            optimizer.zero_grad()

            outputs = model(images, coor)

            loss = loss_function(outputs, target)

            loss_sum = loss_sum + loss.item()
            loss_sum_print = loss_sum_print + loss.item()
            loss.backward()
            optimizer.step()

            if (train_i+1)%cfg.print_count == 0:
                lr = [x['lr'] for x in optimizer.param_groups]
                print('Epoch:', epoch_i,'Step:', train_i, 
                    'Train Loss:', loss_sum_print/cfg.print_count,'Learning Rate:', lr)
                loss_sum_print = 0

        f_loss.write(str(epoch_i) + ', ' + str(loss_sum/l) + ', ')
        loss_sum = 0
        
        scheduler.step()
        
        # 训练完每个epoch进行验证
        model.eval()
        with torch.no_grad():
            loss_sum = 0
            l = len(val_dataloader)

            for val_i, (images, coor, target) in enumerate(val_dataloader):
                images, coor, target = images.to(device), coor.to(device), target.to(device)

                outputs = model(images, coor)

                '''
                # Create dummy out_H_head                            
                outputs = torch.zeros(out_H.size()).to(device)
                tgt = (torch.rand(out_H.size())*2-1).to(device)
                tgt[:,0,:] = 0.1*torch.ones(tgt[:,0,:].size())                        

                # Compute inference
                for t in range(1, out_H.size()[1]):   
                outputs = net(src=in_B.to(device),tgt=tgt.to(device),var=torch.cat((in_F.to(device), in_T.to(device), in_D.to(device)), dim=1), device=device)                     
                tgt[:,t,:] = outputs[:,t-1,:]     
                outputs = net(in_B.to(device),tgt.to(device),torch.cat((in_F.to(device), in_T.to(device), in_D.to(device)), dim=1), device=device)
                '''

                loss = loss_function(outputs, target)
                loss_sum = loss_sum + loss.item()
            if loss_sum<loss_min:
                loss_min=loss_sum
                epoch_best = epoch_i

            print('Epoch:', epoch_i, 'Val Loss:', loss_sum/l)
            f_loss.write(str(loss_sum/l) + ', Best Epoch:' + str(epoch_best) + ', Loss: '+ str(loss_min/l) +'\n')
            f_loss.flush()
            torch.cuda.empty_cache()

        torch.save(model, cfg.save_path+'/'+str(epoch_i)+'.pt')
        torch.cuda.empty_cache()
    f_loss.close

# 自定义损失函数
class CustomLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self):
        return

# 加载数据集 
def load_nuscenesDataset(args):
    if args.model_type == 'TransformerCoor':
        train_data = NuscenesDatasetCoor(cfg=args, phase='train')
        val_data = NuscenesDatasetCoor(cfg=args, phase='val')
    if args.model_type == 'TransformerCoor_T':
        train_data = NuScenesDataset_T(cfg=args, phase='train')
        val_data = NuScenesDataset_T(cfg=args, phase='val')
    if args.model_type == 'TransformerImage':
        train_data = NuscenesDatasetImage(cfg=args, phase='train')
        val_data = NuscenesDatasetImage(cfg=args, phase='val')
    if args.model_type == 'Transformer':
        train_data = ResNetDataset(cfg=args, phase='train')
        val_data = ResNetDataset(cfg=args, phase='val')
    return train_data, val_data

def load_model(args):
    if args.model_type == 'Transformer':
        model = ImageResNet50_LSTM(
            out_features = args.predict_n * 2)
    elif args.model_type == 'TransformerImage':
        model = Transformer_image(
            n_src_vocab=500,
            n_trg_vocab=500,
            d_k=32,
            d_v=32,
            d_model=256,
            d_word_vec=256,
            d_inner=1024,
            n_layers=3,
            n_head=4,
            dropout=0.1,
            n_position=1,
            n_position_d=args.predict_n,
            input_dim=128,
            output_dim= 2, #args.predict_n * 2, 
            output_features= 2 #args.predict_n * 2
        )
    elif args.model_type == 'TransformerCoor' or 'TransformerCoor_T':
        model = Transformer(
            n_src_vocab=500,
            n_trg_vocab=500,
            d_k=32,
            d_v=32,
            d_model=256, 
            d_word_vec=256,
            d_inner=1024,
            n_layers=3,
            n_head=4,
            dropout=0.1,
            n_position=args.input_n + 1,
            n_position_d=args.predict_n,
            input_dim= 2, #args.input_n,
            output_dim= 2 #args.predict_n * 2,
        )
    return model

def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='total number of training epochs')
    parser.add_argument('--device', type=str, default='0', help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=3, help='batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')
    parser.add_argument('--save-path', type=str, default='weights/resnet_nuscenes', help='path to save checkpoint')
    parser.add_argument('--lr-init', type=float, default=0.002, help='path to save checkpoint')
    parser.add_argument('--lr-final', type=float, default=0.0001, help='path to save checkpoint')
    parser.add_argument('--res-chain', type=int, default=0, help='1或0, 表示使用或不使用residual chain loss')

    parser.add_argument('--model-type', type=str, default='TransformerCoor_T', help='')
    parser.add_argument('--print-count', type=int, default=100, help='表示相应训练次数后打印')

    # parser.add_argument('--data-root', type=str, default='D:/ECHO/4_2_WS2324_TUM/autonomousDriving/code_project/dataset/v1.0-mini/', help='')
    parser.add_argument('--data-root', type=str, default='D:\ECHO/4_2_WS2324_TUM/autonomousDriving/data/samples/', help='')
    parser.add_argument('--data-path', type=str, default='data/nuscenes/image_scenes/', help='')
    parser.add_argument('--data-path-T', type=str, default='data/nuscenes/', help='')
    parser.add_argument('--image-n', type=int, default=1, help='每一帧环视相机图像有六张图, image_n代表使用其中几张')
    parser.add_argument('--input-n', type=int, default=5, help='是否使用序列图像, 0代表不使用, 其他数值n代表使用n张')
    parser.add_argument('--predict-n', type=int, default=5, help='')
    parser.add_argument('--image-size', type=int, default=[800, 450], help='')
    parser.add_argument('--crop-size', type=int, default=[704, 384], help='')

    return parser.parse_args()

if __name__ == '__main__':
    
    cfg = parse_cfg() 
    if cfg.model_type == 'Transformer':
        train_val_imageWithCoor(cfg)
    else:
        train_val(cfg)

    # python train/Transformer_pipeline_train.py --data-root /opt/data/private/nuscenes/trainval/ --save-path weights/resnet_nuscenes --batch-size 10 --num-workers 8 --epochs 35 --input-n 6
    # python train/Transformer_pipeline_train.py --data-root /opt/data/private/nuscenes/trainval/ --save-path weights/resnet_nuscenes --device 2 --batch-size 8 --num-workers 8 --epochs 50 --model-type 'LSTM'
    # nvidia-smi

    # python train/Transformer_pipeline_train.py --batch-size 16 --epochs 10 --model-type 'TransformerCoor'
    # python train/Transformer_pipeline_train.py --batch-size 1 --epochs 10 --model-type 'TransformerImage' --print-count 50
    # python train/Transformer_pipeline_train.py --batch-size 12 --epochs 10 --model-type 'ResNet50'
    # model type: 'ResNet50' 'LSTM' 'Transformer'

    # compare results -> lstm(with, without Position), 
    #                   Resnet50(with, without Position), Transformer
    # goal: loss↓ -> <1.5
    # len_total_data = 21831
