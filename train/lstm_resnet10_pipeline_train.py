import sys 
sys.path.append('.')

import os
import argparse

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_preparation.dataset_transformer_seq import ResNetDataset

from networks.lstm_rnn import ImagePositionEnhancementResMet50, ImageResNet50_LSTM, Transformer_image


train_losses = []
val_losses = []

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

    train_data = ResNetDataset(cfg=cfg, phase=cfg.data_status + 'train')
    val_data = ResNetDataset(cfg=cfg, phase=cfg.data_status + 'val')

    train_dataloader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=cfg.batch_size*2, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # 第2步：构建网络，设置训练参数：学习率、学习率衰减策略、优化函数（SDG、Adam、……）、损失函数、……

    model = load_model(cfg)

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
        train_loss_epoch = []
        for train_i, (data) in enumerate(train_dataloader):
            image, coor, target = data
            image = [tensor.to(device) for tensor in image]
            coor, target = coor.to(device), target.to(device)
            l = len(train_dataloader)
            # print(train_i, image.shape)
            optimizer.zero_grad()
            if cfg.model_type == 'Transformer':
                coor = coor.unsqueeze(1)
                target = target.unsqueeze(1)
            output = model(image,coor)
            loss = loss_function(output, target)
            loss_sum = loss_sum + loss.item()
            loss_sum_print = loss_sum_print + loss.item()
            loss.backward()
            optimizer.step()

            if cfg.data_status == "":
                printCount = 200
                if (train_i+1)%printCount == 0:
                    lr = [x['lr'] for x in optimizer.param_groups]
                    print('Epoch:', epoch_i,'Step:', train_i, 
                            'Train Loss:', loss_sum_print/printCount,'Learning Rate:', lr)
                    # f_loss.write(str(epoch_i) + ', ' + str(loss_sum/l) + ', ')
                    # f_loss.write(str(loss_sum/l) + ', Best Epoch:' + str(epoch_best) + ', Loss: '+ str(loss_min/l) +'\n')
                    loss_sum_print = 0

        if cfg.data_status == "":
            # f_loss.write(str(epoch_i) + ',' + str(loss_sum/l) )
            f_loss.write(str(epoch_i) + ', ' + str(loss_sum/l) + ', ')
            loss_sum = 0
        else:
            lr = [x['lr'] for x in optimizer.param_groups]
            train_losses.append(loss_sum / len(train_dataloader))
            print_colored_text('Epoch:', epoch_i,'Step:', train_i, 
                            'Train Loss:', loss_sum/len(train_dataloader),'Learning Rate:', lr, "BLUE")
            loss_sum = 0
        
        # np.save('train_loss_LSTM.npy', train_losses)
        scheduler.step()
        
        # 训练完每个epoch进行验证
        model.eval()
        with torch.no_grad():
            loss_sum = 0
            l = len(val_dataloader)
            for val_i, (image, coor, target) in enumerate(val_dataloader):
                image = [tensor.to(device) for tensor in image]
                coor, target = coor.to(device), target.to(device)
                if cfg.model_type == 'Transformer':
                    coor = coor.unsqueeze(1)
                    target = target.unsqueeze(1)
                output = model(image,coor)
                loss = loss_function(output, target)
                loss_sum = loss_sum + loss.item()
            if loss_sum<loss_min:
                loss_min=loss_sum
                epoch_best = epoch_i
            val_losses.append(loss_sum/l)
            print_colored_text('Epoch:', epoch_i, 'Val Loss:', loss_sum/l, "GREEN")
            # f_loss.write(str(loss_sum/l), 'Best Epoch:', epoch_best, 'Loss', str(loss_min/l) +'\n')
            f_loss.write(str(loss_sum/l) + ', Best Epoch:' + str(epoch_best) + ', Loss: '+ str(loss_min/l) +'\n')
            f_loss.flush()
            torch.cuda.empty_cache()

        # np.save('val_loss_LSTM.npy', val_losses)
        torch.save(model, cfg.save_path+'/'+str(epoch_i)+'.pt')
        torch.cuda.empty_cache()
    f_loss.close

def load_model(args):
    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        device = torch.device('cuda:0')

    if args.model_type == 'LSTM':
        model = ImageResNet50_LSTM(
            out_features = args.predict_n * 2).to(device)
    elif args.model_type == 'Transformer':
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
            output_dim=args.input_n + 1, 
            output_features=args.predict_n * 2
        ).to(device)
    elif args.model_type == 'ResNet50':
        model = ImagePositionEnhancementResMet50(
            input_coor_features=args.input_n * 2, 
            input_channels=(args.input_n + 1) * 3,
            out_features=args.predict_n * 2
        ).to(device)
    return model

def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='total number of training epochs')
    parser.add_argument('--device', type=str, default='0', help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')
    parser.add_argument('--save-path', type=str, default='weights/resnet_nuscenes', help='path to save checkpoint')
    parser.add_argument('--lr-init', type=float, default=0.002, help='path to save checkpoint')
    parser.add_argument('--lr-final', type=float, default=0.0001, help='path to save checkpoint')

    parser.add_argument('--model-type', type=str, default='Transformer', help='')
    parser.add_argument('--data-status', type=str, default='', help='')
    # parser.add_argument('--data-root', type=str, default='D:/ECHO/4_2_WS2324_TUM/autonomousDriving/code_project/dataset/v1.0-mini/', help='')
    parser.add_argument('--data-root', type=str, default='D:\ECHO/4_2_WS2324_TUM/autonomousDriving/data/samples/', help='')
    parser.add_argument('--data-path', type=str, default='data/nuscenes/image_scenes/', help='')
    parser.add_argument('--image-n', type=int, default=1, help='每一帧环视相机图像有六张图, image_n代表使用其中几张')
    parser.add_argument('--input-n', type=int, default=3, help='是否使用序列图像, 0代表不使用, 其他数值n代表使用n张')
    parser.add_argument('--predict-n', type=int, default=6, help='')
    parser.add_argument('--image-size', type=int, default=[800, 450], help='')
    parser.add_argument('--crop-size', type=int, default=[704, 384], help='')

    return parser.parse_args()

from colorama import Fore, Style

def print_colored_text(*args):
    args = [str(arg) for arg in args]
    colored_text = " ".join(args[:-1])
    color_code = args[-1]

    colored_text = f"{Fore.__dict__[color_code]}{colored_text}{Style.RESET_ALL}"
    print(colored_text)

if __name__ == '__main__':
    
    cfg = parse_cfg() 
    train_val(cfg)

    # python train/lstm_resnet10_pipeline_train.py --data-root /opt/data/private/nuscenes/trainval/ --save-path weights/resnet_nuscenes --device 2 --batch-size 8 --num-workers 8 --epochs 50
    # python train/lstm_resnet10_pipeline_train.py --data-root /opt/data/private/nuscenes/trainval/ --save-path weights/resnet_nuscenes --device 2 --batch-size 8 --num-workers 8 --epochs 1 --model-type 'LSTM'
    # nvidia-smi

    # python train/lstm_resnet10_pipeline_train.py --batch-size 2 --epochs 10 --model-type 'Transformer'
    # python train/lstm_resnet10_pipeline_train.py --batch-size 6 --epochs 10 --model-type 'LSTM'
    # python train/lstm_resnet10_pipeline_train.py --batch-size 12 --epochs 10 --model-type 'ResNet50'
    # model type: 'ResNet50' 'LSTM' 'Transformer'

    # compare results -> lstm(with, without Position), 
    #                   Resnet50(with, without Position), Transformer
    # goal: loss↓ -> <1.5
    # len_total_data = 21831
