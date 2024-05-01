import sys 
sys.path.append('.')

import os
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_preparation.resnet_dataset_image_coor import ResNetDataset

from networks.simple_renet50 import SimpleResNet50, ResNet50WithPos, ImagePositionEnhancementResMet50
import plotly.express as px

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

    train_data = ResNetDataset(cfg=cfg, phase='train')
    val_data = ResNetDataset(cfg=cfg, phase='val')

    train_dataloader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=cfg.batch_size*2, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # 第2步：构建网络，设置训练参数：学习率、学习率衰减策略、优化函数（SDG、Adam、……）、损失函数、……

    model = ImagePositionEnhancementResMet50(
        {"num_coor": cfg.input_n*2,
        "out_features": 12,
        "input_past_channels": cfg.input_n*3}).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr_init)

    # 学习率衰减策略
    lf = lambda x: (1 - x / cfg.epochs) * (1.0 - cfg.lr_final) + cfg.lr_final
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    loss_function = torch.nn.MSELoss()

    # 第3步：循环读取数据训练网络

    for epoch_i in range(cfg.epochs):
        model.train()

        loss_sum = 0
        for train_i, (data) in enumerate(train_dataloader):
            image, coor, target = data
            image, coor, target = image.to(device), coor.to(device), target.to(device)
            # print(train_i, image.shape)
            optimizer.zero_grad()

            output = model(image,coor)
            loss = loss_function(output, target)
            loss_sum = loss_sum + loss.item()
            loss.backward()
            optimizer.step()
        
            if (train_i+1)%100 == 0:
                lr = [x['lr'] for x in optimizer.param_groups]
                train_losses.append(loss_sum / 100)
                print_colored_text('Epoch:', epoch_i,'Step:', train_i, 
                           'Train Loss:', loss_sum/100,'Learning Rate:', lr, "BLUE")
                loss_sum = 0

        loss_sum = 0
        '''
        lr = [x['lr'] for x in optimizer.param_groups]
        train_losses.append(loss_sum / len(train_dataloader))
        print_colored_text('Epoch:', epoch_i,'Step:', train_i, 
                           'Train Loss:', loss_sum/len(train_dataloader),'Learning Rate:', lr, "BLUE")
        loss_sum = 0
        '''
        scheduler.step()
        
        # 训练完每个epoch进行验证
        model.eval()
        with torch.no_grad():
            loss_sum = 0
            l = len(val_dataloader)
            for val_i, (image, coor, target) in enumerate(val_dataloader):
                image, coor, target = image.to(device), coor.to(device), target.to(device)
                output = model(image, coor)
                loss = loss_function(output, target)
                loss_sum = loss_sum + loss.item()
            val_losses.append(loss_sum/l)
            print_colored_text('Epoch:', epoch_i, 'Val Loss:', loss_sum/l, "GREEN")

        torch.save(model, cfg.save_path+'/'+str(epoch_i)+'.pt')
        torch.cuda.empty_cache()


def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='total number of training epochs')
    parser.add_argument('--device', type=str, default='0', help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')
    parser.add_argument('--save-path', type=str, default='weights/resnet_nuscenes', help='path to save checkpoint')
    parser.add_argument('--lr-init', type=float, default=0.002, help='path to save checkpoint')
    parser.add_argument('--lr-final', type=float, default=0.0001, help='path to save checkpoint')

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

    '''
    # 绘制训练和验证损失曲线
    plt.plot(range(1, cfg.epochs + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, cfg.epochs + 1), val_losses, label='Validation Loss', marker='o')
    plt.show()  # 显示图形
    '''

    '''
    # 生成示例的数据框
    import pandas as pd
    df = pd.DataFrame({
        'Epoch': list(range(1, cfg.epochs + 1)),
        'Training Loss': train_losses,
        'Validation Loss': val_losses
    })

    # 使用 Plotly Express 绘制折线图
    fig = px.line(df, x='Epoch', y=['Training Loss', 'Validation Loss'], 
                labels={'value': 'Loss', 'Epoch': 'Epochs'},
                title='Training and Validation Loss Curve')

    # 显示图形
    fig.show()
    '''
