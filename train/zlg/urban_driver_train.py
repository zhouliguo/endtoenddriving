import sys 
sys.path.append('.')

import os
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_preparation.urban_driver_data import NuplanDataset

from networks.urban_driver import UrbanDriverModel

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
        device = torch.device('gpu:0')

    # 第1步：构建数据读取迭代器

    train_data = NuplanDataset(data_path = os.path.join(cfg.data_path, 'train'), phase='train')
    val_data = NuplanDataset(data_path = os.path.join(cfg.data_path, 'val'), phase='val')

    train_dataloader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=cfg.batch_size*2, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # 第2步：构建网络，设置训练参数：学习率、学习率衰减策略、优化函数（SDG、Adam、……）、损失函数、……

    model = UrbanDriverModel().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)

    loss_function = torch.nn.MSELoss()

    # 第3步：循环读取数据训练网络

    for epoch_i in range(cfg.epochs):
        model.train()

        for train_i, (input, target) in enumerate(train_dataloader):
            input, target = input.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(input)
            loss = loss_function(output, target)

            print(loss.item())

            loss.backward()
            optimizer.step()

        # 训练完每个epoch进行验证
        model.eval()
        with torch.no_grad():
            for val_i, (input, target) in enumerate(val_dataloader):
                input, target = input.to(device), target.to(device)
                output = model(input)
                loss = loss_function(output, target)

        torch.save(model, cfg.save_path+'/'+str(epoch_i)+'.pt')
        torch.cuda.empty_cache()
\
def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='total number of training epochs')
    parser.add_argument('--data-path', type=str, default='data/nuplan/mini', help='data path')
    parser.add_argument('--device', type=str, default='cpu', help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--learning-rate', type=int, default=0.01, help='initial learning rate')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')
    parser.add_argument('--save-path', type=str, default='weights/urban_driver', help='path to save checkpoint')

    return parser.parse_args()

if __name__ == '__main__':
    cfg = parse_cfg()
    
    train_val(cfg)
