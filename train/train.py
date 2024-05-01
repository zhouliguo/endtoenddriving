import sys
import os
import argparse
import torch
from networks.cnn import MyNetwork, MyNetwork_NOimage, Net_coo, Net_img
from torch.utils.data import DataLoader, random_split
from dataset.dataloader03 import DT_Img, DT_Noimg
import utils
import cv2
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import csv
from torch.optim.lr_scheduler import StepLR


def train_val(cfg, load_model, device0):
    device = device0

    # 第1步：构建数据读取迭代器

    train_set = DT_Img(cfg, phase='train')
    val_set = DT_Img(cfg, phase='val')

    # 创建数据加载器
    train_dataloader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                                pin_memory=True)
    
    print('len of train', len(train_dataloader))
    print('len of val', len(val_dataloader))

    # 第2步：构建网络，设置训练参数：学习率、学习率衰减策略、优化函数（SDG、Adam、……）、损失函数、……

    model = load_model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.initial_lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.initial_lr)

    loss_function = torch.nn.MSELoss()
    # loss_function = torch.nn.L1Loss()

    lf = lambda x: (1 - x / cfg.epochs) * (1.0 - cfg.final_lr) + cfg.final_lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 第3步：循环读取数据训练网络

    val_losses = ['validation loss']
    train_losses = ['train loss']
    lr = ['learning rate']
    ep = ['epoch']
    save_info = [
        ['min epoch of train loss', 0],
        ['min train loss', 100],
        ['learning rate', 0],
        ['min epoch of val loss', 0],
        ['min val loss', 100],
        ['learning rate', 0]
    ]

    for epoch_i in range(cfg.epochs):
        model.train()
        sum_train = 0.0

        for (image, coor_past, coor_future) in tqdm(train_dataloader, desc=f'train Epoch {epoch_i} / {cfg.epochs}' ):
            time.sleep(0.00001)
            # print('shape of image', image.size())
            image = image.to(device)
            coor_future = coor_future.to(device)
            coor_past = coor_past[:,:-2].to(device)  # coor_past[:,0:-2]是后一个点相对于前一个点的增量, coor_past[:,-2:]是当前点的绝对坐标，
            delta_sum_xy = torch.zeros((coor_past.size(0), 2)).to(device)

            optimizer.zero_grad()
            output = model([image, coor_past])

            
            for i in range(1, cfg.predict_n):
                delta_sum_xy[:, 0] = delta_sum_xy[:, 0] + output[:, 2 * (i - 1)]
                delta_sum_xy[:, 1] = delta_sum_xy[:, 1] + output[:, 2 * (i - 1) + 1]
                coor_future[:, 2 * i] = coor_future[:, 2 * i] - delta_sum_xy[:, 0]
                coor_future[:, 2 * i + 1] = coor_future[:, 2 * i + 1] - delta_sum_xy[:, 1]
            
            
            loss = loss_function(output, coor_future)
            sum_train += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

        train_loss = sum_train / (len(train_dataloader))
        scheduler.step()

        print(f'The train loss of epoch {epoch_i} is :', train_loss)
        train_losses.append(train_loss)
        lr.append(optimizer.param_groups[0]['lr'])
        ep.append(epoch_i)

        if train_loss < save_info[1][1]:
            save_info[1][1] = train_loss
            save_info[0][1] = epoch_i
            save_info[2][1] = optimizer.param_groups[0]['lr']

        # 训练完每个epoch进行验证
        model.eval()
        with torch.no_grad():
            loss_sum = 0.0
            for (image, coor_past, coor_future) in tqdm(val_dataloader, desc=f'val Epoch {epoch_i} / {cfg.epochs}'):
                image = image.to(device)
                coor_future = coor_future.to(device)
                coor_past = coor_past[:,:-2].to(device)  # coor_past[:,0:-2]是后一个点相对于前一个点的增量, coor_past[:,-2:]是当前点的绝对坐标，

                output = model([image, coor_past])
                
                for i in range(1, cfg.predict_n):
                    output[:, 2 * i] = output[:, 2 * i] + output[:, 2 * (i - 1)]
                    output[:, 2 * i + 1] = output[:, 2 * i + 1] + output[:, 2 * (i - 1) + 1]
                
                loss_sum += loss_function(output, coor_future).item()

            val_i = len(val_dataloader)
            val_loss = loss_sum / val_i
            if val_loss < save_info[4][1]:
                save_info[4][1] = val_loss
                save_info[3][1] = epoch_i
                save_info[5][1] = optimizer.param_groups[0]['lr']

        val_losses.append(val_loss)

        print(f'The val loss of epoch {epoch_i} is :', val_loss)

        if epoch_i > 10:
            save_path = cfg.save_path + '/' + str(epoch_i) + '.pt'
            torch.save(model, save_path)
        torch.cuda.empty_cache()
    
    csv_file = cfg.save_path + '/' + 'result report of train'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(save_info)
    
    csv_file1 = cfg.save_path + '/' + 'loss'
    all_loss = [train_losses, val_losses, lr, ep]
    all_loss = list(map(list, zip(*all_loss)))
    with open(csv_file1, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(all_loss)

    plt.figure()
    plt.plot(np.arange(len(val_losses) - 1), np.array(train_losses[1:]).astype(float), label='train loss')
    plt.plot(np.arange(len(val_losses) - 1), np.array(val_losses[1:]).astype(float), label='validation loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(cfg.Image_name)
    plt.savefig(cfg.save_path + '/' + 'train_val_loss.png')


def train_val_Noimage(cfg, load_model, device0):
    device = device0

    # 第1步：构建数据读取迭代器

    train_set = DT_Noimg(cfg, phase='train')
    val_set = DT_Noimg(cfg, phase='val')

    # 创建数据加载器
    train_dataloader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                                pin_memory=True)
    
    print('len of train', len(train_dataloader))
    print('len of val', len(val_dataloader))

    # 第2步：构建网络，设置训练参数：学习率、学习率衰减策略、优化函数（SDG、Adam、……）、损失函数、……

    model = load_model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.initial_lr)

    loss_function = torch.nn.MSELoss()
    # loss_function = torch.nn.L1Loss()

    lf = lambda x: (1 - x / cfg.epochs) * (1.0 - cfg.final_lr) + cfg.final_lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 第3步：循环读取数据训练网络

    val_losses = ['validation loss']
    train_losses = ['train loss']
    lr = ['learning rate']
    ep = ['epochs']
    save_info = [
        ['min epoch of train loss', 0],
        ['min train loss', 100],
        ['learning rate', 0],
        ['min epoch of val loss', 0],
        ['min val loss', 100],
        ['learning rate', 0]
    ]

    for epoch_i in range(cfg.epochs):
        model.train()
        sum_train = 0.0

        for (image, coor_past, coor_future) in tqdm(train_dataloader, desc=f'train Epoch {epoch_i} / {cfg.epochs}' ):
            time.sleep(0.00001)
            coor_future = coor_future.to(device)
            coor_past = coor_past[:,:-2].to(device)  # coor_past[:,0:-2]是后一个点相对于前一个点的增量, coor_past[:,-2:]是当前点的绝对坐标，
            delta_sum_xy = torch.zeros((coor_past.size(0), 2)).to(device)
            image = image.to(device)

            optimizer.zero_grad()
            output = model([image, coor_past])
            # print('size of output', output.size())
            '''
            for i in range(1, cfg.predict_n):
                delta_sum_xy[:, 0] = delta_sum_xy[:, 0] + output[:, 2 * (i - 1)]
                delta_sum_xy[:, 1] = delta_sum_xy[:, 1] + output[:, 2 * (i - 1) + 1]
                coor_future[:, 2 * i] = coor_future[:, 2 * i] - delta_sum_xy[:, 0]
                coor_future[:, 2 * i + 1] = coor_future[:, 2 * i + 1] - delta_sum_xy[:, 1]
            '''
            
            loss = loss_function(output, coor_future)
            sum_train += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

        train_loss = sum_train / (len(train_dataloader))
        scheduler.step()

        print(f'The train loss of epoch {epoch_i} is :', train_loss)
        train_losses.append(train_loss)
        lr.append(optimizer.param_groups[0]['lr'])
        ep.append(epoch_i)

        if train_loss < save_info[1][1]:
            save_info[1][1] = train_loss
            save_info[0][1] = epoch_i
            save_info[2][1] = optimizer.param_groups[0]['lr']

        # 训练完每个epoch进行验证
        model.eval()
        with torch.no_grad():
            loss_sum = 0.0
            for (image, coor_past, coor_future) in tqdm(val_dataloader, desc=f'val Epoch {epoch_i} / {cfg.epochs}'):
                coor_future = coor_future.to(device)
                coor_past = coor_past[:,:-2].to(device)  # coor_past[:,0:-2]是后一个点相对于前一个点的增量, coor_past[:,-2:]是当前点的绝对坐标，
                image = image.to(device)


                output = model([image, coor_past])
                '''
                for i in range(1, cfg.predict_n):
                    output[:, 2 * i] = output[:, 2 * i] + output[:, 2 * (i - 1)]
                    output[:, 2 * i + 1] = output[:, 2 * i + 1] + output[:, 2 * (i - 1) + 1]
                '''
                loss_sum += loss_function(output, coor_future).item()

            val_i = len(val_dataloader)
            val_loss = loss_sum / val_i
            if val_loss < save_info[4][1]:
                save_info[4][1] = val_loss
                save_info[3][1] = epoch_i
                save_info[5][1] = optimizer.param_groups[0]['lr']

        val_losses.append(val_loss)

        print(f'The val loss of epoch {epoch_i} is :', val_loss)

        save_path = cfg.save_path + '/' + str(epoch_i) + '.pt'
        torch.save(model, save_path)
        torch.cuda.empty_cache()
    
    csv_file = cfg.save_path + '/' + 'result report of train'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(save_info)
    
    csv_file1 = cfg.save_path + '/' + 'loss'
    all_loss = [train_losses, val_losses, ep, lr]
    all_loss = list(map(list, zip(*all_loss)))
    with open(csv_file1, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(all_loss)

    plt.figure()
    plt.plot(np.arange(len(val_losses) - 1), np.array(train_losses[1:]).astype(float), label='train loss')
    plt.plot(np.arange(len(val_losses) - 1), np.array(val_losses[1:]).astype(float), label='validation loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(cfg.Image_name)
    plt.savefig(cfg.save_path + '/' + 'train_val_loss.png')

def test(cfg, load_model, device0):
    device = device0

    # 第1步：构建数据读取迭代器

    val_set = DT_Img(cfg, phase='val')

    # 创建数据加载器

    val_dataloader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                                pin_memory=True)
    
    print('len of val', len(val_dataloader))

    # 第2步：构建网络，设置训练参数：学习率、学习率衰减策略、优化函数（SDG、Adam、……）、损失函数、……

    model = load_model.to(device)

    loss_function = torch.nn.MSELoss()
    # loss_function = torch.nn.L1Loss()

    # 第3步：循环读取数据训练网络
        
        # 训练完每个epoch进行验证
    model.eval()
    with torch.no_grad():
        loss_sum = 0.0

        for (image, coor_past, coor_future) in val_dataloader:
            image = image.to(device)
            coor_future = coor_future.to(device)
            coor_past = coor_past[:,:-2].to(device)  # coor_past[:,0:-2]是后一个点相对于前一个点的增量, coor_past[:,-2:]是当前点的绝对坐标，

            output = model([image, coor_past])

            for i in range(1, cfg.predict_n):
                output[:, 2 * i] = output[:, 2 * i] + output[:, 2 * (i - 1)]
                output[:, 2 * i + 1] = output[:, 2 * i + 1] + output[:, 2 * (i - 1) + 1]

            loss_sum += loss_function(output, coor_future).item()

        val_i = len(val_dataloader)
        val_loss = loss_sum / val_i
        

    print(f'The val loss is :', val_loss)
    torch.cuda.empty_cache()
    


def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500, help='total number of training epochs')
    parser.add_argument('--Image-name', type=str, default='coo, dx, image func', help='titel of saved image')
    
    parser.add_argument('--device', type=str, default='1', help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=36, help='batch size')
    parser.add_argument('--initial-lr', type=int, default=0.1, help='initial learning rate')
    parser.add_argument('--final-lr', type=int, default=0.0001, help='final learning rate')
    parser.add_argument('--num-workers', type=int, default=2, help='number of workers')
    parser.add_argument('--save-path', type=str, default='/data/tumdriving/hao/weights/one_camera/coo_b36_original_onelstm', help='path to save weights')

    parser.add_argument('--image-root', type=str, default='/data/tumdriving/hao/gr/dataset/Nuscenes/nuscenes/trainval/', help='图像路径的根路径')
    parser.add_argument('--data-path', type=str, default='/data/tumdriving/hao/gr/dataset/Nuscenes/image_t_r/', help='用于保存图像路径和标签的csv文件的路径')
    parser.add_argument('--image-n', type=int, default=1, help='每一帧环视相机图像有六张图, image_n代表使用其中几张')
    parser.add_argument('--input-n', type=int, default=4, help='输入序列帧数')
    parser.add_argument('--predict-n', type=int, default=6, help='')
    parser.add_argument('--image-size', type=int, default=[270, 480], help='[h, w]')
    parser.add_argument('--crop-size', type=int, default=[224, 480], help='[h, w]')
    parser.add_argument('--IMAGE-FINAL_DIM', type=int, default=(224, 480), help='')
    parser.add_argument('--IMAGE-RESIZE_SCALE', type=int, default=0.3, help='')
    parser.add_argument('--IMAGE_TOP_CROP', type=int, default=46, help='')
    parser.add_argument('--IMAGE-ORIGINAL-HEIGHT', type=int, default=900, help='')
    parser.add_argument('--IMAGE-ORIGINAL-WIDTH', type=int, default=1600, help='[h, w]')

    return parser.parse_args()


if __name__ == '__main__':
    cfg = parse_cfg()
    device = torch.device('cuda:1')
    if os.path.exists(cfg.save_path):
        pass
    else:
        os.mkdir(cfg.save_path)
    # load_model = torch.load('/data/tumdriving/hao/weights/one_camera/img_lr0.01_b36_002/123.pt')
    # load_model = MyNetwork(input_n=cfg.input_n, out_feature=12)
    load_model = MyNetwork_NOimage(input_n=cfg.input_n, out_feature=12)
    # load_model = Net_coo(input_n=cfg.input_n, out_feature=12)
    # load_model = Net_img(input_n=cfg.input_n, out_feature=12)

    # train_val(cfg, load_model, device)
    train_val_Noimage(cfg, load_model, device)
    # test(cfg, load_model, device)

