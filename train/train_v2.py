import sys
sys.path.append('.')
import argparse
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
import torch
from data_preparation.nuscenes_data_v1 import ResNetDataset, SeqResNetDataset
from data_preparation.dataloader_v2 import DT_Img
from networks.resnet50_v1 import ResnetBaseline, SeqResnet, ResnetLSTM
import torchvision.transforms as transforms
import os
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# Evaluate the model
def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(torch.sum(loss, dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(torch.sum(loss, dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(
    pred_pos, pred_pos_gt, consider_ped=None, mode='sum'
):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(torch.sum(loss, dim=1)) * consider_ped
    else:
        loss = torch.sqrt(torch.sum(loss, dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',
                        type=int, 
                        default=10,
                        help='specify the random seed')
    parser.add_argument('--lr',
                        type=float, 
                        default=5e-4,
                        help='specify the learning rate')
    parser.add_argument('--max_epochs',
                        type=int, 
                        default=50,
                        help='specify the total epochs for training')
    parser.add_argument('--batch_size',
                        type=int, 
                        default=32,
                        help='specify the batch size')
    parser.add_argument('--device',
                        type=str, 
                        default='cpu',
                        help='e.g. cpu or 0 or 0,1,2,3')    
    parser.add_argument('--data_root',
                        type=str, 
                        default='/mnt/ssd/nuscenes/trainval/',
                        help='e.g. cpu or 0 or 0,1,2,3') 
    parser.add_argument('--csv_root',
                        type=str, 
                        default='/mnt/ssd/xzt/unified-driving/data/nuscenes/image_t_r/',
                        help='path to save all csv files') 
    parser.add_argument('--mini',
                        type=bool, 
                        default=False,
                        help='use mini dataset or not') 
    parser.add_argument('--resume',
                        type=str, 
                        help='resume training from a checkpoint') 
    parser.add_argument('--model',
                        type=str, 
                        default='SeqResnet',
                        help='specify the model to train') 
    parser.add_argument('--coord_mode',
                        type=str, 
                        default='relative',
                        help='specify the coordinate mode: relative or delta')
    
    # added
    parser.add_argument('--image-n', type=int, default=1, help='每一帧环视相机图像有六张图, image_n代表使用其中几张')
    parser.add_argument('--input-n', type=int, default=4, help='输入序列帧数')
    parser.add_argument('--predict-n', type=int, default=6, help='')
    parser.add_argument('--image-size', type=int, default=[450, 800], help='[h, w]')
    parser.add_argument('--crop-size', type=int, default=[384, 704], help='[h, w]')
    parser.add_argument('--IMAGE-FINAL_DIM', type=int, default=(224, 480), help='')
    parser.add_argument('--IMAGE-RESIZE_SCALE', type=int, default=0.3, help='')
    parser.add_argument('--IMAGE_TOP_CROP', type=int, default=46, help='')
    parser.add_argument('--IMAGE-ORIGINAL-HEIGHT', type=int, default=900, help='')
    parser.add_argument('--IMAGE-ORIGINAL-WIDTH', type=int, default=1600, help='[h, w]')

    args = parser.parse_args()
    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        device = torch.device(f'cuda')

    seed = args.seed
    learning_rate = args.lr
    max_epochs = args.max_epochs
    batch_size = args.batch_size
    seed_everything(seed)
    is_mini = args.mini
    coord_mode = args.coord_mode
    input_n = args.input_n
    input_n = input_n - 1 if coord_mode == 'delta' else input_n
    predict_n = args.predict_n

    """
    Define coordinate mode:
    1. relative: the coordinates are relative to the current coordinate (default)
    2. delta: the coordinates are the difference between the current coordinate and the previous coordinate, e.g. [(x1-x0,y1-y0), (x2-x1,y2-y1)]
    """
    writer = SummaryWriter(log_dir="logs") # for logging

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
                    transforms.Resize((288,512)),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
    test_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])

    print("Loading dataset...")
    if not is_mini:
        train_set = DT_Img(args, phase='train')
        val_set = DT_Img(args, phase='val')
    else:
        train_set = DT_Img(args, phase='mini_train')
        val_set = DT_Img(args, phase='mini_val')

    train_dataloader = DataLoader(train_set, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_set, batch_size=1, drop_last=True, shuffle=False, num_workers=8)

    # TODO: add args (coord_mode) to specify the model
    if args.model == 'ResnetBaseline':
        model = ResnetBaseline().to(device)
    elif args.model == 'SeqResnet':
        model = SeqResnet(input_n, predict_n).to(device)
    elif args.model == 'ResnetLSTM':
        model = ResnetLSTM(input_n, predict_n).to(device)
    
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    loss_list = []
    loss_fn = nn.L1Loss()
    start_epoch = -1

    if args.resume:
        model.load_state_dict(torch.load(args.resume)["model"])
        optimizer.load_state_dict(torch.load(args.resume)["optimizer"])
        start_epoch = torch.load(args.resume)["epoch"]


    for epoch in range(max_epochs):
        if epoch <= start_epoch:
            continue
        loop = tqdm(train_dataloader, total =len(train_dataloader))
        loss_sum = 0
        for batch in loop: #loop
            img, coor_past, target, _ = batch   
            img, coor_past, target = img.to(device), coor_past.to(device), target.to(device)
            pred = model(img, coor_past)

            loss = loss_fn(pred,target.view(img.shape[0],-1).to(torch.float32))
            loss_sum += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loop.set_description(f'Epoch [{epoch}/{max_epochs}]')
            loop.set_postfix(loss = loss.item())
        mean_loss = round(loss_sum/len(train_dataloader), 2)
        print(f"mean L1 loss for epoch {epoch}: {mean_loss}")
        writer.add_scalar("loss", mean_loss, epoch)
        loss_list.append(mean_loss)
        # save checkpoint
        ckpt = dict(model = model.state_dict(),
                    optimizer = optimizer.state_dict(),
                    epoch = epoch)
        if not os.path.exists("models"):
            os.makedirs("models")
        torch.save(ckpt, os.path.join("models", f"{args.model}_last_delta.ckpt"))

    writer.close()
    torch.save(model, os.path.join("models", f"{args.model}_final_delta.pth"))
    # Evaluate the model
    ade_error, fde_error = 0.0, 0.0
    
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            img, coor_past, target, imagepath_cur = batch   
            # print(imagepath_cur[0])

           
            img, coor_past, target = img.to(device), coor_past.to(device), target.to(device)
            pred = model(img, coor_past)
            ade = displacement_error(pred.view(-1,1,2), target.view(-1,1,2))
            ade_error += ade
            fde = final_displacement_error(pred.view(-1,1,2), target.view(-1,1,2))
            fde_error += fde
        
    print(f"Average Displacement Error: {ade_error/len(val_dataloader)}")
    print(f"Final Displacement Error: {fde_error/len(val_dataloader)}")



