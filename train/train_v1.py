import sys
sys.path.append('.')
import argparse
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
import torch
from data_preparation.nuscenes_data_v1 import ResNetDataset, SeqResNetDataset
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
                        default=1e-2,
                        help='specify the learning rate')
    parser.add_argument('--max_epochs',
                        type=int, 
                        default=100,
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
                        default='/opt/data/private/nuscenes/trainval',
                        help='e.g. cpu or 0 or 0,1,2,3') 
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
    train_set = SeqResNetDataset(csv_root="data/nuscenes/image_scenes/", # ResNetDataset
                        data_root = args.data_root,
                        phase="train" if not is_mini else "mini_train",
                        transform=train_transforms)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=8)

    val_set = SeqResNetDataset(csv_root="data/nuscenes/image_scenes/", # ResNetDataset
                        data_root = args.data_root,
                        phase="val" if not is_mini else "mini_val",
                        transform=test_transforms)
    val_dataloader = DataLoader(val_set, batch_size=1, drop_last=True, shuffle=False, num_workers=8)
    if args.model == 'ResnetBaseline':
        model = ResnetBaseline().to(device)
    elif args.model == 'SeqResnet':
        model = SeqResnet().to(device)
    elif args.model == 'ResnetLSTM':
        model = ResnetLSTM().to(device)
    
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
            """
            img, x, y, target, yaw = batch
            img, x, y, target, yaw = img.to(device), x.to(device), y.to(device), target.to(device), yaw.to(device)

            # prediction = current position + offset
            cur_pos = torch.cat((x.unsqueeze(1), y.unsqueeze(1)), dim=1)
            cur_pos = cur_pos.repeat(1,6)
            out = model(img, yaw).to(torch.float32)
            pred = cur_pos + out
            """
            img, coords, target = batch["imgs"], batch["coordinates"], batch["target"]    
            img, coords, target = img.to(device), coords.to(device), target.to(device)
            pred = model(img, coords)
            
            #print(pred.shape, target.shape)
            loss = loss_fn(pred,target.view(img.shape[0],-1).to(torch.float32))
            loss_sum += loss.item()
            #print(pred, target.view(x.shape[0],-1))
            #print("===============================")
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
        torch.save(ckpt, os.path.join("models", f"{args.model}_last_lr-2.ckpt"))

    writer.close()
    torch.save(model, os.path.join("models", f"{args.model}_final_lr-2.pth"))
    # Evaluate the model
    ade_error, fde_error = 0.0, 0.0
    
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            """
            img, x, y, target, yaw = batch
            img, x, y, target, yaw = img.to(device), x.to(device), y.to(device), target.to(device), yaw.to(device)

            cur_pos = torch.cat((x.unsqueeze(1), y.unsqueeze(1)), dim=1)
            cur_pos = cur_pos.repeat(1,6)
            out = model(img, yaw).to(torch.float32)
            pred = cur_pos + out
            """
            img, coords, target = batch["imgs"], batch["coordinates"], batch["target"]    
            img, coords, target = img.to(device), coords.to(device), target.to(device)
            pred = model(img, coords)
            ade = displacement_error(pred.view(-1,1,2), target.view(-1,1,2))
            ade_error += ade
            fde = final_displacement_error(pred.view(-1,1,2), target.view(-1,1,2))
            fde_error += fde
        
    print(f"Average Displacement Error: {ade_error/len(val_dataloader)}")
    print(f"Final Displacement Error: {fde_error/len(val_dataloader)}")



