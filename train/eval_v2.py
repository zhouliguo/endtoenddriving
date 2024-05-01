import sys
sys.path.append('.')
import argparse
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
import torch
from data_preparation.nuscenes_data_v1 import ResNetDataset, SeqResNetDataset
from data_preparation.dataloader_v1 import DT_Img
from networks.resnet50_v1 import ResnetBaseline, SeqResnet, ResnetLSTM
import torchvision.transforms as transforms
import os
import numpy as np
import torch.nn as nn

from stp3.datas.NuscenesData import FuturePredictionDataset
from stp3.trainer import TrainingModule
from stp3.metrics import IntersectionOverUnion, PanopticMetric, PlanningMetric
from stp3.utils.network import preprocess_batch, NormalizeInverse
from stp3.utils.instance import predict_instance_segmentation_and_trajectories
from stp3.utils.visualisation import make_contour

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
    parser.add_argument('--batch_size',
                        type=int, 
                        default=1,
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
    parser.add_argument('--model',
                        type=str, 
                        default='SeqResnet',
                        help='specify the model to train') 
    parser.add_argument('--ckpt',
                        type=str, 
                        help='checkpoint path to be evaluated') 
    parser.add_argument('--stp3_ckpt',
                        type=str, 
                        default='/mnt/ssd/xzt/experiments/ST-P3/models/STP3_plan.ckpt',
                        help='checkpoint path to be evaluated') 
    
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
    batch_size = args.batch_size
    seed_everything(seed)
    is_mini = args.mini

    print("Loading dataset...")
    val_set = DT_Img(args, phase='val')
    val_dataloader = DataLoader(val_set, batch_size=batch_size, drop_last=True, shuffle=False, num_workers=8)

    if args.model == 'ResnetBaseline':
        model = ResnetBaseline().to(device)
    elif args.model == 'SeqResnet':
        model = SeqResnet().to(device)
    elif args.model == 'ResnetLSTM':
        model = ResnetLSTM().to(device)
    
    assert args.ckpt is not None
    model = torch.load(args.ckpt).eval()
    # Evaluate the model
    ade_error, fde_error = 0.0, 0.0

    # Load STP3 for collision evaluation
    stp3_ckpt = args.stp3_ckpt
    trainer = TrainingModule.load_from_checkpoint(stp3_ckpt, strict=True)
    # print(f'Loaded weights from \n {stp3_ckpt}')
    trainer.eval()
    # device = torch.device('cuda:1')
    # trainer.to(device)
    stp3 = trainer.model
    cfg = stp3.cfg

    results = {}
    metric_planning_val = []
    future_second = args.predict_n // 2
    for i in range(future_second):
        metric_planning_val.append(PlanningMetric(cfg, 2*(i+1)).to(device))
    
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
            img, coor_past, target, filename = batch 
            filename = filename[0]  
            occupancy_path = os.path.join('occupancy', filename + '.npy')
            gt_trajs_path = os.path.join('gt_trajs', filename + '.npy')
            occupancy = torch.from_numpy(np.load(occupancy_path)).to(device)
            gt_trajs = torch.from_numpy(np.load(gt_trajs_path)).to(device)

           
            img, coor_past, target = img.to(device), coor_past.to(device), target.to(device)
            pred = model(img, coor_past).view(img.shape[0], -1, 3)
            for j in range(future_second):
                cur_time = (j+1)*2
                # print(pred.shape, target[:,:cur_time].shape, occupancy[:,:cur_time].shape)
                metric_planning_val[j](pred[:,:cur_time].detach(), target[:,:cur_time], occupancy[:,:cur_time])

            ade = displacement_error(pred.view(-1,1,2), target.view(-1,1,2))
            ade_error += ade
            fde = final_displacement_error(pred.view(-1,1,2), target.view(-1,1,2))
            fde_error += fde
    
    results['ade'] = ade_error/len(val_dataloader)
    results['fde'] = fde_error/len(val_dataloader)

    for i in range(future_second):
        scores = metric_planning_val[i].compute()
        for key, value in scores.items():
            results['plan_'+key+'_{}s'.format(i+1)]=value.mean()
    for key, value in results.items():
        print(f'{key} : {value.item()}')
    # print(f"Average Displacement Error: {ade_error/len(val_dataloader)}")
    # print(f"Final Displacement Error: {fde_error/len(val_dataloader)}")



