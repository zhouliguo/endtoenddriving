import sys 
sys.path.append('.')

import os
import argparse

import torch
from torch.utils.data import DataLoader

from data_preparation.urban_driver_data import NuplanDataset

from networks.urban_driver import UrbanDriverModel

# 测试
def test(cfg):

    if cfg.device == 'cpu':
        device = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device
        device = torch.device('gpu:0')

    test_data = NuplanDataset(data_path = os.path.join(cfg.data_path, 'val'), phase='test')
    test_dataloader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = torch.load(cfg.weight).to(device)

    model.eval()
    with torch.no_grad():
        for input, target in test_dataloader:
            input, target = input.to(device), target.to(device)
            output = model(input)

    torch.cuda.empty_cache()

def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='weights/urban_driver/0.pt', help='total number of training epochs')
    parser.add_argument('--data-path', type=str, default='data/nuplan/mini', help='data path')
    parser.add_argument('--device', type=str, default='cpu', help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')

    return parser.parse_args()

if __name__ == '__main__':
    cfg = parse_cfg()

    test(cfg)
