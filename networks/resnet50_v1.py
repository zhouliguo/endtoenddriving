import torch
import torch.nn as nn
import torchvision

class ResnetBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        # remove the last layer of resnet50
        self.resnet50.fc = nn.Identity()
        self.fc1 = nn.Linear(2048, 64)
        self.fc2 = nn.Linear(65, 12)
        self.actv = nn.ReLU()

    def forward(self, x, yaw):
        x = self.actv(self.resnet50(x.float()))
        x = self.actv(self.fc1(x))
        x = torch.cat([x, yaw.unsqueeze(1)], dim=1)
        out = self.fc2(x)
        return out

class SeqResnet(nn.Module):
    def __init__(self, input_n=4, predict_n=12):
        super().__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.resnet50.conv1 = nn.Conv2d(3*5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # remove the last layer of resnet50
        self.resnet50.fc = nn.Identity()
        self.coor_fc = nn.Linear(in_features=3*(input_n + 1), out_features=64)
        self.fc1 = nn.Linear(2048, 64)
        self.final_fc = nn.Linear(128, 18)
        self.actv = nn.ReLU()

    def forward(self, x, coords):
        x = self.actv(self.resnet50(x.flatten(1, 2).float())) # After flattening, x has shape (B, 15, H, W) 
        x = self.actv(self.fc1(x)) # (B, 64)
        coords = self.coor_fc(coords.flatten(1, 2).float()) # (B, (input_n + 1)*3) -> (B, 64)
        
        x = torch.cat([x, coords], dim=1)
        out = self.final_fc(x)
        return out

class ResnetLSTM(nn.Module):
    def __init__(self, input_n=4, predict_n=12):
        super().__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        # remove the last layer of resnet50
        self.resnet50.fc = nn.Identity()

        self.coor_fc = nn.Linear(in_features=2, out_features=64)
        self.fc1 = nn.Linear(2048, 64)
        self.lstm = nn.LSTM(128, 512, 2) # 64+64
        self.final_fc = nn.Linear(512, 12)
        self.actv = nn.ReLU()

    def forward(self, x, coords):
        B, N = x.shape[0], x.shape[1]
        x = x.flatten(0, 1) # (5*B, 3, 288, 512)
        coords = coords.flatten(0, 1) # (5*B, 2)

        x = self.actv(self.resnet50(x.float()))
        x = self.actv(self.fc1(x))
        coords = self.coor_fc(coords.float())
        x = torch.cat([x, coords], dim=1)
        
        x = x.view(B, N, -1) # (B, 5, 128)
        x = x.permute(1,0,2)
        x, _ = self.lstm(x) # (5, B, 512)
        x = x.permute(1,0,2) # (B, 5, 512)
        out = self.final_fc(x[:,-1]) # (B, 12)
        return out