import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# 定义resnet50

class ResNet50Custom(nn.Module):
    def __init__(self, num_classes=512):
        super(ResNet50Custom, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        # self.resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.resnet.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=num_classes), nn.Tanh())

    def forward(self, x):
        x = self.resnet(x)
        return x


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CustomLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        # print('shape of out', out.size())  # batch size * 12
        return out

class Net_coo(nn.Module):
    def __init__(self, input_n=4, out_feature=12):
        super(Net_coo, self).__init__()
        self.input_n = input_n
        self.out_feature = out_feature
        in_feature = 2 * input_n - 2

        self.lstm = CustomLSTM(input_size=2, hidden_size=512, num_layers=2, num_classes=2)
        self.fc = nn.Sequential(
            nn.Linear(12,100),
            nn.ReLU(),
            nn.Linear(100, out_features=out_feature)
        )

    def process_coo(self, x1, x2):
        # x1: old coordinates
        # x2: predicted coordinates from lstm
        x0 = torch.cat((x1, x2), dim=1)
        x = x0[:, 1:, :]
        return x

    def forward(self, x):
        coo = x[1].reshape(-1, 3, 2)
        x1 = self.lstm(coo)
        x1 = x1.reshape(-1, 1, 2)

        coo = self.process_coo(coo, x1)
        x2 = self.lstm(coo)
        x2 = x2.reshape(-1, 1, 2)

        coo = self.process_coo(coo, x2)
        x3 = self.lstm(coo)
        x3 = x3.reshape(-1, 1, 2)

        coo = self.process_coo(coo, x3)
        x4 = self.lstm(coo)
        x4 = x4.reshape(-1, 1, 2)

        coo = self.process_coo(coo, x4)
        x5 = self.lstm(coo)
        x5 = x5.reshape(-1, 1, 2)

        coo = self.process_coo(coo, x5)
        x6 = self.lstm(coo)
        x6 = x6.reshape(-1, 1, 2)
 
        output = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        output = output.reshape(-1, 12)
        output = self.fc(output)
        
        return output


class Net_img(nn.Module):
    def __init__(self, input_n=4, out_feature=12):
        super(Net_img, self).__init__()
        self.input_n = input_n
        self.out_feature = out_feature
        in_feature = 2 * input_n - 2

        self.lstm = CustomLSTM(input_size=2, hidden_size=512, num_layers=2, num_classes=2)
        self.resnet = ResNet50Custom(num_classes=12)
        self.fc1 = nn.Linear(12, 96)
        self.fc = nn.Sequential(
            nn.Linear(108,48),
            nn.ReLU(),
            nn.Linear(48, out_features=out_feature)
        )

    def process_coo(self, x1, x2):
        # x1: old coordinates
        # x2: predicted coordinates from lstm
        x0 = torch.cat((x1, x2), dim=1)
        x = x0[:, 1:, :]
        return x

    def forward(self, x):
        coo = x[1].reshape(-1, 3, 2)
        x1 = self.lstm(coo)
        x1 = x1.reshape(-1, 1, 2)

        coo = self.process_coo(coo, x1)
        x2 = self.lstm(coo)
        x2 = x2.reshape(-1, 1, 2)

        coo = self.process_coo(coo, x2)
        x3 = self.lstm(coo)
        x3 = x3.reshape(-1, 1, 2)

        coo = self.process_coo(coo, x3)
        x4 = self.lstm(coo)
        x4 = x4.reshape(-1, 1, 2)

        coo = self.process_coo(coo, x4)
        x5 = self.lstm(coo)
        x5 = x5.reshape(-1, 1, 2)

        coo = self.process_coo(coo, x5)
        x6 = self.lstm(coo)
        x6 = x6.reshape(-1, 1, 2)
 
        output1 = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        output1 = output1.reshape(-1, 12)
        output1 = self.fc1(output1)

        img = x[0]
        img = self.resnet(img)

        output = torch.cat((img, output1), dim=1)
        output = self.fc(output)
        
        return output


class MyNetwork(nn.Module):
    def __init__(self, input_n=4, out_feature=12):
        super(MyNetwork, self).__init__()
        self.input_n = input_n
        self.out_features = out_feature
        self.in_feature = 2 * input_n - 2

        self.resnet = ResNet50Custom(num_classes=12)
        self.lstm = CustomLSTM(input_size=2, hidden_size=512, num_layers=2, num_classes=96)
        self.fc = nn.Sequential(
            nn.Linear(108, 48),
            nn.ReLU(),
            nn.Linear(48, out_features=out_feature)
        )

    def forward(self, x):
        img = self.resnet(x[0])
        coo = x[1].reshape(-1, 3, 2)
        coo = self.lstm(coo)
        output = torch.cat((img, coo), dim=1)
        output = self.fc(output)
        
        return output

class MyNetwork_NOimage(nn.Module):
    def __init__(self, input_n=4, out_feature=12):
        super(MyNetwork_NOimage, self).__init__()
        self.input_n = input_n
        self.out_feature = out_feature
        in_feature = 2 * input_n - 2

        self.lstm = CustomLSTM(input_size=2, hidden_size=512, num_layers=2, num_classes=512)
        self.fc = nn.Sequential(
            nn.Linear(512,100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, out_features=out_feature)
        )

    def forward(self, x):
        x_coordinate = x[1].reshape(-1, 3, 2)
        x = self.lstm(x_coordinate)
        output = self.fc(x)
        
        return output
 