import torch
from torch import nn
from torch import Tensor
from torchvision import models

class LSTMModel(nn.Module):
    def __init__(self, pretrained=True, input_n=1, in_feature=256, out_features=12) -> None:
        super().__init__()
        self.input_n = input_n
        self.in_feature = in_feature

        self.linear0 = nn.Linear(2, in_feature)
        self.relu0 = nn.ReLU()
        self.lstm = nn.LSTM(in_feature, 256, 2)
        self.linear1 = nn.Linear(256, out_features)

    def forward(self, x: Tensor) -> Tensor:
        bs = x.size(0)
        x = x.view(-1, 2)
        x = self.linear0(x)
        x = self.relu0(x)
        x = x.view(bs, -1, self.in_feature)
        x = x.permute(1,0,2)
        x, (h, c) = self.lstm(x)
        x = x.permute(1,0,2)
        x = self.linear1(x[:,-1])
        
        return x
    
class LSTMModel1(nn.Module):
    def __init__(self, pretrained=True, input_n=5, in_feature=2, out_features=12) -> None:
        super().__init__()
        self.input_n = input_n
        self.in_feature = in_feature
        self.linear0 = nn.Linear(6, in_feature)
        self.lstm = nn.LSTM(in_feature, 512, 2)
        self.linear1 = nn.Linear(512, in_feature)

    def forward(self, x: Tensor) -> Tensor:
        #x = x[1]
        x = self.linear0(x)
        bz = x.size(0)
        x = x.view(bz, -1, self.in_feature)    # batch size, sequence length, feature length
        sl = x.size(1)
        fl = x.size(2)
        k = torch.zeros((sl+6, bz, fl)).to(x.device)    # sequence length, batch size, feature size
        #y = torch.zeros((x.size(0), 6, x.size(1)*x.size(2))).to(x.device) 
        x = x.permute(1,0,2)    # sequence length, batch size, feature size
        k[:sl] = x
        for i in range(6):       
            x, (h, c) = self.lstm(x)
            x = x.flatten(0,1)
            x = self.linear1(x)
            x = x.view(-1, bz, self.in_feature)    # sequence length, batch size, feature size
            #y[:,i] = x.permute(1,0,2).flatten(1,2)
            k[sl+i] = x[-1]
            x[:sl-1] = k[i+1:i+sl]
        k = k[sl:].permute(1,0,2).flatten(1,2)
        return k

class LSTMModel2(nn.Module):
    def __init__(self, pretrained=True, input_n=5, in_feature=2, out_features=12) -> None:
        super().__init__()
        self.input_n = input_n
        self.in_feature = in_feature
        self.linear0 = nn.Linear(2, in_feature)
        self.relu0 = nn.ReLU()
        self.lstm = nn.LSTM(in_feature, 256, 2)
        self.linear1 = nn.Linear(256, 2)
        self.out_features = out_features

    def forward(self, x: Tensor) -> Tensor:
        #x = x[1]
        bs = x.size(0)
        x = x.view(-1, 2)
        x = self.linear0(x)
        x = self.relu0(x)
        x = x.view(bs, -1, self.in_feature)    # batch size, sequence length, feature length
        sl = x.size(1)
        fl = x.size(2)
        k = torch.zeros((sl+6, bs, fl)).to(x.device)    # sequence length, batch size, feature size
        y =  torch.zeros(bs, self.out_features).to(x.device) 
        x = x.permute(1,0,2)    # sequence length, batch size, feature size
        k[:sl] = x
        for i in range(6):
            x, (h, c) = self.lstm(x)
            #x = x.permute(1,0,2)
            x = x.flatten(0,1)
            x = self.linear1(x)
            x = x.view(-1, bs, 2)
            y[:, i*2:i*2+2] = x[-1,:]
            x = x.flatten(0,1)
            x = self.linear0(x)
            x = self.relu0(x)
            x = x.view(-1, bs, self.in_feature)
            k[sl+i] = x[-1]
            x[:sl-1] = k[i+1:i+sl]
        return y
    
class GRUModel(nn.Module):
    def __init__(self, pretrained=True, input_n=1, in_feature=1, out_features=12) -> None:
        super().__init__()
        self.input_n = input_n
        self.in_feature = in_feature

        self.gru = nn.GRU(in_feature, 1024, 2)
        self.linear = nn.Linear(1024, out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1, self.in_feature)
        x = x.permute(1,0,2)
        x, (h, c) = self.gru(x)
        x = x.permute(1,0,2)
        x = self.linear(x[:,-1])
        
        return x
    
class ResNetLSTM(nn.Module):
    def __init__(self, type=50, pretrained=True, input_n=1, out_features=12) -> None:
        super().__init__()
        self.input_n = input_n
        if pretrained:
            weights = 'ResNet'+str(type)+'_Weights.DEFAULT'
        else:
            weights = None
        
        if type == 50:
            self.model = models.resnet50(weights = weights)
        elif type == 101:
            self.model = models.resnet101(weights = weights)
        else:
            self.model = models.resnet152(weights = weights)

        self.model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.lstm = nn.LSTM(1000, 512, 2)
        self.linear = nn.Linear(512, 12)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(0,1)
        x = self.model(x)
        x = x.view(-1, self.input_n-1, 1000)
        x = x.permute(1,0,2)
        x, (h, c) = self.lstm(x)
        x = x.permute(1,0,2)
        x = self.linear(x[:,-1])
        
        return x

class LSTMCellModel(nn.Module):
    def __init__(self, pretrained=True, input_n=1, out_features=12) -> None:
        super().__init__()

        self.lstm1 = nn.LSTMCell(2, 512)
        self.lstm2 = nn.LSTMCell(512, 512)
        self.linear = nn.Linear(512, 2)

    def forward(self, x: Tensor, future = 1) -> Tensor:
        outputs = []

        h_t = torch.zeros(x.size(0), 512, dtype=torch.float32).to(x.device)
        c_t = torch.zeros(x.size(0), 512, dtype=torch.float32).to(x.device)
        h_t2 = torch.zeros(x.size(0), 512, dtype=torch.float32).to(x.device)
        c_t2 = torch.zeros(x.size(0), 512, dtype=torch.float32).to(x.device)

        #x = x.view(-1, 4, 2)
        for input_t in x.split(2, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        
        for i in range(1, future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        outputs = torch.cat(outputs, dim=1)
        return outputs
    
class LSTMCellModel1(nn.Module):
    def __init__(self, pretrained=True, input_n=5, in_feature=2, out_features=12) -> None:
        super().__init__()
        self.out_features = out_features

        self.linear0 = nn.Linear(2, in_feature)
        self.relu0 = nn.ReLU()
        self.linear1 = nn.Linear(256, 2)

        self.lstm00 = nn.LSTMCell(in_feature, 256)
        self.lstm01 = nn.LSTMCell(256, 256)

        self.lstm10 = nn.LSTMCell(in_feature, 256)
        self.lstm11 = nn.LSTMCell(256, 256)

        '''
        self.lstm20 = nn.LSTMCell(in_feature, 512)
        self.lstm21 = nn.LSTMCell(512, 512)

        self.lstm30 = nn.LSTMCell(in_feature, 512)
        self.lstm31 = nn.LSTMCell(512, 512)

        self.lstm40 = nn.LSTMCell(in_feature, 512)
        self.lstm41 = nn.LSTMCell(512, 512)

        self.lstm50 = nn.LSTMCell(in_feature, 512)
        self.lstm51 = nn.LSTMCell(512, 512)

        self.lstm60 = nn.LSTMCell(in_feature, 512)
        self.lstm61 = nn.LSTMCell(512, 512)
        '''

        self.linear = nn.Linear(512, out_features)

    def forward(self, x: Tensor, future = 1) -> Tensor:
        #x = x[1]

        bs = x.size(0)

        h_t = torch.zeros(bs, 256, dtype=torch.float32).to(x.device)
        c_t = torch.zeros(bs, 256, dtype=torch.float32).to(x.device)
        h_t2 = torch.zeros(bs, 256, dtype=torch.float32).to(x.device)
        c_t2 = torch.zeros(bs, 256, dtype=torch.float32).to(x.device)
        y =  torch.zeros(bs, self.out_features).to(x.device)

        x = x.view(-1, 2)
        x = self.linear0(x)
        x = self.relu0(x)
        x = x.view(-1, 3, 256)

        h_t, c_t = self.lstm00(x[:,0], (h_t, c_t))
        h_t2, c_t2 = self.lstm01(h_t, (h_t2, c_t2))

        h_t, c_t = self.lstm00(x[:,1], (h_t, c_t))
        h_t2, c_t2 = self.lstm01(h_t, (h_t2, c_t2))

        h_t, c_t = self.lstm00(x[:,2], (h_t, c_t))
        h_t2, c_t2 = self.lstm01(h_t, (h_t2, c_t2))

        # 1
        x = self.linear1(h_t2)
        y[:,:2] = x
        x = self.linear0(x)
        x = self.relu0(x)

        h_t, c_t = self.lstm00(x, (h_t, c_t))
        h_t2, c_t2 = self.lstm01(h_t, (h_t2, c_t2))

        # 2
        x = self.linear1(h_t2)
        y[:,2:4] = x
        x = self.linear0(x)
        x = self.relu0(x)

        h_t, c_t = self.lstm00(x, (h_t, c_t))
        h_t2, c_t2 = self.lstm01(h_t, (h_t2, c_t2))

        # 3
        x = self.linear1(h_t2)
        y[:,4:6] = x
        x = self.linear0(x)
        x = self.relu0(x)

        h_t, c_t = self.lstm00(x, (h_t, c_t))
        h_t2, c_t2 = self.lstm01(h_t, (h_t2, c_t2))
    
        # 4
        x = self.linear1(h_t2)
        y[:,6:8] = x
        x = self.linear0(x)
        x = self.relu0(x)

        h_t, c_t = self.lstm00(x, (h_t, c_t))
        h_t2, c_t2 = self.lstm01(h_t, (h_t2, c_t2))

        # 5
        x = self.linear1(h_t2)
        y[:,8:10] = x
        x = self.linear0(x)
        x = self.relu0(x)

        h_t, c_t = self.lstm00(x, (h_t, c_t))
        h_t2, c_t2 = self.lstm01(h_t, (h_t2, c_t2))

        # 6
        x = self.linear1(h_t2)
        y[:,10:12] = x
        x = self.linear0(x)
        x = self.relu0(x)

        #x = self.linear(h_t2)
        return y
    
class ResNetLSTMCell(torch.nn.Module):
    def __init__(self, type=50, pretrained=True, input_n=1, out_features=12) -> None:
        super().__init__()
        if pretrained:
            weights = 'ResNet'+str(type)+'_Weights.DEFAULT'
        else:
            weights = None
        
        if type == 50:
            self.model = models.resnet50(weights = weights)
        elif type == 101:
            self.model = models.resnet101(weights = weights)
        else:
            self.model = models.resnet152(weights = weights)

        self.lstm1_1 = nn.LSTMCell(2, 512)
        self.lstm1_2 = nn.LSTMCell(512, 512)
        self.linear1 = nn.Linear(512, 2)

        self.lstm2_1 = nn.LSTMCell(1000, 512)
        self.lstm2_2 = nn.LSTMCell(512, 1000)
        self.linear2 = nn.Linear(1000, 2)

        #self.lstm = torch.nn.LSTM(input_size=1000, hidden_size = 12, num_layers = 2)

    '''
    只用坐标
    '''
    def forward1(self, x: Tensor) -> Tensor:
        outputs = []

        h_t = torch.zeros(x.size(0), 512, dtype=torch.float32).to(x.device)
        c_t = torch.zeros(x.size(0), 512, dtype=torch.float32).to(x.device)
        h_t2 = torch.zeros(x.size(0), 512, dtype=torch.float32).to(x.device)
        c_t2 = torch.zeros(x.size(0), 512, dtype=torch.float32).to(x.device)

        #x = x.view(-1, 4, 2)
        for input_t in x.split(2, dim=1):
            h_t, c_t = self.lstm1_1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm1_2(h_t, (h_t2, c_t2))
            output = self.linear1(h_t2)
            outputs += [output]
        for i in range(1,6):
            h_t, c_t = self.lstm1_1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm1_2(h_t, (h_t2, c_t2))
            output = self.linear1(h_t2)
            outputs += [output]

        outputs = torch.cat(outputs, dim=1)
        return outputs

    '''
    只用图像
    '''
    def forward2(self, x: Tensor) -> Tensor:
        outputs = []
        bs = x.size(0)
        device = x.device
        h_t = torch.zeros(bs, 512, dtype=torch.float32).to(device)
        c_t = torch.zeros(bs, 512, dtype=torch.float32).to(device)
        h_t2 = torch.zeros(bs, 1000, dtype=torch.float32).to(device)
        c_t2 = torch.zeros(bs, 1000, dtype=torch.float32).to(device)
    
        x = x.flatten(0, 1)

        x = self.model(x)

        x = x.view(bs, -1, 1000)

        for i in range(x.size(1)):
            h_t, c_t = self.lstm2_1(x[:,i], (h_t, c_t))
            h_t2, c_t2 = self.lstm2_2(h_t, (h_t2, c_t2))
            output = self.linear2(h_t2)
            outputs += [output]
        for i in range(1,6):
            h_t, c_t = self.lstm2_1(h_t2, (h_t, c_t))
            h_t2, c_t2 = self.lstm2_2(h_t, (h_t2, c_t2))
            output = self.linear2(h_t2)
            outputs += [output]

        outputs = torch.cat(outputs, dim=1)
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.forward1(x[1])
        x2 = self.forward2(x[0])
        return x1

# 测试网络
if __name__ == '__main__':

    #rnn = torch.nn.LSTM(1000, 12, 2)    # (input_feature_size, hidden_size, num_layers)
    #input = torch.randn(5, 8, 1000)   # (seq_length, batch_size, feature_size)
    #h0 = torch.randn(2, 8, 12)  # (num_layers, batch_size, hidden_size)
    #c0 = torch.randn(2, 8, 12)
    #output, (hn, cn) = rnn(input, (h0, c0))

    model = LSTMCellModel1()
    
    x = torch.randn(4,8)

    x = model(x)

    x = torch.randn(4,20)

    x = model(x)

    loss_function = torch.nn.L1Loss()
    input = torch.FloatTensor([1,2,3])
    target = torch.FloatTensor([4,8,10])
    loss = loss_function(input, target)
    print(loss.item())