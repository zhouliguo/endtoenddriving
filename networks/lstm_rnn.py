import torch
import torch.nn as nn
from torchvision import models
from networks.Models import Transformer
# from Models import Transformer

class ImagePositionEnhancementResMet50(nn.Module):
    def __init__(self,input_coor_features=6, input_channels=9 ,out_features=12 ):
        '''
        middle_features: 通过ResNet50提取的图片特征数量

        提取的图片特征添加过去坐标/角度进行训练
        '''

        super(ImagePositionEnhancementResMet50, self).__init__()
        self.resnet50 = models.resnet50()
        self.middle_features = 256
        self.input_coor_features = input_coor_features # input_n*2
        self.input_channels = input_channels # (input_n+1)*3 输入的图片数量×3

        # Modify the first convolutional layer to accept the desired number of input channels
        # (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.resnet50.conv1 = self.modify_conv1(self.resnet50.conv1, self.input_channels)
        '''
        不能简单改变in_channels，需要创建新的卷积层
        self.resnet50.conv1.in_channels = self.input_channels
        print("Modified input channels:", self.resnet50.conv1.in_channels)
        print("Conv1 weights shape:", self.resnet50.conv1.weight.shape)
        '''

        self.resnet50.fc = nn.Linear(in_features=2048, out_features=self.middle_features, bias=True)
        self.coor_fc = nn.Linear(in_features=self.input_coor_features, out_features=self.middle_features, bias=True)
        self.final_fc = nn.Linear(self.middle_features*2, out_features=out_features)

    def forward(self, images, coordinates):
        image = torch.cat(images,dim=1)
        # print(image.shape[1])
        images_features = self.resnet50(image)
        # output = images_features
        coordinates_features = self.coor_fc(coordinates)
        # print(images_features.shape, coordinates_features.shape)
        fused_features = torch.cat((images_features, coordinates_features), dim=1)
        # 放大coordinates的features

        output = self.final_fc(fused_features)

        return output 
    
    def modify_conv1(self, conv1, num_input_channels):
        # Create a new convolutional layer with the desired number of input channels
        new_conv1 = nn.Conv2d(num_input_channels, conv1.out_channels, kernel_size=conv1.kernel_size,
                              stride=conv1.stride, padding=conv1.padding, bias=False)
        return new_conv1
    
class ImageResNet50_LSTM(nn.Module):
    def __init__(self, out_features = 12,input_coor_features=2):
        '''
        middle_features: 通过ResNet50提取的图片特征数量

        提取的图片特征添加过去坐标/角度进行训练
        '''

        super(ImageResNet50_LSTM, self).__init__()
        self.resnet50 = models.resnet50()
        self.middle_features = 256
        lstm_hidden_size = 128
        self.middle_coor_features = 32

        self.coor_fc = nn.Linear(in_features=input_coor_features, out_features=self.middle_coor_features, bias=True)
        self.resnet50.fc = nn.Linear(in_features=2048, out_features=self.middle_features, bias=True)
        self.lstmModel = LstmNet(input_size=self.middle_features+self.middle_coor_features, hidden_size=lstm_hidden_size,
                                 output_size=out_features)

    def forward(self, images, coors):
        # Process each image through ResNet50
        images = images[1:]
        images_features = [self.resnet50(image) for image in images]

        grouped_values = torch.chunk(coors, chunks=int(len(coors)/2), dim=1)
        coors_features = [self.coor_fc(tensor) for tensor in grouped_values]

        fused_features = [torch.cat((img_feat, coor_feat), dim=1) 
                          for img_feat, coor_feat in zip(images_features, coors_features)]
        features_combined = torch.stack(fused_features)
        # print(features_combined.shape)
        output = self.lstmModel(features_combined)
        return output
    
class LstmNet(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=3):
        super().__init__()
 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers) # utilize the LSTM model in torch.nn 
        self.forwardCalculation = nn.Linear(hidden_size, output_size)
 
    def forward(self, _x):
        x, _ = self.lstm(_x)  # size (seq_len, batch, input_size)
        x = x[-1, :, :]  # 去最后一个时间步
        b, h = x.shape
        x = x.view(b, h)
        x = self.forwardCalculation(x)

        '''
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s*b, h)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1) size (seq_len, batch, output_size)
        '''
        return x

class Transformer_image(nn.Module):
    def __init__(self,
            n_src_vocab=500,
            n_trg_vocab=500,
            d_word_vec=512,
            d_model=512,
            d_inner=2048,
            n_layers=6,
            n_head=8,
            d_k=64,
            d_v=64,
            dropout=0.1,
            n_position=1, # 
            n_position_d=6,
            input_dim = 256, #
            output_dim = 6,
            output_features = 6):
        super(Transformer_image, self).__init__()
        self.resnet50 = models.resnet50()
        self.middle_features = input_dim
        self.resnet50.fc = nn.Linear(in_features=2048, out_features=self.middle_features, bias=True)
        self.transformer = Transformer(
                n_src_vocab=500,
                n_trg_vocab=500,
                d_k=32,
                d_v=32,
                d_model=256,
                d_word_vec=256,
                d_inner=1024,
                n_layers=3,
                n_head=4,
                dropout=0.1,
                n_position=n_position,
                n_position_d=n_position_d,
                input_dim=self.middle_features,
                output_dim=output_dim
            )
        # 添加线性层改变输出维度
        # self.linear_output = nn.Linear(output_dim, output_features)
        
    def forward(self, images, true_data):
        images = torch.unbind(images, dim=1) # [batch_size, 6(inputn+1), 3, 384, 704]

        # Process each split image through ResNet50
        images_features = [self.resnet50(image) for image in images] # (batch, len(input_size))

        # Concatenate features along the channel dimension
        features_combined = torch.stack(images_features).transpose(0, 1) # torch.Size([batch_size, 6(input_n+1), 128])
        '''
        例如：我们一开始的输入为：[[0, 3, 4, 5, 6, 7, 8, 1, 2, 2]]，Shape为(1, 10)，表示batch_size为1, 每句10个词。
        在经过Embedding后，Shape就变成了(1, 10, 128)，表示batch_size为1, 每句10个词，每个词被编码为了128维的向量。
        '''
        output = self.transformer(features_combined, true_data)
        # output = self.linear_output(out_features)
        # output = out_features

        return output
    

if __name__ == '__main__':
    # lstm_model = LstmRNN(3, 16, output_size=3, num_layers=1) # 16 hidden units
    # print('LSTM model:', lstm_model)

    lstm = nn.LSTM(input_size = 100, hidden_size = 20, num_layers = 3)
    print("lstm", lstm)

    hparams = {
        "num_coor": 6,
        "input_past_channels": 3*2,
        "out_features": 256,
    }
    model = Transformer_image( )
    # print(model)

    test_image1 = torch.randn((6, 3, 224, 224))  # torch.Size([1, 3, 224, 224])
    test_image2 = torch.randn((6, 3, 224, 224))
    images_list = [test_image1,test_image2,test_image2,test_image2]
    images_packed = torch.stack(images_list)

    model.eval()

    # 输入 (seq_len, batch, input_size)

    # test_image = torch.randn((6, 9, 224, 224)) # batch_size, channels, h, w
    test_out = torch.randn((6, 1, 6)) # batch_size, dim, len

    test_out = torch.randn((6, 6))
    # test_out = nn.Embedding(test_out)

    with torch.no_grad():
        output = model(images_packed, test_out)
        print(output.shape)
