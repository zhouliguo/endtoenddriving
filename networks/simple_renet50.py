# 定义简单的卷积神经网络模型ResNet50
import torch
from torchvision import models
from torch import nn

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return attention * x
    
class SimpleResNet50(nn.Module):
    def __init__(self, hparams):
        super(SimpleResNet50, self).__init__()
        self.hparams = hparams
        self.resnet50 = models.resnet50()
        self.resnet50.fc = nn.Linear(in_features=2048, out_features=256, bias=True)
        self.coordinate_fc = nn.Linear(in_features=self.hparams['num_coor'], out_features=256)
        
        self.fc = nn.Linear(in_features=256 + hparams['num_coor'], out_features=self.hparams['out_features'], bias=True)
        # self.fc = nn.Linear(in_features=256, out_features=self.hparams['out_features'], bias=True)

    def forward(self, images,coordinates):
        images_features = self.resnet50(images)
        coordinates_features = coordinates
        combined_features = torch.cat((images_features, coordinates_features), dim=1)
        output = self.fc(combined_features)
        return output 
    
    def test_coordinates(self, coordinates):
        coordinates_features = self.coordinate_fc(coordinates)
        return coordinates_features
    
    def test_images(self, images):
        images_features = self.resnet50(images)
        return images_features
    
class ResNet50WithPos(nn.Module):
    def __init__(self, hparams):
        super(ResNet50WithPos, self).__init__()
        self.hparams = hparams
        self.resnet50 = models.resnet50()

        # Modify the first convolutional layer to accept the desired number of input channels
        # (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_input_channels = 3 + hparams['num_coor'] + hparams['input_n']

        self.resnet50.conv1 = self.modify_conv1(self.resnet50.conv1, num_input_channels)

        # 不能简单改变in_channels，需要创建新的卷积层
        # self.resnet50.conv1.in_channels = num_input_channels
        # print("Modified input channels:", self.resnet50.conv1.in_channels)
        # print("Conv1 weights shape:", self.resnet50.conv1.weight.shape)

        self.resnet50.fc = nn.Linear(in_features=2048, out_features=self.hparams['out_features'], bias=True)

    def forward(self, images):
        # images_combinate = self.catchImageWithPos(images, coordinates)
        output = self.resnet50(images)
        return output 
    
    def catchImageWithPos(self, images, coordinates):
        '''
        把每一个坐标值放在image的新通道上
        '''

        batch_size, num_channels, height, width = images.size()
        for i in range(coordinates.size(1)):
            coord_channel = torch.full((batch_size, 1, height, width), 
                                       coordinates[:, i].item(), dtype=images.dtype, device=images.device)
            images = torch.cat((images, coord_channel), dim=1)
        print(images.shape)
        
        return images
    
    def modify_conv1(self, conv1, num_input_channels):
        # Create a new convolutional layer with the desired number of input channels
        new_conv1 = nn.Conv2d(num_input_channels, conv1.out_channels, kernel_size=conv1.kernel_size,
                              stride=conv1.stride, padding=conv1.padding, bias=False)

        return new_conv1
    
class ImagePositionEnhancementResMet50(nn.Module):
    def __init__(self, hparams):
        '''
        middle_features: 通过ResNet50提取的图片特征数量

        提取的图片特征添加过去坐标/角度进行训练
        
        TODO？过去的坐标输出归一化
        '''

        super(ImagePositionEnhancementResMet50, self).__init__()
        self.hparams = hparams
        self.resnet50 = models.resnet50()
        self.middle_features = 256
        num_input_channels = 3 + hparams['input_past_channels']

        # Modify the first convolutional layer to accept the desired number of input channels
        # (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        

        self.resnet50.conv1 = self.modify_conv1(self.resnet50.conv1, num_input_channels)
        '''
        不能简单改变in_channels，需要创建新的卷积层
        self.resnet50.conv1.in_channels = num_input_channels
        print("Modified input channels:", self.resnet50.conv1.in_channels)
        print("Conv1 weights shape:", self.resnet50.conv1.weight.shape)
        '''

        self.resnet50.fc = nn.Linear(in_features=2048, out_features=self.middle_features, bias=True)
        self.coor_fc = nn.Linear(in_features=hparams['num_coor'], out_features=self.middle_features, bias=True)
        self.final_fc = nn.Linear(self.middle_features*2, self.hparams['out_features'])

    def forward(self, images, coordinates):
        images_features = self.resnet50(images)
        # output = images_features
        coordinates_features = self.coor_fc(coordinates)

        fused_features = torch.cat((images_features, coordinates_features), dim=1)
        # 放大coordinates的features

        output = self.final_fc(fused_features)

        return output 
    
    def modify_conv1(self, conv1, num_input_channels):
        # Create a new convolutional layer with the desired number of input channels
        new_conv1 = nn.Conv2d(num_input_channels, conv1.out_channels, kernel_size=conv1.kernel_size,
                              stride=conv1.stride, padding=conv1.padding, bias=False)

        return new_conv1
    
if __name__ == '__main__':

    print("restnet50", models.resnet50())

    hparams = {
        "num_coor": 6,
        "input_past_channels": 3*2,
        "out_features": 3,
    }
    model = ImagePositionEnhancementResMet50(hparams)
    # print(model)

    test_image = torch.randn((6, 9, 224, 224))  # torch.Size([1, 3, 224, 224])
    
    test_coordinates = torch.randn((6, hparams['num_coor']))  # torch.Size([1, 6])
    # print(test_coordinates.size(1))

    model.eval()

    with torch.no_grad():
        output = model(test_image, test_coordinates)
        print(output.shape)
