import torch
import torch.nn as nn
from datasets import ImageNet
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

dataset = ImageNet('/home/datacenter/ssd2/ImageNet_val', '/home/zhaiyize/models/resnet_inception/data/imagenet_classes.txt', '/home/zhaiyize/models/resnet_inception/data/imagenet_2012_validation_synset_labels.txt')
val_data = DataLoader(dataset=dataset,
                        batch_size=32,
                        shuffle=True,
                        num_workers=4)
for img, label in val_data:
    print('Image batch dimensions:', img.shape)
    print('Image label dimensions:', label.shape)
    break

writer = SummaryWriter(log_dir='./model', comment='Inception_Resnet')
images, labels = next(iter(val_data))

class Basic_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, bias):
        super(Basic_Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Stem(nn.Module):
    def __init__(self, in_channels):
        super(Stem, self).__init__()
        self.feature = nn.Sequential(
            Basic_Conv2d(in_channels, 32, 3, stride=2, padding=0, bias=False),  # 149 x 149 x 32
            Basic_Conv2d(32, 32, 3, stride=1, padding=0, bias=False),  # 147 x 147 x 32
            Basic_Conv2d(32, 64, 3, stride=1, padding=1, bias=False),  # 147 x 147 x 64
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),  # 73 x 73 x 64
            Basic_Conv2d(64, 80, 1, stride=1, padding=0, bias=False),  # 72 x 72 x 80
            Basic_Conv2d(80, 192, 3, stride=1, padding=0, bias=False),  # 71 x 71 x 192
            nn.MaxPool2d(3, stride=2, padding=0), # 35 x 35 x 192
        )
        self.branch_0 = Basic_Conv2d(192, 96, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Basic_Conv2d(192, 48, 1, stride=1, padding=0, bias=False),
            Basic_Conv2d(48, 64, 5, stride=1, padding=2, bias=False),
        )
        self.branch_2 = nn.Sequential(
            Basic_Conv2d(192, 64, 1, stride=1, padding=0, bias=False),
            Basic_Conv2d(64, 96, 3, stride=1, padding=1, bias=False),
            Basic_Conv2d(96, 96, 3, stride=1, padding=1, bias=False),
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            Basic_Conv2d(192, 64, 1, stride=1, padding=0, bias=False)
        )
    
    def forward(self, x):
        x = self.feature(x)
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)   # out_channels=320

writer.add_graph(Stem(3), images)


class Inception_ResNet_A(nn.Module):
    def __init__(self, in_channels, scale=1.0):
        super(Inception_ResNet_A, self).__init__()
        self.scale = scale
        self.branch_0 = Basic_Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Basic_Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False),
            Basic_Conv2d(32, 32, 3, stride=1, padding=1, bias=False)
        )
        self.branch_2 = nn.Sequential(
            Basic_Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False),
            Basic_Conv2d(32, 48, 3, stride=1, padding=1, bias=False),
            Basic_Conv2d(48, 64, 3, stride=1, padding=1, bias=False)
        )
        self.conv = nn.Conv2d(128, 320, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x = torch.cat((x0, x1, x2), dim=1)
        x_res = self.conv(x)
        x_res = self.relu(x_res + self.scale * x_res)
        return x_res  # out_channels=320


class Reduction_A(nn.Module):
    def __init__(self, in_channels):
        super(Reduction_A, self).__init__()
        self.branch0 = Basic_Conv2d(in_channels, 384, kernel_size=3, stride=2, padding=0, bias=False)
        self.branch1 = nn.Sequential(
            Basic_Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0, bias=False),
            Basic_Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            Basic_Conv2d(256, 384, kernel_size=3, stride=2, padding=0, bias=False)
        )
        self.branch2 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return torch.cat((x0, x1, x2), dim=1)  # out_channels=1088


class Inception_ResNet_B(nn.Module):
    def __init__(self,in_channels, scale=1.0):
        super(Inception_ResNet_B, self).__init__()
        self.scale = scale
        self.branch_0 = Basic_Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Basic_Conv2d(in_channels, 128, 1, stride=1, padding=0, bias=False),
            Basic_Conv2d(128, 160, (1, 7), stride=1, padding=(0, 3), bias=False),
            Basic_Conv2d(160, 192, (7, 1), stride=1, padding=(3, 0), bias=False)
        )
        self.conv = nn.Conv2d(384, 1088, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        return self.relu(x + self.scale * x_res)  # out_channels=1088


class Reduction_B(nn.Module):
    def __init__(self, in_channels):
        super(Reduction_B, self).__init__()
        self.branch_0 = nn.Sequential(
            Basic_Conv2d(in_channels, 256, 1, stride=1, padding=0, bias=False),
            Basic_Conv2d(256, 384, 3, stride=2, padding=0, bias=False)
        )
        self.branch_1 = nn.Sequential(
            Basic_Conv2d(in_channels, 256, 1, stride=1, padding=0, bias=False),
            Basic_Conv2d(256, 288, 3, stride=2, padding=0, bias=False),
        )
        self.branch_2 = nn.Sequential(
            Basic_Conv2d(in_channels, 256, 1, stride=1, padding=0, bias=False),
            Basic_Conv2d(256, 288, 3, stride=1, padding=1, bias=False),
            Basic_Conv2d(288, 320, 3, stride=2, padding=0, bias=False)
        )
        self.branch_3 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)  # out_channels=2080


class Inception_ResNet_C(nn.Module):
    def __init__(self, in_channels, scale=1.0, activation=True):
        super(Inception_ResNet_C, self).__init__()
        self.scale = scale
        self.activation = activation
        self.branch_0 = Basic_Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Basic_Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False),
            Basic_Conv2d(192, 224, (1, 3), stride=1, padding=(0, 1), bias=False),
            Basic_Conv2d(224, 256, (3, 1), stride=1, padding=(1, 0), bias=False)
        )
        self.conv = nn.Conv2d(448, 2080, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)  

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        if self.activation:
            return self.relu(x + self.scale * x_res)
        return x + self.scale * x_res  # out_channels=2080


class Inception_ResNet_v2(nn.Module):
    def __init__(self, in_channels=3, classes=1000):
        super(Inception_ResNet_v2, self).__init__()
        blocks = []
        blocks.append(Stem(in_channels))
        for i in range(10):
            blocks.append(Inception_ResNet_A(320, 0.17))
        blocks.append(Reduction_A(320))
        for i in range(20):
            blocks.append(Inception_ResNet_B(1088, 0.10))
        blocks.append(Reduction_B(1088))
        for i in range(9):
            blocks.append(Inception_ResNet_C(2080, 0,20))
        blocks.append(Inception_ResNet_C(2080, activation=False))
        self.feature = nn.Sequential(*blocks)
        self.conv = Basic_Conv2d(2080, 1536, 1, stride=1, padding=0, bias=False)
        self.global_average_poolong = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.8)
        self.linear = nn.Linear(1536, classes)

    def forward(self, x):
        x = self.feature(x)
        x = self.conv(x)
        x = self.global_average_poolong(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

writer.add_graph(Inception_ResNet_v2(), images)