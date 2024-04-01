import torch
import torch.nn as nn
import torch.nn.functional as F

# model

class new_BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channel1, out_channel2):
        super(new_BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channel1, 3, bias=False, padding='same')
        # print(f"in_channels = {in_channels}, out_channel1 = {out_channel1}")

        self.bn1 = nn.BatchNorm2d(out_channel1)

        self.conv2 = nn.Conv2d(out_channel1, out_channel1, 3, bias=False, padding='same')
        self.bn2 = nn.BatchNorm2d(out_channel1)

        self.shortcut1 = nn.Sequential()
        if in_channels != out_channel1:
            self.shortcut1.add_module('con1', nn.Conv2d(in_channels, out_channel1, 1, bias=False, padding='same'))
            self.shortcut1.add_module('b1', nn.BatchNorm2d(out_channel1))

        self.conv3 = nn.Conv2d(out_channel1, out_channel2, 3, bias=False, padding='same')
        self.bn3 = nn.BatchNorm2d(out_channel2)

        self.conv4 = nn.Conv2d(out_channel2, out_channel2, 3, bias=False, padding='same')
        self.bn4 = nn.BatchNorm2d(out_channel2)

        self.shortcut2 = nn.Sequential()
        if in_channels != out_channel2:
            self.shortcut2.add_module('con2', nn.Conv2d(in_channels, out_channel2, 1, bias=False, padding='same'))
            self.shortcut2.add_module('b2', nn.BatchNorm2d(out_channel2))


    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = F.relu(self.bn2(self.conv2(y)), inplace=True)
        # print(f'y = {y.shape}, x_short = {self.shortcut1(x).shape}')
        y = y + self.shortcut1(x)
        y = F.relu(self.bn3(self.conv3(y)), inplace=True)
        y = F.relu(self.bn4(self.conv4(y)), inplace=True)
        y = y + self.shortcut2(x)
        return y


class new_EPNet(nn.Module):
    def __init__(self, config):
        super(new_EPNet, self).__init__()
        shape = config['input_shape']
        input_channel = config['input_channels']
        self.conv = nn.Conv2d(input_channel, 32, 5, padding='same')
        self.stage1 = new_BasicBlock(32, 32, 64)
        # print('stage1')
        self.stage2 = new_BasicBlock(64, 128, 128)
        # print('stage2')
        # self.stage3 = BasicBlock(128, 128)
        with torch.no_grad():
            self.feature = self._forward(torch.zeros(shape)).view(-1).shape[0]
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature, config['n_classes']),
            nn.Sigmoid()
        )
    
    def _forward(self, x):
        # print("before premute: ", x.shape)
        x = x.permute(0, 3, 2, 1)
        # print("after  premute: ", x.shape)
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        # x = self.stage3(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        # print("average pool:", x.shape)
        return x

    def forward(self, x):
        x = self._forward(x)
        # x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

