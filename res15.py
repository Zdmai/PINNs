import torch
import torch.nn as nn
import torch.nn.functional as F

# model

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, bias=False, padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, bias=False, padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d()
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('conv', nn.Conv2d(in_channels, out_channels, 1, bias=False, padding='same'))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))
        
    
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = F.relu(self.bn2(self.conv2(y)), inplace=True)
        y = y + self.shortcut(x)
        y = F.relu(y, inplace=True)
        y = self.dropout(y)
        return y


class EPNet(nn.Module):
    def __init__(self, config):
        super(EPNet, self).__init__()
        shape = config['input_shape']
        input_channel = config['input_channels']
        self.conv = nn.Conv2d(input_channel, 32, 5, padding='same')
        self.stage1 = BasicBlock(32, 64)
        self.stage2 = BasicBlock(64, 128)
        self.stage3 = BasicBlock(128, 128)
        with torch.no_grad():
            self.feature = self._forward_test(torch.zeros(shape)).view(-1).shape[0]
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature, config['n_classes']),
            nn.Sigmoid()
        )
    
    def _forward_test(self, x):
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        # print("average pool:", x.shape)
        return x

    def forward(self, x):
        x = self._forward_test(x)
        # x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

