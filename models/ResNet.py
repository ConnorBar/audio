"""
  ResNet18 from scratch in PyTorch 
  https://www.geeksforgeeks.org/resnet18-from-scratch-using-pytorch/
"""

import torch.nn as nn

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_channels, out_channels, stride=1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_channels != self.expansion * out_channels:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(self.expansion * out_channels)
      )

  def forward(self, x):
    out = self.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    out = self.relu(out)
    return out

class ResNet18(nn.Module):
  def __init__(self, num_classes=10):
    super(ResNet18, self).__init__()
    self.in_channels = 64
    self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
    self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
    self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
    self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

  def _make_layer(self, block, out_channels, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_channels, out_channels, stride))
      self.in_channels = out_channels * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x):
    out = self.relu(self.bn1(self.conv1(x)))
    
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)

    out = self.avgpool(out)
    out = out.view(out.size(0), -1) # think this flattens? need to check
    out = self.fc(out)
    return out