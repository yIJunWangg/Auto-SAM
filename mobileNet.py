#!/usr/bin/env python
# !-*-coding:utf-8 -*-
# !@Time    :2023/4/29 10:10
# !@Author  : murInj
# !@Filer    : .py
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size = max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(Block, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)
        self.se = SeModule(expand_size) if se else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        skip = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))

        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)


class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=1000, act=nn.ReLU):
        super(MobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = act(inplace=True)

        # self.bneck = nn.Sequential(
        #     Block(3, 16, 16, 16, nn.ReLU, True, 2),  # Output: (5, 16, 80, 80)
        #     Block(3, 16, 72, 24, nn.ReLU, False, 2),  # Output: (5, 24, 40, 40)
        #     Block(3, 24, 88, 24, nn.ReLU, False, 1),  # Output: (5, 24, 40, 40)
        #     Block(5, 24, 96, 40, act, True, 2),  # Output: (5, 40, 20, 20)
        #     Block(5, 40, 240, 40, act, True, 1),  # Output: (5, 40, 20, 20)
        #     Block(5, 40, 240, 40, act, True, 1),  # Output: (5, 40, 20, 20)
        #     Block(5, 40, 120, 48, act, True, 1),  # Output: (5, 48, 20, 20)
        #     Block(5, 48, 144, 48, act, True, 1),  # Output: (5, 48, 20, 20)
        #     Block(5, 48, 288, 48, act, True, 2),  # Output: (5, 96, 10, 10)
        #     Block(5, 96, 576, 96, act, True, 1),  # Output: (5, 96, 10, 10)
        #     Block(5, 96, 576, 96, act, True, 1),  # Output: (5, 96, 10, 10)
        # )

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU, True, 2),  # Output: (5, 16, 80, 80)
            Block(3, 16, 72, 24, nn.ReLU, False, 2),  # Output: (5, 24, 40, 40)
            Block(3, 24, 88, 24, nn.ReLU, False, 1),  # Output: (5, 24, 40, 40)
            Block(5, 24, 96, 40, act, True, 2),  # Output: (5, 40, 20, 20)
            Block(5, 40, 240, 40, act, True, 1),  # Output: (5, 40, 20, 20)
            Block(5, 40, 240, 40, act, True, 1),  # Output: (5, 40, 20, 20)
            Block(5, 40, 120, 48, act, True, 1),  # Output: (5, 48, 20, 20)
            Block(5, 48, 144, 48, act, True, 1),  # Output: (5, 48, 20, 20)
            Block(5, 48, 288, 48, act, True, 2),  # Output: (5, 96, 10, 10)
            Block(5, 96, 576, 96, act, True, 1),  # Output: (5, 96, 10, 10)
            Block(5, 96, 576, 96, act, True, 1),  # Output: (5, 96, 10, 10)
        )

        self.conv2 = nn.Conv2d(96, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.hs2 = act(inplace=True)

        self.conTran = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 4, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=2, stride=2),
        )

        self.sigmoid = nn.Sigmoid()

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)

        out = self.hs2(self.bn2(self.conv2(out)))
        out = self.sigmoid(self.conTran(out))

        return out
