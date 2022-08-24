# -*- coding: utf-8 -*-
"""
@time : 2022/8/24 10:38
@author : fuqiang
@file : model.py
@project : dataframe
"""
from torchvision import models
from torch import nn


def init_resnet18(classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, classes), nn.Sigmoid())
    return model