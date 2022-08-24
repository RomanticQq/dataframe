# -*- coding: utf-8 -*-
"""
@time : 2022/8/24 9:40
@author : fuqiang
@file : train.py
@project : dataframe
"""
import torch
from torch.utils.data.dataloader import DataLoader
from dataset import TrainDataset
from tqdm import tqdm
from model import init_resnet18
import torch.nn.functional as F


width_height = 64
classes = 9
train_data = TrainDataset('./data/WM-811K-simple-val.pkl', width_height, classes)
train_loader = DataLoader(train_data, batch_size=4, shuffle=False, pin_memory=False, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = init_resnet18(classes)
critertion = torch.nn.CrossEntropyLoss()
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
model = model.to(device)
for data, label in tqdm(train_loader):
    data = data.to(device)
    label = label.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = critertion(output, F.one_hot(label, classes).float())
    loss.backward()
    optimizer.step()