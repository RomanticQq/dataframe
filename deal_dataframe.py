# -*- coding: utf-8 -*-
"""
@time : 2022/8/24 10:40
@author : fuqiang
@file : deal_dataframe.py
@project : dataframe
"""
# 处理demo  这里的代码在这个project中跑不同
# 可以去下载WM-811K数据集
import pandas as pd
df = pd.read_pickle('data/WM-811K-simple-val.pkl')
df['failureNum'] = df.failureType
mapping = {'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3,
           'Loc': 4, 'Near-full': 5, 'Random': 6, 'Scratch': 7, 'none': 8}
df = df.replace({'failureNum': mapping})
df = df[(df['failureNum'] >= 0) & (df['failureNum'] <= 8)]
df = df.drop(['failureType'], axis=1)
df = df.drop(['dieSize'], axis=1)
df = df.drop(['lotName'], axis=1)
df = df.drop(['trianTestLabel'], axis=1)
df = df.drop(['waferIndex'], axis=1)
df.to_pickle('./data/WM-811K-torch.pkl')