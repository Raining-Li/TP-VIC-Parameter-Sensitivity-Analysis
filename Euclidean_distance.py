# -*-coding:utf-8 -*-
'''
Description: 计算喀拉喀什河与其他流域的欧氏距离
author:liyazhen
date:2025-11-10
'''
import pandas as pd
import numpy as np

# 假设你的数据存储在一个 CSV 文件中
# 读取数据
df = pd.read_csv(r"F:\TPMFD\meteo\25km_ifthen\allvar_2005-2010_normalized_mean.txt", sep='\t')  # 请替换为实际文件路径

# 获取 'kalakashi_mean' 列
kalakashi_mean = df['kalakashi_mean'].values

# 计算 'kalakashi_mean' 与其他列的欧几里得距离
euclidean_distances = {}

for column in df.columns:
    if column != 'basin_var' and column != 'kalakashi_mean':  # 排除非变量列和 'kalakashi_mean' 本身
        # 获取其他列数据
        other_column = df[column].values
        # 计算欧几里得距离
        distance = np.sqrt(np.sum((kalakashi_mean - other_column) ** 2))
        # print((kalakashi_mean - other_column) ** 2)
        euclidean_distances[column] = distance

# 打印计算出的欧几里得距离
for column, distance in euclidean_distances.items():
    print(f"Euclidean distance between 'kalakashi_mean' and '{column}': {distance}")