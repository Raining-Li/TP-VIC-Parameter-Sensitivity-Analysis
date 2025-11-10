# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np

df = pd.read_csv(r"F:\allvar_normalized.txt", sep='\t')  # 请替换为实际文件路径

UQA = df['UQA'].values

euclidean_distances = {}

for column in df.columns:
    if column != 'basin_var' and column != 'UQA':  # 排除非变量列和 'kalakashi_mean' 本身
        # 获取其他列数据
        other_column = df[column].values
        # 计算欧几里得距离
        distance = np.sqrt(np.sum((UQA - other_column) ** 2))
        # print((UQA - other_column) ** 2)
        euclidean_distances[column] = distance

# 打印计算出的欧几里得距离
for column, distance in euclidean_distances.items():

    print(f"Euclidean distance between 'UQA' and '{column}': {distance}")
