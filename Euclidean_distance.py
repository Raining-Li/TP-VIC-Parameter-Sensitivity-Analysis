# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np

df = pd.read_csv(r"F:\allvar_normalized.txt", sep='\t')  

UQA = df['UQA'].values

euclidean_distances = {}

for column in df.columns:
    if column != 'basin_var' and column != 'UQA': 

        other_column = df[column].values
        # Euclidean distance
        distance = np.sqrt(np.sum((UQA - other_column) ** 2))
        # print((UQA - other_column) ** 2)
        euclidean_distances[column] = distance


for column, distance in euclidean_distances.items():

    print(f"Euclidean distance between 'UQA' and '{column}': {distance}")

