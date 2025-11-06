# -*-coding:utf-8 -*-

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from scipy.stats import variation, zscore
import math
import os


def Kling_Gupta(simulation_s, evaluation):
    # calculate error in timing and dynamics r (Pearson's correlation coefficient)
    sim_mean = np.mean(simulation_s)
    obs_mean = np.mean(evaluation)
    # print(simulation_s,evaluation)
    r = np.sum((simulation_s - sim_mean) * (evaluation - obs_mean)) / \
        np.sqrt(np.sum((simulation_s - sim_mean) ** 2) *
                np.sum((evaluation - obs_mean) ** 2))
    # calculate error in spread of flow alpha
    alpha = np.std(simulation_s) / np.std(evaluation)
    # calculate error in volume beta (bias of mean discharge)
    beta = np.sum(simulation_s) / np.sum(evaluation)
    # calculate the Kling-Gupta Efficiency KGE
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    # print(kge)
    return kge

