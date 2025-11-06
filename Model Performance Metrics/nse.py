# -*-coding:utf-8 -*-

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from scipy.stats import variation, zscore
import math
import os


def Calcnash(obs, sim):
    nash = ("%.3f" % (
            1 - (np.sum((sim - obs) ** 2) / np.sum((obs - np.mean(obs)) ** 2))))  

    return nash


