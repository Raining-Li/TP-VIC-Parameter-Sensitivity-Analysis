# -*-coding:utf-8 -*-

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from scipy.stats import variation, zscore
import math
import os


def filter_nan(s, o, nodata):
    # s-simdata; o-obsdata
    print(s.shape,o.shape)
    data = np.transpose(np.array([s.flatten(), o.flatten()]))
    data[data < nodata] = np.NaN
    data = data[~np.isnan(data).any(1)]

    return data[:, 0], data[:, 1]

def SPAEF(s, o, nodata):
    # remove NANs
    s, o = filter_nan(s, o, nodata)
    # print(s,o)

    bins = round(math.sqrt(len(o)))
    print(bins)
    #compute corr coeff
    alpha = np.corrcoef(s,o)[0,1]
    #compute ratio of CV

    if np.all(variation(o)==0) or np.all(o==0) or np.all(s==0) or np.all(variation(s)==0):
        spaef='nan'
        alpha='nan'
        beta='nan'
        gamma='nan'
    else:
        beta = variation(s)/variation(o)
        #compute zscore mean=0, std=1
        o=zscore(o)
        s=zscore(s)
        #compute histograms
        hobs,binobs = np.histogram(o,bins)
        hsim,binsim = np.histogram(s,bins)
        #convert int to float, critical conversion for the result
        hobs=np.float64(hobs)
        hsim=np.float64(hsim)
        #find the overlapping of two histogram
        minima = np.minimum(hsim, hobs)
        #compute the fraction of intersection area to the observed histogram area, hist intersection/overlap index
        gamma = np.sum(minima)/np.sum(hobs)
        #compute SPAEF finally with three vital components
        spaef = 1- np.sqrt( (alpha-1)**2 + (beta-1)**2 + (gamma-1)**2 )

    return spaef, alpha, beta, gamma

