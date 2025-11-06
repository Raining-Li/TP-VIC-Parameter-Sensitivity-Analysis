from __future__ import division
import numpy as np
from scipy.stats import norm, gaussian_kde, rankdata
import read_param_file
import matplotlib.pyplot as plt
import os
import pandas as pd
from SALib.analyze import delta
# Perform Delta moment-independent Analysis on file of model results
# Returns a dictionary with keys 'delta', 'delta_conf', 'S1', and 'S1_conf'
# Where each entry is a list of size D (the number of parameters)
# Containing the indices in the same order as the parameter file

def analyze(pfile, input_file, Y,outfile, delim=' '):
    param_file = read_param_file.read_param_file(pfile)
    # Y = np.loadtxt(output_file, delimiter=delim, usecols=(column,))
    X = np.loadtxt(input_file, delimiter=delim, ndmin=2)
    print(X.shape)
    if len(X.shape) == 1:
        X = X.reshape((len(X), 1))


    delta_results = pd.DataFrame(delta.analyze(param_file, X, Y,num_resamples=100))
    delta_results.to_csv(outfile,mode='a',sep='\t',index=None)
    print(delta_results)


def datalists(path):
    arr = []
    lists = os.listdir(path)
    for i in lists:
        # if i.split('.')[0].split('_')[-1] == 'kge':#according to dataformat modify
        fni = path + '\\'+i
        arr.append(fni)
    return arr

if __name__ == '__main__':
    VICresult_path = r"D:\SA\simulation_error_SLH"
    paramfile = r"D:\SA\param.txt"
    paramset=r"D:\SA\VIC_param_Symme.txt"
    VICresult_list = datalists(VICresult_path)

    outpath0 = r"D:\SA\SA_result_delta"
    if not os.path.exists(outpath0):
        os.makedirs(outpath0)
        print(outpath0 + ' 创建成功！！')
    else:
        print(outpath0+' 存在！！')

    for VICresult in VICresult_list:
        filename=VICresult.split('\\')[-1]
        keyword=filename.split('.')[0].split('_')
        # print(filename)

        if 'spaef' not in keyword[-2:] and 'kge' not in keyword[-2:]:
            outfile = os.path.join(outpath0, filename.split('.')[0]+'_kge.txt')
        else:
            outfile = os.path.join(outpath0, filename)
        if not os.path.exists(outfile):
            Ydata_initial = pd.read_csv(VICresult, sep='\s+', header=None)
            Ydata = Ydata_initial.sort_values(by=0).reset_index(drop=True)  # 按第一列进行排序,并重新设置索引

            analyze(paramfile, paramset, Ydata.iloc[:,1].values, outfile)
