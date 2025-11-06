from __future__ import division

# Optional - turn off bytecode (.pyc files)
import os
import sys
sys.dont_write_bytecode = True
import pandas as pd
import numpy as np
from read_param_file import read_param_file
from pygam import LinearGAM
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

def analyze(pfile, input_file, Y, column=0, delim=' ', plot=True):
    print(pfile)

    param_file = read_param_file.read_param_file(pfile)
    dim = param_file['num_vars']
    X = np.loadtxt(input_file, delimiter=delim)
    if len(X.shape) == 1:
        X = X.reshape((len(X), 1))
    print(X.shape)

    model = LinearRegression()

    model.fit(X, Y)
    base_R2 = model.score(X, Y)

    R2 = []
    _R2 = []
    SA_index = []

    for i in range(dim):
        x = np.delete(X, i, axis=1)
        model = LinearRegression()
        model.fit(x, Y)
        R2 = np.append(R2, model.score(X, Y))
        _R2 = np.append(_R2, abs(R2[i] - base_R2))

    for i in range(dim):
        SA_index = np.append(SA_index, float(_R2[i]) / max(_R2))
        print("Parameter Name          SA_index")
        print(param_file['names'][i] + "              " + str(SA_index[i] * 100))
    return SA_index

def analyze(pfile, input_file, Y, column = 0, delim = ' ', plot = True):

    param_file = read_param_file(pfile)
    dim = param_file['num_vars']
    X = np.loadtxt(input_file, delimiter = delim)

    model = LinearGAM()

    model.fit(X, Y)
    base_GCV = model.score(X, Y)

    GCV = []
    _GCV = []
    SA_index = []

    for i in range(dim):
        x = np.delete(X, i, axis=1)
        model = LinearGAM()
        model.fit(x, Y)
        GCV = np.append(GCV, model.score(X, Y))
        _GCV = np.append(_GCV, abs(GCV[i]-base_GCV))

    for i in range(dim):
        SA_index = np.append(SA_index, float(_GCV[i])/max(_GCV))
        print("Parameter Name          SA_index")
        print(param_file['names'][i]+"              "+str(SA_index[i]*100))
    return SA_index

def datalists(path):
    arr = []
    lists = os.listdir(path)
    for i in lists:
        if i.split('.')[-1] == 'txt':#according to dataformat modify
            fni = path + '\\'+i
            arr.append(fni)
    return arr

if __name__ == '__main__':
    VICresult_path = r"D:\SA\simulation_error_SLH"
    paramfile = r"D:\SA\param.txt"
    VICresult_list = datalists(VICresult_path)

    outpath = r"D:\SA\SA_result_MARs"

    for VICresult in VICresult_list:

        VICresult_name = VICresult.split('\\')[-1]
        outfile = os.path.join(outpath, VICresult_name)

        with open(outfile, mode='a') as file:
            print(VICresult)
            Ydata = pd.read_csv(VICresult, sep='\t', header=0)
            indicator = ['comb', 'kge', 'spaef']
            for ind in indicator:
                print(ind)
                file.write(ind + '\n')
                file.write(
                    'Parameter' + '\t' + 'First_Order' + '\t' + ' First_Order_Confidence' + '\t' + ' Total_Order' + '\t' + ' Total_Order_Confidence' + '\n')
                Y = Ydata[ind][:]
                print(len(Y))

                analyze(r"D:\SA\param.txt", r"D:\SA\VIC_param_Symme.txt", Y, column = 0)