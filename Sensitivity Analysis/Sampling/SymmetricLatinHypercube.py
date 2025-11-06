# coding=utf-8
# Optional - turn off bytecode (.pyc files)
#%%

import sys
# sys.dont_write_bytecode = True
sys.path.append(r'D:\SA')

from DoE import sampling
from util import scale_samples_general, read_param_file, discrepancy
import numpy as np
import random as rd
import matplotlib.pyplot as plt
# np.set_printoptions(suppress=True,precision=6)
# Set random seed (does not affect quasi-random Sobol sampling)
seed = 1
np.random.seed(seed)
rd.seed(seed)

# Read the parameter range file and generate samples
param_file = r"D:\SA\param.txt"
pf = read_param_file(param_file)
# Generate samples (choose method here)
# symmetric_LH
param_values = sampling.SymmetricLatinHypercubeDesignDecorrelation(100, pf['num_vars'])

# Samples are given in range [0, 1] by default. Rescale them to your parameter bounds.
scale_samples_general(param_values, pf['bounds'])

# print(type(paes))  #<class 'numpy.ndarray'>
print(param_values)

np.savetxt('VIC_param_Symme.txt', param_values, delimiter=' ',fmt='%.06f')



