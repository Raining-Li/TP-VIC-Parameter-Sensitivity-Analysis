# coding=utf-8
# Optional - turn off bytecode (.pyc files)
#%%

import sys
# sys.dont_write_bytecode = True
sys.path.append(r'D:\SA')
from DoE import saltelli
from DoE import sampling
from UQ.util import scale_samples_general, read_param_file, discrepancy
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
# saltelli
param_values = saltelli.sample(100, pf['num_vars'],calc_second_order = False)

# Samples are given in range [0, 1] by default. Rescale them to your parameter bounds.
scale_samples_general(param_values, pf['bounds'])

np.savetxt('VIC_param_saltelli.txt', param_values, delimiter=' ',fmt='%.06f')



