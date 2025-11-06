# coding=utf-8
# Optional - turn off bytecode (.pyc files)
#%%
# import sys
# sys.path.append(r"D:/UQ-PyL/software/UQ-PyL/")
import sys
# sys.dont_write_bytecode = True
sys.path.append(r'D:\UQ-PyL')
from UQ.DoE import saltelli
from UQ.DoE import sampling
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
param_file = r"D:\UQ-PyL\VIC15_yarkantRunoff.txt"
pf = read_param_file(param_file)
# Generate samples (choose method here)
# saltelli
# # param_values = saltelli.sample(350, pf['num_vars'],calc_second_order = False)
# param_values = saltelli.sample(20, pf['num_vars'],calc_second_order = False)
# symmetric_LH
param_values = sampling.SymmetricLatinHypercubeDesignDecorrelation(500, pf['num_vars'])
print(param_values.shape)
plt.figure()
ax = plt.subplot()
plt.scatter(param_values[:, 0], param_values[:, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.title('Saltelli Sampling')
plt.show()
# Samples are given in range [0, 1] by default. Rescale them to your parameter bounds.
scale_samples_general(param_values, pf['bounds'])

# print(type(paes))  #<class 'numpy.ndarray'>,表示数组
print(param_values)
# np.savetxt('Input_VIC_symmetric_LHde3300.txt', param_values, delimiter=' ',fmt='%.06f')
np.savetxt('VIC_param15_Symme_yar2.txt', param_values, delimiter=' ',fmt='%.06f')


