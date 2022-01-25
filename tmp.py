import numpy as np 

import matplotlib.pyplot as plt

l2 = np.loadtxt("l2.csv", delimiter=',')
pvb = np.loadtxt("pvb.csv", delimiter=',')
iter = np.arange(1,25)
print(l2, pvb, iter)