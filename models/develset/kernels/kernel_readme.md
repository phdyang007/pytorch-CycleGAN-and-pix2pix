# readme of kernels


## 2021-02-20 12:46:46 updates

1. The folder `np` saved the kernel in `numpy` format.
2. The folder `torch` saved the kernel in `torch` format.
## 2020-12-19 18:04:41 updates

Because we need to test the effect of kernels.
The combokernels should have 24 kernels.
So the comboOptkernel shape is updated to [4, 24, 35, 35]
the comboOptkernel[type][0] means there is only one kernel used in computation.

## 2020-12-01 12:04:41
1. the `comboOptKernel.npy` save the precomputed kernels.

```
lithosim/pvbandsim/pvbandsim.cpp:733
    void initialComboKernel()

\\ this line restrict how many kernel to use
lithosim/pvbandsim/pvbandsim.cpp:758
    nk = 9;

\\ comboOptKernel
shape: [4, 1, 35, 35]
```

2. the `optKernel.npy` save the kernels and the transposed kernels.

```
\\ optKernels
shape: [4, 24, 35, 35]
also you can restrict how many kernel to use when litho simulate.
```

3. the `optKernel_scale.npy` save the kernels scales.

```
\\ optKernel_scale
shape: [4, 24]

[[86.943428 35.417973 35.407127 14.170837 11.240212  4.143778  4.142495
   1.910848  1.716264  1.584437  1.505652  1.500406  1.466139  1.149518
   1.149352  1.010004  0.99177   0.941917  0.941153  0.78716   0.565007
   0.564228  0.490721  0.448742]
 [83.156715 34.974804 34.963085 14.879087 11.032078  4.448975  4.447021
   2.218136  1.612225  1.53863   1.406106  1.363298  1.356663  1.119907
   1.119321  1.097257  0.951011  0.950398  0.828199  0.810574  0.573675
   0.571789  0.485302  0.472674]
 [86.943428 35.417973 35.407127 14.170837 11.240212  4.143778  4.142495
   1.910848  1.716264  1.584437  1.505652  1.500406  1.466139  1.149518
   1.149352  1.010004  0.99177   0.941917  0.941153  0.78716   0.565007
   0.564228  0.490721  0.448742]
 [83.156715 34.974804 34.963085 14.879087 11.032078  4.448975  4.447021
   2.218136  1.612225  1.53863   1.406106  1.363298  1.356663  1.119907
   1.119321  1.097257  0.951011  0.950398  0.828199  0.810574  0.573675
   0.571789  0.485302  0.472674]]
```