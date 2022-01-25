'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2021-01-14 13:38:24
LastEditTime: 2021-03-23 18:32:56
Contact: cgjhaha@qq.com
Description: visulize the levelset function phi
'''
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from models.const import *

# PLT_STYLE = 'seaborn-bright'
DPI = 300

target_path = Path('./binary_images/M1_test1/M1_test1.png')
ls_path = target_path.parent / 'M1_test1.npy'
target_ls = np.load(str(ls_path))[0]
ls_img_path = target_path.parent / 'M1_test1_LS.pdf'


fig = plt.figure()  #定义新的三维坐标轴
ax3 = plt.axes(projection='3d')


target_ls = target_ls[LITHOSIM_OFFSET:MASK_TILE_END_Y, LITHOSIM_OFFSET:MASK_TILE_END_X]
print(target_ls.shape)

#定义三维数据
xx = np.arange(0, 1280, 1)
yy = np.arange(0, 1280, 1)
X, Y = np.meshgrid(xx, yy)
Z = target_ls

xy0 = np.where(target_ls == 0)
x0 = xy0[0]
y0 = xy0[1]
# x0


#作图
ax3.scatter3D(y0, x0, 0, c='r', s=2)
ax3.plot_surface(X,Y,Z, alpha=0.7, rstride = 200, cstride = 200, cmap='rainbow')

#ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
# plt.show()
plt.savefig(str(ls_img_path), dpi=DPI, bbox_inches='tight')



