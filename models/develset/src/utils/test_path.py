'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2021-01-24 16:24:55
LastEditTime: 2021-01-24 19:34:03
Contact: cgjhaha@qq.com
Description: test the pathlib
'''
import shutil
from pathlib import Path

# bimage_path = Path('/home/guojin/projects/develset_opc/levelset_net/binary_images')

# d_path = Path('/home/guojin/data/datasets/iccad_2013/targets')
# d_path = Path('/home/guojin/data/datasets/iccad_2013/ls_params')
# for test_dir in sorted(bimage_path.iterdir()):
#     print(test_dir)
#     for png in test_dir.glob('*.pt'):
#         # print(png)
#         # png_path = test_dir / png
#         out_path = d_path / png.name
#         shutil.copy(str(png), str(out_path))



t_path = Path('/home/guojin/data/datasets/iccad_2013/targets')

for t in sorted(t_path.glob('*.png')):
    print(Path(t.stem).stem)


