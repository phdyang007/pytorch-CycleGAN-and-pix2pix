'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2021-04-02 10:37:53
LastEditTime: 2021-04-15 14:38:24
Contact: cgjhaha@qq.com
Description: find the trained list
'''
import shutil
from math import floor
from pathlib import Path


# trained_list = [
#     'M1_test0',
#     'M1_test10',
#     'M1_test1',
#     'M1_test2',
#     'M1_test3',
#     'M1_test4',
#     'M1_test5',
#     'M1_test6',
#     'M1_test7',
#     'M1_test8',
#     'M1_test9',
# ]


def find_trained_list(masks_path):
    trained_list = []
    for mask in sorted(masks_path.glob('*.png')):
        mask_name = mask.stem
        trained_list.append(mask_name)
    return trained_list

def find_untrained_list(masks_path: Path, all_path: Path):
    trained_list = find_trained_list(masks_path)

    untrained_list = []
    for mask in sorted(all_path.glob('*.png')):
        mask_name = mask.stem
        if mask_name not in trained_list:
            untrained_list.append(mask_name)
    return untrained_list


def partition(ls, size):
    """
    Returns a new list with elements size
    of which is a list of certain size.

        >>> partition([1, 2, 3, 4], 3)
        [[1, 2, 3], [4]]
    """
    return [ls[i:i+size] for i in range(0, len(ls), size)]


def split_train_data(mask_path: Path, train_dir: Path, split_num: int):
    all_target_path = train_dir / 'train' / 'targets'
    untrained = find_untrained_list(mask_path, all_target_path)
    un_num = len(untrained)
    size = floor(un_num / split_num)
    split_list = partition(untrained, size)
    return split_list


def create_split_folder(train_dir: Path, split_num: int):
    for i in range(split_num):
        train_f = train_dir / 'train_{}'.format(i)
        train_f.mkdir(exist_ok=True)
        targets_f = train_f / 'targets'
        targets_f.mkdir(exist_ok=True)
        ls_f = train_f / 'ls_params'
        ls_f.mkdir(exist_ok=True)



def split_copy(mask_path: Path, train_dir: Path, split_num: int):
    split_list = split_train_data(mask_path, train_dir, split_num)
    for i, l in enumerate(split_list):
        print('copy the {} folder'.format(i))
        for name in l:
            targets_f = train_dir / 'train_{}'.format(i) / 'targets'
            ls_f = train_dir / 'train_{}'.format(i) / 'ls_params'
            # target:
            s_t_n = train_dir / 'train' / 'targets' / '{}.png'.format(name)
            d_t_n = targets_f / '{}.png'.format(name)

            s_l_n = train_dir / 'train' / 'ls_params' / '{}.pt'.format(name)
            d_l_n = ls_f / '{}.pt'.format(name)
            shutil.copy(s_t_n, d_t_n)
            shutil.copy(s_l_n, d_l_n)


if __name__ == '__main__':
    train_dir = Path('/research/d2/xfyao/guojin/data/datasets/develset_ganopc_train')
    mask_path = Path('/research/d2/xfyao/guojin/develset_opc/levelset_net/l11ganopc_outputs/ckpts_2021-04-01_20-03-41/mask')
    split_num = 8
    create_split_folder(train_dir, split_num)
    split_copy(mask_path, train_dir, split_num)





