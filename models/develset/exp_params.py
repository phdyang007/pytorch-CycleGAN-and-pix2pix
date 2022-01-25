'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2021-08-21 13:26:16
LastEditTime: 2021-08-29 19:39:14
Contact: gjchen21@cse.cuhk.edu.hk
Description:
'''


CASE_SMALL = [1,2,5,6,7,8,9,10]
CASE_BIG = [3, 4]
CASE_ALL =  [1,2,3,4,5,6,7,8,9,10]


params = [
    {
        'num': 0,
        'num_kernel': [12, 15, 18, 24],
        'sigmoid': [25, 40, 50 , 60, 80],
        'pvb': [0.1, 0.15, 4, 7, 8.5],
        'vel': [0],
        'add_curv': [0, 1],
        'curv_coef': [0.1, 0.2, 0.5, 0.8, 0.9, 1],
        'case': [1]
    },
    {
        'num': 1,
        'num_kernel': [12, 15, 18, 24],
        'sigmoid': [25, 40, 50 , 60, 80],
        'pvb': [0.1, 0.15, 4, 7, 8.5],
        'vel': [0],
        'add_curv': [0, 1],
        'curv_coef': [0.1, 0.2, 0.5, 0.8, 0.9, 1],
        'case': [2]
    },
    {
        'num': 2,
        'num_kernel': [12, 15, 18, 24],
        'sigmoid': [25, 40, 50 , 60, 80],
        'pvb': [0.15, 4, 8.5],
        'vel': [0],
        'add_curv': [0, 1],
        'curv_coef': [0.1, 0.5, 0.8, 1],
        'case': [3]
    },
    {
        'num': 3,
        'num_kernel': [12, 15, 18, 24],
        'sigmoid': [25, 40, 50 , 60, 80],
        'pvb': [0.15, 4, 8.5],
        'vel': [0],
        'add_curv': [0, 1],
        'curv_coef': [0.1, 0.5, 0.8, 1],
        'case': [4]
    },
    {
        'num': 4,
        'num_kernel': [12, 15, 18, 24],
        'sigmoid': [25, 40, 50 , 60, 80],
        'pvb': [0.1, 0.15, 4, 7, 8.5],
        'vel': [0],
        'add_curv': [0, 1],
        'curv_coef': [0.1, 0.2, 0.5, 0.8, 0.9, 1],
        'case': [5]
    },
    {
        'num': 5,
        'num_kernel': [12, 15, 18, 24],
        'sigmoid': [25, 40, 50 , 60, 80],
        'pvb': [0.1, 0.15, 4, 7, 8.5],
        'vel': [0],
        'add_curv': [0, 1],
        'curv_coef': [0.1, 0.2, 0.5, 0.8, 0.9, 1],
        'case': [6]
    },
    {
        'num': 6,
        'num_kernel': [12, 15, 18, 24],
        'sigmoid': [25, 40, 50 , 60, 80],
        'pvb': [0.1, 0.15, 4, 7, 8.5],
        'vel': [0],
        'add_curv': [0, 1],
        'curv_coef': [0.1, 0.2, 0.5, 0.8, 0.9, 1],
        'case': [7]
    },
    {
        'num': 7,
        'num_kernel': [12, 15, 18, 24],
        'sigmoid': [25, 40, 50 , 60, 80],
        'pvb': [0.1, 0.15, 4, 7, 8.5],
        'vel': [0],
        'add_curv': [0, 1],
        'curv_coef': [0.1, 0.2, 0.5, 0.8, 0.9, 1],
        'case': [8]
    },
    {
        'num': 8,
        'num_kernel': [12, 15, 18, 24],
        'sigmoid': [25, 40, 50 , 60, 80],
        'pvb': [0.1, 0.15, 4, 7, 8.5],
        'vel': [0],
        'add_curv': [0, 1],
        'curv_coef': [0.1, 0.2, 0.5, 0.8, 0.9, 1],
        'case': [9]
    },
    {
        'num': 9,
        'num_kernel': [12, 15, 18, 24],
        'sigmoid': [25, 40, 50 , 60, 80],
        'pvb': [0.1, 0.15, 4, 7, 8.5],
        'vel': [0],
        'add_curv': [0, 1],
        'curv_coef': [0.1, 0.2, 0.5, 0.8, 0.9, 1],
        'case': [10]
    },
]

trained_params = []



def get_all_groups(trained_params):
    trained_groups = []
    for p in trained_params:
        for pvb in p['pvb']:
            for vel in p['vel']:
                for add_curv in p['add_curv']:
                    for curv_coef in p['curv_coef']:
                        for case in p['case']:
                            g = [pvb, vel, add_curv, curv_coef, case]
                            trained_groups.append(g)
    return trained_groups

def get_real_group_num():
    trained_groups = get_all_groups(trained_params)
    to_groups = get_all_groups(params)
    print(f'plan to train {len(to_groups)} groups')
    real_groups = []
    for p in params:
        for pvb in p['pvb']:
            for vel in p['vel']:
                for add_curv in p['add_curv']:
                    for curv_coef in p['curv_coef']:
                        for case in p['case']:
                            t = [pvb, vel, add_curv, curv_coef, case]
                            if t not in trained_groups:
                                real_groups.append(t)
    print(f'only need to train {len(real_groups)} groups')

if __name__ == '__main__':

    get_real_group_num()

