'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2021-04-15 20:48:22
LastEditTime: 2021-04-18 13:38:46
Contact: cgjhaha@qq.com
Description: parse the results from the txt and save the results to a sqlite db.
'''
import re
from pathlib import Path
from database import RESULT_DB




my_db = RESULT_DB('./result.db')
base_dir = Path('/home/guojin/projects/develset_opc/levelset_net/multiruns/iccad13_outputs/best_results')



# M1_test9 15 58292.0 66096.0 3.91


for t in base_dir.glob('*.txt'):
    # ilt_1.0_pvb_0.1_curv_True_cw_1_exp_2.txt
    t_name = t.stem

    pat = 'ilt_(\d+(\.\d+)?)'
    g = re.search(pat, t_name)
    ilt_weight = g.group(1)

    pat = 'pvb_(\d+(\.\d+)?)'
    g = re.search(pat, t_name)
    pvb_weight = g.group(1)

    pat = "curv_(True|False)"
    g = re.search(pat, t_name)
    add_curv = g.group(1)

    pat = 'cw_(\d+(\.\d+)?)'
    g = re.search(pat, t_name)
    curv_weight = g.group(1)

    pat = 'exp_(\d+(\.\d+)?)'
    g = re.search(pat, t_name)
    ilt_exp = g.group(1)

    with open(t, 'r') as lines:
        for line in lines:
            case_name, epoch, L2, pv_band, runtime = line.split(' ')
            # print(CASE_NBME, EPOCH, L2, PV_BAND )
            my_db.insert_record(
                case_name,
                ilt_weight,
                pvb_weight,
                add_curv,
                curv_weight,
                ilt_exp,
                epoch,
                L2,
                pv_band,
                runtime,
            )
    print(f'{t_name} processed.')