'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2021-04-15 20:48:22
LastEditTime: 2021-04-21 20:16:06
Contact: cgjhaha@qq.com
Description: parse the results from the txt and save the results to a sqlite db.
'''
import re
import pandas as pd
from pathlib import Path
from database import CPP_RESULT_DB
from pandas import Series,DataFrame

TOP_NUM = 3
# M1_test9 15 58292.0 66096.0 3.91
# pvb_0.15_vel_0.2_curv_True_cw_0.5

# def print_seq():
#     print("="*50)
#     print("\n"*3)


def dir2db(base_dirs, my_db):
    for base_dir in base_dirs:
        for t in base_dir.glob('pvb_*_vel_*'):
            # pvb_0.15_vel_0.2_curv_True_cw_0.5
            if not t.is_dir():
                continue
            t_name = t.name
            print(f'parsing {t_name}')

            ilt_weight = 1

            pat = 'pvb_(\d+(\.\d+)?)'
            g = re.search(pat, t_name)
            pvb_weight = g.group(1)

            pat = 'vel_(\d+(\.\d+)?)'
            g = re.search(pat, t_name)
            vel_weight = g.group(1)

            pat = "curv_(True|False)"
            g = re.search(pat, t_name)
            add_curv = g.group(1)

            pat = 'cw_(\d+(\.\d+)?)'
            g = re.search(pat, t_name)
            curv_weight = g.group(1)

            ilt_exp = 2

            for test in t.glob('test*'):
                # print(test)
                test_name = test.name
                pat = 'test(\d+(\.\d+)?)'
                g = re.search(pat, test_name)
                test_id = g.group(1)
                case_name = f'M1_test{test_id}'
                # pandas save table
                data_dict = {
                    'name': [],
                    'epoch': [],
                    'pv_band': [],
                    'epe': [],
                    'l2': [],
                    'img_path': [],
                    'c_score': [], # contest score = 5000*epe + 4 * pvband
                    's_score': [], # simple score = pvband + l2
                }
                for im in test.glob('img_nom_*'):
                    # img_nom_e3_pvb_48619_epe_58_l2_92332
                    im_name = im.name
                    img_path = str(im)
                    data_dict['img_path'].append(img_path)
                    data_dict['name'].append(test_name)

                    pat = 'e(\d+(\.\d+)?)'
                    g = re.search(pat, im_name)
                    epoch = g.group(1)
                    data_dict['epoch'].append(int(epoch))

                    pat = 'pvb_(\d+(\.\d+)?)'
                    g = re.search(pat, im_name)
                    pv_band = g.group(1)
                    data_dict['pv_band'].append(float(pv_band))

                    pat = 'epe_(\d+(\.\d+)?)'
                    g = re.search(pat, im_name)
                    epe = g.group(1)
                    data_dict['epe'].append(int(epe))

                    pat = 'l2_(\d+(\.\d+)?)'
                    g = re.search(pat, im_name)
                    l2 = g.group(1)
                    data_dict['l2'].append(float(l2))

                    data_dict['c_score'].append(5000*int(epe) + 4*float(pv_band))
                    data_dict['s_score'].append(float(pv_band) + float(l2))

                df = DataFrame(data_dict)
                # print(df.sort_values(by=['epoch']))
                epe_sorts = df.sort_values(by=['epe'])[:TOP_NUM]
                # print(epe_sorts)
                s_score_sorts = df.sort_values(by=['s_score'])[:TOP_NUM]
                c_score_sorts = df.sort_values(by=['c_score'])[:TOP_NUM]
                for s in [epe_sorts, s_score_sorts, c_score_sorts]:
                    for i in range(TOP_NUM):
                        epoch = s.iloc[i]['epoch']
                        pv_band = s.iloc[i]['pv_band']
                        epe = s.iloc[i]['epe']
                        L2 = s.iloc[i]['l2']
                        img_path = s.iloc[i]['img_path']
                        s_score = s.iloc[i]['s_score']
                        c_score = s.iloc[i]['c_score']
                        my_db.insert_record(
                            case_name,
                            ilt_weight,
                            pvb_weight,
                            vel_weight,
                            add_curv,
                            curv_weight,
                            ilt_exp,
                            epoch,
                            L2,
                            pv_band,
                            epe,
                            img_path,
                            s_score,
                            c_score
                        )



if __name__ == '__main__':
    my_cpp_db = CPP_RESULT_DB('./results/cppresult.db')

    # log: before 
    base_dirs = [
        Path('/home/guojin/data/datasets/cu-ilt-results/cpp_results_2021-04-18'),
        Path('/home/guojin/data/datasets/cu-ilt-results/cpp_results_2021-04-19')
    ]
    dir2db(base_dirs, my_cpp_db)
