# Author: Bentian Jiang
# Copyright@CUHK
# Please do not discolse this code to others

import numpy as np
import cv2 as cv
import os
from fnmatch import fnmatch
import PolyUtil
import subprocess
from database import result_db

def show_img(img, title, wait_key=0, output_path=None):
    cv.imshow(title, img)
    cv.waitKey(wait_key)
    if output_path is not None:
        cv.imwrite(output_path, img)  
    return


def find_external_contours(gray_img):
    cnts, hier = cv.findContours(
        gray_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    return cnts, hier

def find_all_contours(gray_img, contour_approx=cv.CHAIN_APPROX_SIMPLE):
    # cnts, hier = cv.findContours(gray_img, cv.RETR_EXTERNAL, contour_approx)
    cnts, hier = cv.findContours(gray_img, cv.RETR_TREE, contour_approx)
    return cnts, hier

def report_all_polygons(gray_img, save_path=None):
    new_img_gray = np.zeros((2048, 2048), np.uint8)

    my_m_cnts,_ = find_all_contours(gray_img.copy())

    cv.fillPoly(new_img_gray, my_m_cnts, 255)
    print(save_path)

    app_cnts, _ = cv.findContours(new_img_gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    new_img_gray = np.zeros((2048, 2048), np.uint8)
    cv.fillPoly(new_img_gray, app_cnts, 2)
    new_img_gray = new_img_gray - np.ones((2048,2048))
    new_img_gray.astype(np.uint8)

    my_large_polys_img = enlarge_img(new_img_gray)
    print(np.max(my_large_polys_img), np.min(my_large_polys_img), "begin extraction.")
    polys = PolyUtil.bimg_to_poly_coord(my_large_polys_img)
    my_polys = []
    for poly in polys:
        check_polygon(poly)
        if filter_out_singleton_pixel(poly):
            my_polys.append(poly)
    #poly_save_path = save_path.split('.')[0] + '.txt
    poly_save_path = save_path.replace('.png','.txt')
    print(save_path)
    print('------------------------poly save path is-------------------------- ')
    print(poly_save_path)
    with open(poly_save_path, 'w') as f:
        for poly in my_polys:
            f.write("POLY")
            for p in poly:
                f.write(" %i %i" % (p[0], p[1]))
            f.write("\n")
    process = subprocess.Popen(['./mask_fracturing',poly_save_path])
    process.communicate()
    process.kill()
    process.terminate()
    return new_img_gray

def enlarge_img(img, ori=2048, mul=2):
    new_img_gray = np.zeros((ori * mul, ori * mul), np.int)
    for x in range(len(img)):
        for y in range(len(img[0])):
            new_img_gray[x * mul][y * mul] = img[x][y]
            new_img_gray[x * mul + 1][y * mul + 1] = img[x][y]
            new_img_gray[x * mul][y * mul + 1] = img[x][y]
            new_img_gray[x * mul + 1][y * mul] = img[x][y]
    return new_img_gray

def check_polygon(cnt):
    for x in range(len(cnt)):
        lp = cnt[x-1]
        p = cnt[x]
        if p[0] != lp[0] and p[1] != lp[1]:
            for ele in cnt:
                print(ele)
            print(x)
            raise NotImplementedError("Error: Not valid polygon")
    print("Valid polygon")
    return True

def filter_out_singleton_pixel(cnt):
    # special for gan-opc family results, since their mask files are floating pixels, binarized masks may contain many singleton pixels
    # which may significantly increase the usless fracturing shot counts (1 singleton pixel = 1 shot)
    for x in range(len(cnt)):
        p = cnt[x]
        lp = cnt[x-1]
        if len(cnt) <= 4 and (abs(p[0] - lp[0]) == 1 or abs(p[1] - lp[1]) == 1):
            return False
    return True

def manhattanize_target_masks(masks_root, output_dir):
    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir))
    for path, _, files in os.walk(masks_root):
        for name in files:
            if fnmatch(name, "*.jpg") or fnmatch(name, "*.png"):
                png_file = os.path.join(path, name)
                #print(png_file)
                design_name = name.split('.')[0]
                mask = cv.imread(png_file)
                maskgray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
                save_path = os.path.join(output_dir, design_name + '.png')
                #print(png_file, save_path)
                report_all_polygons(maskgray, save_path=save_path)

def get_minshots(filepath):
    dic = {}
    lines = open(filepath)
    for line in lines:
        case_name, min_shots = line.split(' ')
        dic[case_name] = min_shots
    return dic


# if __name__ == "__main__":
#     masks_root="/path/to/your/input/folder/"
#     output_dir="/path/to/your/output/folder/"
#     manhattanize_target_masks(masks_root, output_dir)


if __name__ == "__main__":
    my_db = result_db('./result.db')
    base_dir = '/home/hongduo/school/develset_opc/levelset_net/iccad13_outputs'
    ilt_weight = 1.0
    pvb_weight = 7.5
    add_curv = False
    base_name = 'ckpts_{}_{}_{}'.format(ilt_weight, pvb_weight, add_curv)
    work_dir = os.path.join(base_dir, base_name)
    masks_root = os.path.join(work_dir,'mask')
    manhattanize_target_masks(masks_root, masks_root)
    best_score_name = 'best_result_ilt_{}_pvb_{}_curv_{}.txt'.format(ilt_weight, pvb_weight, add_curv)
    best_score_path = os.path.join(work_dir, best_score_name)
    min_shots_path = os.path.join(masks_root, 'minShots.txt')
    dic = get_minshots(min_shots_path)
    lines = open(best_score_path, 'r')
    for line in lines:
        CASE_NBME, EPOCH, L2, PV_BAND = line.split(' ')
        my_db.insert_record(CASE_NBME, ilt_weight, pvb_weight, add_curv, EPOCH, L2, PV_BAND, dic[CASE_NBME])
    my_db.close()