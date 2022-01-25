# Author: Xiaopeng Zhang
# Copyright@CUHK
# Please do not discolse this code to others

import numpy as np
# import TimeUtil
# import ImageUtil
from scipy.spatial import distance



class Corner:
    def __init__(self, idx, x, y):
        self.id = idx  # The id in the list<corners>
        self.x = x
        self.y = y
        self.val = 1  # The corners are all white points.

        self.isUsed = False  # Describe whether this corner is used by a polygon.
        self.x_pre_id = -1  # In horizontal (x) direction, the id of the previous corner (-1 represents there are not such a previous corner).
        self.x_next_id = -1
        self.y_pre_id = -1
        self.y_next_id = -1

        self.x_pre_num = 0  # In horizontal (x) direction, the number of the previous corners.
        self.y_pre_num = 0



def coverBy0(arr):
    # print('Info: arr.shape = ', arr.shape)
    [M, N] = arr.shape

    arr0 = np.zeros((M + 2, N + 2))
    arr0 = arr0 - 1

    for m in range(M):
        for n in range(N):
            if arr[m][n] != 1 and arr[m][n] != -1:
                print('Error: arr[m][n] != 1 and arr[m][n] != -1.  arr[m][n] = ', arr[m][n])
            arr0[m+1][n+1] = arr[m][n]

    return arr0


def isLineEnd(arr0, x0, y0):
    if arr0[x0][y0] != 1:  # Only white points can be line end.
        return False
    white_n_cnt = 0
    if arr0[x0 + 1][y0] == 1:
        white_n_cnt += 1
    if arr0[x0][y0 + 1] == 1:
        white_n_cnt += 1
    if arr0[x0 - 1][y0] == 1:
        white_n_cnt += 1
    if arr0[x0][y0 - 1] == 1:
        white_n_cnt += 1
    if white_n_cnt <= 1:
        # print('Warning: white_n_cnt = ', white_n_cnt)
        return True
    return False

def isCorner(arr0, x0, y0):
    if arr0[x0][y0] != 1:  # Only white points can be corners.
        return False
    # isLineEnd(arr0, x0, y0)

    if (arr0[x0 - 1][y0] * arr0[x0 + 1][y0] < 0) and (arr0[x0][y0 - 1] * arr0[x0][y0 + 1]) < 0:  # for exterior corner
        return True

    # for interior corner:
    i_range = [-1, 1]
    for xi in i_range:
        for yi in i_range:
            if arr0[x0 + xi][y0 + yi] == -1 and arr0[x0 + xi][y0] == 1 and arr0[x0][y0 + yi] == 1:
                return True

    return False


'''
    Find corners.
    Return: 0 -> not corner;
            1 -> a normal corner (interior or exterior)
            2 -> a overlap corner (in fact, it's 2 corners in 1 pixel)
'''
def isCorner_overlap(arr0, x0, y0):
    if arr0[x0][y0] != 1:  # Only white points can be corners.
        return 0
    # isLineEnd(arr0, x0, y0)

    if (arr0[x0 - 1][y0] * arr0[x0 + 1][y0] < 0) and (arr0[x0][y0 - 1] * arr0[x0][y0 + 1]) < 0:  # for exterior corner
        return 1

    # if arr0[x0-1][y0]==1 and arr0[x0+1][y0]==1 and arr0[x0][y0+1]==1 and arr0[x0][y0-1]==1 \

    # for interior corner:
    i_range = [-1, 1]
    for xi in i_range:
        for yi in i_range:
            if arr0[x0 + xi][y0 + yi] == -1 and arr0[x0 + xi][y0] == 1 and arr0[x0][y0 + yi] == 1:
                if arr0[x0 - xi][y0 - yi] == -1:
                    print('Warning: overlap corner.')
                    return 2
                return 1

    return 0


def isCorner_exceptLineEnd(arr0, x0, y0):
    if arr0[x0][y0] != 1:  # Only white points can be corners.
        return False
    if isLineEnd(arr0, x0, y0):
        return False

    # for exterior corner:
    if (arr0[x0 - 1][y0] * arr0[x0 + 1][y0] < 0) and (arr0[x0][y0 - 1] * arr0[x0][y0 + 1]) < 0:
        return True

    # another exterior corner:
    diag_no_cnt = 0
    i_range = [-1, 1]
    for xi in i_range:
        for yi in i_range:
            if arr0[x0 + xi][y0 + yi] == -1:
                diag_no_cnt += 1
    adj_yes_cnt = 0
    for xi in i_range:
        if arr0[x0 + xi][y0] == 1:
            adj_yes_cnt += 1
    for yi in i_range:
        if arr0[x0][y0 + yi] == 1:
            adj_yes_cnt += 1
    if diag_no_cnt == 3 and adj_yes_cnt >= 3:
        return True


    # for interior corner:
    i_range = [-1, 1]
    for xi in i_range:
        for yi in i_range:
            if arr0[x0 + xi][y0 + yi] == -1 and arr0[x0 + xi][y0] == 1 and arr0[x0][y0 + yi] == 1 and arr0[x0 + xi][y0 - yi] == 1 and arr0[x0 - xi][y0 + yi] == 1:
                return True

    return False


'''
    Input the image array, and then return the list<Corner>
    # Default: white is 1, while black is -1.
'''
def find_corners(img):
    corners = []
    # print('Info: img.shape = ', img.shape)
    [height, width] = img.shape

    arr0 = coverBy0(img)
    # [h0, w0] = arr0.shape

    for x in range(height):
        for y in range(width):
            x0 = x + 1
            y0 = y + 1
            # if isCorner(arr0, x0, y0):
            # # if isCorner_exceptLineEnd(arr0, x0, y0):
            #     corner = Corner(len(corners), x, y)
            #     corners.append(corner)
            for i in range(isCorner_overlap(arr0, x0, y0)):
                corner = Corner(len(corners), x, y)
                corners.append(corner)

    return corners


def cal_relation(corners):
    x_dict = dict()
    y_dict = dict()
    for cid in range(len(corners)):
        corner = corners[cid]
        if cid != corner.id:
            print('Error: cid != corner.id')

        if not x_dict.__contains__(corner.x):
            x_dict[corner.x] = []
        x_dict[corner.x].append(corner.id)

        if not y_dict.__contains__(corner.y):
            y_dict[corner.y] = []
        y_dict[corner.y].append(corner.id)


    for key, x_cid_list in x_dict.items():
        if len(x_cid_list) % 2 != 0:
            print('Error: len(x_cid_list) = ', len(x_cid_list))
        for index in range(len(x_cid_list)):
            cid = x_cid_list[index]
            corner = corners[cid]
            corner.x_pre_num = index
            if index > 0:
                corner.x_pre_id = x_cid_list[index - 1]
            if index < len(x_cid_list) - 1:
                corner.x_next_id = x_cid_list[index + 1]

    for key, y_cid_list in y_dict.items():
        if len(y_cid_list) % 2 != 0:
            print('Error: len(y_cid_list) = ', len(y_cid_list))
        for index in range(len(y_cid_list)):
            cid = y_cid_list[index]
            corner = corners[cid]
            corner.y_pre_num = index
            if index > 0:
                corner.y_pre_id = y_cid_list[index - 1]
            if index < len(y_cid_list) - 1:
                corner.y_next_id = y_cid_list[index + 1]



img_to_poly_coord_errCnt = 0

'''
    Input the image array (including 1 and -1), and then return the list<poly<coordinates>>.
    Suppose there are not such a polygon as a line.
    The 'record_path' is only used for debugging.
'''
def bimg_to_poly_coord(bimg, record_path=''):
    corners = find_corners(bimg)

    # img = ImageUtil.bimg_to_img(bimg)
    # rbg_path = record_path + '.corner.png'
    # ImageUtil.highlight_corners(img, corners, rbg_path)


    # print(TimeUtil.get_time_str() + 'Info: find_corners.')
    cal_relation(corners)
    # print(TimeUtil.get_time_str() + 'Info: cal_relation.')

    polys = []

    for start_corner in corners:

        if start_corner.isUsed == False:
            poly = []
            start_id = start_corner.id
            next_direction = 'x'
            # start_corner.isUsed = True
            isEnd = False
            corner = start_corner  # the current corner

            while(not isEnd):
                poly.append([corner.x, corner.y])
                corner.isUsed = True

                if next_direction == 'x':
                    next_direction = 'y'
                    if corner.x_pre_num % 2 == 1:  # look for the next corner in the left side
                        next_id = corner.x_pre_id
                    else:
                        next_id = corner.x_next_id
                else:
                    next_direction = 'x'
                    if corner.y_pre_num % 2 == 1:  # look for the next corner in the top side
                        next_id = corner.y_pre_id
                    else:
                        next_id = corner.y_next_id

                if next_id == start_id:
                    isEnd = True
                    continue

                # print('Warning: type(corners) = ', type(corners))
                next_corner = corners[next_id]
                if next_corner.isUsed == True:
                    global img_to_poly_coord_errCnt
                    img_to_poly_coord_errCnt += 1
                    print('Error: next_corner.isUsed == True. Error cnt = ', img_to_poly_coord_errCnt)
                    print('record_path = ', record_path)
                    return polys
                next_corner.isUsed = True

                corner = next_corner

            polys.append(poly)
    # print(TimeUtil.get_time_str() + 'Info: other img_to_poly_coord.')
    return polys



'''
    Transfer the type of coordinates: list<list> -> list<contour>
'''
def coord_list_to_contour(coord_list):
    contour_list = []
    for coords in coord_list:
        coord_num = len(coords)

        contour = np.zeros([coord_num, 1, 2], dtype=np.int32)
        for i in range(coord_num):
            contour[i][0][0] = coords[i][1]
            contour[i][0][1] = coords[i][0]

        contour_list.append(contour)
    return contour_list





'''
    Check whether the three coordinates are small triple or include a small pair.
    
    Return: 0 -> no small triple or small pair
            2 -> small pair (idx1 and idx2)
            3 -> small triple
'''
def check_small_triple(coords, idx1, idx2, idx3, small_thre):
    d1 = distance.cityblock(coords[idx1], coords[idx2])
    # print('coords[idx1] = ', coords[idx1])
    # print('coords[idx2] = ', coords[idx2])
    # print('d1 = ', d1)
    if d1 == 0:
        return 2
    d2 = distance.cityblock(coords[idx2], coords[idx3])
    if d1 < small_thre and d2 < small_thre:
        return 3
    return 0


'''
    (0, 0), (1, 0), (1, 2) -> (0, 2)
'''
def get_substitute(coords, idx1, idx2, idx3):
    c1 = coords[idx1]
    c2 = coords[idx2]
    c3 = coords[idx3]

    if c1[0] == c2[0] and c2[1] == c3[1]:
        return [c3[0], c1[1]]
    elif c1[1] == c2[1] and c2[0] == c3[0]:
        return [c1[0], c3[1]]
    else:
        print('Error: the 3 coordinates are invalid.')


def legal_1_coord(poly, small_thre):
    changed = False

    coords = poly.copy()
    coord_ori_num = len(coords)
    if coord_ori_num <= 4:
        return poly


    # print('Info: coord_ori_num = ', coord_ori_num)
    # print(poly)
    coords.append(coords[0].copy())
    coords.append(coords[1].copy())

    for i in range(coord_ori_num):
        check_res = check_small_triple(coords, i, i + 1, i + 2, small_thre)
        if check_res == 2:  # 2 -> small pair (idx1 and idx2)
            changed = True
            if i < coord_ori_num - 1:  # not the last one
                del poly[i:(i+2)]
            else:
                del poly[i]
                del poly[0]

        elif check_res == 3:  # small triple
            changed = True
            new_coord = get_substitute(coords, i, i + 1, i + 2)
            if i < coord_ori_num - 2:
                poly = poly[0:i] + [new_coord] + poly[(i+3):]
            elif i == coord_ori_num - 2:
                poly = poly[1:i] + [new_coord]
            elif i == coord_ori_num - 1:
                poly = poly[2:i] + [new_coord]
            else:
                print('Error: invalid index.')

        if changed:
            return legal_1_coord(poly, small_thre)

    if not changed:
        return poly



def legal_coords(polys, small_thre):
    result = []
    for poly in polys:
        # print('Warning: poly = ', poly)
        result.append(legal_1_coord(poly, small_thre))
    return result



# def repair_small_corner(arr, polys, small_thre):
#     coords_3_vec = []
#     bvnbm = arr.copy()
#
#     for i_poly in range(len(polys)):
#         coords = polys[i_poly].copy()
#         coord_num = len(coords)
#
#         # find all 3-points that are very close to each other
#         coords.append(coords[0].copy())
#         coords.append(coords[1].copy())
#         dists = []
#         for i in range(len(coords)):
#             if i < len(coords) - 1:
#                 dists.append(distance.cityblock(coords[i], coords[i + 1]))
#
#         for i in range(len(dists)):
#             if i < len(dists) - 1:
#                 if dists[i] <= small_thre and dists[i + 1] <= small_thre:
#                     points_3 = []
#                     points_3.append(coords[i])
#                     points_3.append(coords[i + 1])
#                     points_3.append(coords[i + 2])
#                     coords_3_vec.append(points_3)



def isRayIntersectsSegment(poi, s_poi, e_poi):  # [x,y] [lng,lat]
    # 输入：判断点，边起点，边终点，都是[lng,lat]格式数组
    if s_poi[1] == e_poi[1]:  # 排除与射线平行、重合，线段首尾端点重合的情况
        return False
    if s_poi[1] > poi[1] and e_poi[1] > poi[1]:  # 线段在射线上边
        return False
    if s_poi[1] < poi[1] and e_poi[1] < poi[1]:  # 线段在射线下边
        return False
    if s_poi[1] == poi[1] and e_poi[1] > poi[1]:  # 交点为下端点，对应spoint
        return False
    if e_poi[1] == poi[1] and s_poi[1] > poi[1]:  # 交点为下端点，对应epoint
        return False
    # if s_poi[0] < poi[0] and e_poi[1] < poi[1]:  # 线段在射线左边
    if s_poi[0] < poi[0] and e_poi[0] < poi[0]:
        return False

    xseg = e_poi[0] - (e_poi[0] - s_poi[0]) * (e_poi[1] - poi[1]) / (e_poi[1] - s_poi[1])  # 求交
    if xseg < poi[0]:  # 交点在射线起点的左侧
        return False
    return True  # 排除上述情况之后


def isPoiWithinPoly(poi, poly):
    # 输入：点，多边形三维数组
    # poly=[[[x1,y1],[x2,y2],……,[xn,yn],[x1,y1]],[[w1,t1],……[wk,tk]]] 三维数组

    # 可以先判断点是否在外包矩形内
    # if not isPoiWithinBox(poi,mbr=[[0,0],[180,90]]): return False
    # 但算最小外包矩形本身需要循环边，会造成开销，本处略去
    sinsc = 0  # 交点个数
    for epoly in poly:  # 循环每条边的曲线->each polygon 是二维数组[[x1,y1],…[xn,yn]]
        for i in range(len(epoly) - 1):  # [0,len-1]
            s_poi = epoly[i]
            e_poi = epoly[i + 1]
            if isRayIntersectsSegment(poi, s_poi, e_poi):
                sinsc += 1  # 有交点就加1

    return True if sinsc % 2 == 1 else False


'''
    Check if the point 'poi' is in a polygon 'poly'.
    poi: [x, y]
    poly: list<[x, y]> (This polygon is a simple polygon which doesn't include holes.)
'''
def isPoiInSimPoly(poi, poly):
    sinsc = 0  # 交点个数
    for i in range(len(poly) - 1):  # [0,len-1]
        s_poi = poly[i]
        e_poi = poly[i + 1]
        if isRayIntersectsSegment(poi, s_poi, e_poi):
            sinsc += 1  # 有交点就加1

    return True if sinsc % 2 == 1 else False


'''
    Check if the 'poly_in' is in the 'poly_out'.
    poly_out: list<[x, y]>
    poly_in: list<[x, y]>
'''
def isPolyInPoly(poly_out, poly_in):
    for coord in poly_in:
        if not isPoiInSimPoly(coord, poly_out):
            return False
    return True


'''
    Given that the 'hole' is in the 'poly', then concat the 'poly' and the 'hole'.
    Return the concated polygon.
'''
def concat_poly_hole(poly, hole):
    poly = poly.copy()
    hole = hole.copy()
    if poly[0] != poly[len(poly)-1]:
        poly.append(poly[0].copy())
    if hole[0] != hole[len(hole)-1]:
        hole.append(hole[0].copy())
    for i in range(len(hole)):
        idx = len(hole) - i - 1
        poly.append(hole[idx])
    return poly


'''
    Given a list of polygons 'polys', find holes in them, and concat the holes and the related polygons.
    Return the concated list of polygons.
'''
def concat_polys_holes(polys):
    poly_num = len(polys)
    rm_idx = []
    new_polys = []
    for i in range(poly_num):
        if i not in rm_idx:
            for j in range(poly_num):
                if j > i and (j not in rm_idx):
                    if isPolyInPoly(polys[i], polys[j]): # if polys[i] includes polys[j]
                        # new_polys.append(concat_poly_hole(polys[i], polys[j]))
                        # rm_idx.append(i)
                        # rm_idx.append(j)
                        polys[i] = concat_poly_hole(polys[i], polys[j])
                        rm_idx.append(j)
                        # print(polys[i])
                        # print(polys[j])
                        # print(concat_poly_hole(polys[i], polys[j]))
                    elif isPolyInPoly(polys[j], polys[i]):
                        # new_polys.append(concat_poly_hole(polys[j], polys[i]))
                        # rm_idx.append(i)
                        # rm_idx.append(j)
                        polys[j] = concat_poly_hole(polys[j], polys[i])
                        rm_idx.append(i)
                        # print(polys[i])
                        # print(polys[j])
                        # print(concat_poly_hole(polys[i], polys[j]))

    for i in range(poly_num):
        if i not in rm_idx:
            new_polys.append(polys[i])
    return new_polys












