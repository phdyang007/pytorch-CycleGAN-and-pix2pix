'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2020-12-26 17:07:15
LastEditTime: 2021-04-09 17:00:56
Contact: cgjhaha@qq.com
Description: the unit tests for the case4 target image
'''

from src.models.const import *


'''
8     9
 0 1 2
 3 x 4
 5 6 7
10    11
'''
def test_ul_corner(x, y, type_corner, image, ls):
    x1 = x + 512
    y1 = y + 512
    px = (0, 0, y1, x1)
    px0 = (0, 0, y1 - 1, x1 - 1)
    px1 = (0, 0, y1 - 1, x1)
    px2 = (0, 0, y1 - 1, x1 + 1)
    px3 = (0, 0, y1, x1 - 1)
    px4 = (0, 0, y1, x1 + 1)
    px5 = (0, 0, y1 + 1, x1 - 1)
    px6 = (0, 0, y1 + 1, x1)
    px7 = (0, 0, y1 + 1, x1 + 1)
    px8 = (0, 0, y1 - 2, x1 - 2)
    px9 = (0, 0, y1 - 2, x1 + 2)
    px10 = (0, 0, y1 + 2, x1 - 2)
    px11 = (0, 0, y1 + 2, x1 + 2)

    assert image[px] == 1, 'the {} corner of ({},{}) should be {}'.format(type_corner, x, y, 1)
    assert image[px0] == 0, 'the {} corner of ({},{}) position 0 should be {}'.format(type_corner, x, y, 0)
    assert image[px7] == 1, 'the {} corner of ({},{}) position 0 should be {}'.format(type_corner, x, y, 1)
    assert ls[px] == 0, 'the {} corner levelset of ({},{}) should be {}'.format(type_corner, x, y, 0)
    assert ls[px7] == -1, 'the {} corner levelset of ({},{}) position7 should be {}'.format(type_corner, x, y, -1)
    print('test {} corner of ({},{}) pass all'.format(type_corner, x, y))


'''
8     9
 0 1 2
 3 x 4
 5 6 7
10    11
'''
def test_ll_corner(x, y, type_corner, image, ls):
    x1 = x + 512
    y1 = y + 512
    px = (0, 0, y1, x1)
    px0 = (0, 0, y1 - 1, x1 - 1)
    px1 = (0, 0, y1 - 1, x1)
    px2 = (0, 0, y1 - 1, x1 + 1)
    px3 = (0, 0, y1, x1 - 1)
    px4 = (0, 0, y1, x1 + 1)
    px5 = (0, 0, y1 + 1, x1 - 1)
    px6 = (0, 0, y1 + 1, x1)
    px7 = (0, 0, y1 + 1, x1 + 1)
    px8 = (0, 0, y1 - 2, x1 - 2)
    px9 = (0, 0, y1 - 2, x1 + 2)
    px10 = (0, 0, y1 + 2, x1 - 2)
    px11 = (0, 0, y1 + 2, x1 + 2)

    assert image[px] == 1, 'the {} corner of ({},{}) should be {}'.format(type_corner, x, y, 1)
    assert image[px0] == 0, 'the {} corner of ({},{}) px0 should be {}'.format(type_corner, x, y, 0)
    assert image[px7] == 1, 'the {} corner of ({},{}) px7 should be {}'.format(type_corner, x, y, 1)
    assert ls[px] == 0, 'the {} corner levelset of ({},{}) should be {}'.format(type_corner, x, y, 0)
    assert ls[px7] == -1, 'the {} corner levelset of ({},{}) position7 should be {}'.format(type_corner, x, y, -1)
    print('the {} corner ls[px0] is {}, which should be sqrt(2)'.format(type_corner, ls[px0]))
    print('the {} corner ls[px7] is {}, which should be -1'.format(type_corner, ls[px7]))
    print('check the output to see whether pass all test {} corner of ({},{}) pass all'.format(type_corner, x, y))
    print('\n========================\n')

'''
8      9
 0 1 2
 3 x 4
 5 6 7
10    11
'''
def test_ur_corner(x, y, type_corner, image, ls):
    x1 = x + 512
    y1 = y + 512
    px = (0, 0, y1, x1)
    px0 = (0, 0, y1 - 1, x1 - 1)
    px1 = (0, 0, y1 - 1, x1)
    px2 = (0, 0, y1 - 1, x1 + 1)
    px3 = (0, 0, y1, x1 - 1)
    px4 = (0, 0, y1, x1 + 1)
    px5 = (0, 0, y1 + 1, x1 - 1)
    px6 = (0, 0, y1 + 1, x1)
    px7 = (0, 0, y1 + 1, x1 + 1)
    px8 = (0, 0, y1 - 2, x1 - 2)
    px9 = (0, 0, y1 - 2, x1 + 2)
    px10 = (0, 0, y1 + 2, x1 - 2)
    px11 = (0, 0, y1 + 2, x1 + 2)

    assert image[px] == 0, 'the {} corner of ({},{}) should be {}'.format(type_corner, x, y, 0)
    assert image[px0] == 1, 'the {} corner of ({},{}) position 0 should be {}'.format(type_corner, x, y, 1)
    assert image[px7] == 0, 'the {} corner of ({},{}) position 7 should be {}'.format(type_corner, x, y, 0)
    assert ls[px0] == 0, 'the {} corner levelset of ({},{}) position 0 should be {}'.format(type_corner, x, y, 0)
    print('the {} corner ls[px] is {}, which should be sqrt(2)'.format(type_corner, ls[px]))
    print('the {} corner ls[px7] is {}, which should be sqrt(8)'.format(type_corner, ls[px7]))
    print('check the output to see whether pass all test {} corner of ({},{}) pass all.'.format(type_corner, x, y))
    print('\n========================\n')

'''
8      9
 0 1 2
 3 x 4
 5 6 7
10    11
'''
def test_lr_convex(x, y, type_convex, image, ls):
    x1 = x + 512
    y1 = y + 512
    px = (0, 0, y1, x1)
    px0 = (0, 0, y1 - 1, x1 - 1)
    px1 = (0, 0, y1 - 1, x1)
    px2 = (0, 0, y1 - 1, x1 + 1)
    px3 = (0, 0, y1, x1 - 1)
    px4 = (0, 0, y1, x1 + 1)
    px5 = (0, 0, y1 + 1, x1 - 1)
    px6 = (0, 0, y1 + 1, x1)
    px7 = (0, 0, y1 + 1, x1 + 1)
    px8 = (0, 0, y1 - 2, x1 - 2)
    px9 = (0, 0, y1 - 2, x1 + 2)
    px10 = (0, 0, y1 + 2, x1 - 2)
    px11 = (0, 0, y1 + 2, x1 + 2)

    assert image[px] == 1, 'the {} convex of ({},{}) should be {}'.format(type_convex, x, y, 1)
    assert image[px0] == 1, 'the {} convex of ({},{}) px0 should be {}'.format(type_convex, x, y, 1)
    assert image[px7] == 1, 'the {} convex of ({},{}) px7 should be {}'.format(type_convex, x, y, 1)
    assert image[px1] == 0, 'the {} convex of ({},{}) px1 should be {}'.format(type_convex, x, y, 0)
    assert ls[px] == 0, 'the {} convex levelset of ({},{}) should be {}'.format(type_convex, x, y, 0)
    assert ls[px1] == 1, 'the {} convex levelset of ({},{}) px1 should be {}'.format(type_convex, x, y, 1)
    assert ls[px7] == -1, 'the {} convex levelset of ({},{}) px7 should be {}'.format(type_convex, x, y, -1)
    assert ls[px9] == 2, 'the {} convex levelset of ({},{}) px7 should be {}'.format(type_convex, x, y, 2)
    print('check the output to see whether pass all test {} convex of ({},{}) pass all'.format(type_convex, x, y))
    print('\n========================\n')

'''
8      9
 0 1 2
 3 x 4
 5 6 7
10    11
'''
def test_ul_convex(x, y, type_convex, image, ls):
    x1 = x + 512
    y1 = y + 512
    px = (0, 0, y1, x1)
    px0 = (0, 0, y1 - 1, x1 - 1)
    px1 = (0, 0, y1 - 1, x1)
    px2 = (0, 0, y1 - 1, x1 + 1)
    px3 = (0, 0, y1, x1 - 1)
    px4 = (0, 0, y1, x1 + 1)
    px5 = (0, 0, y1 + 1, x1 - 1)
    px6 = (0, 0, y1 + 1, x1)
    px7 = (0, 0, y1 + 1, x1 + 1)
    px8 = (0, 0, y1 - 2, x1 - 2)
    px9 = (0, 0, y1 - 2, x1 + 2)
    px10 = (0, 0, y1 + 2, x1 - 2)
    px11 = (0, 0, y1 + 2, x1 + 2)

    assert image[px] == 1, 'the {} convex of ({},{}) should be {}'.format(type_convex, x, y, 1)
    assert image[px0] == 1, 'the {} convex of ({},{}) px0 should be {}'.format(type_convex, x, y, 1)
    assert image[px1] == 1, 'the {} convex of ({},{}) px1 should be {}'.format(type_convex, x, y, 1)
    assert image[px3] == 0, 'the {} convex of ({},{}) px3 should be {}'.format(type_convex, x, y, 0)
    assert image[px5] == 0, 'the {} convex of ({},{}) px5 should be {}'.format(type_convex, x, y, 0)
    assert image[px6] == 1, 'the {} convex of ({},{}) px6 should be {}'.format(type_convex, x, y, 1)
    assert image[px7] == 1, 'the {} convex of ({},{}) px7 should be {}'.format(type_convex, x, y, 1)
    assert ls[px] == 0, 'the {} convex levelset of ({},{}) should be {}'.format(type_convex, x, y, 0)
    # assert ls[px1] == 1, 'the {} convex levelset of ({},{}) px1 should be {}'.format(type_convex, x, y, 1)
    assert ls[px7] == -1, 'the {} convex levelset of ({},{}) px7 should be {}'.format(type_convex, x, y, -1)
    assert ls[px3] == 1, 'the {} convex levelset of ({},{}) px3 should be {}'.format(type_convex, x, y, 1)
    print('the {} convex ls[px1] is {}, which should be 0'.format(type_convex, ls[px1]))
    print('the {} convex ls[px8] is {}, which should be -1'.format(type_convex, ls[px8]))
    print('the {} convex ls[px9] is {}, which should be -sqrt(5)'.format(type_convex, ls[px9]))
    print('check the output to see whether pass all test {} convex of ({},{}) pass all'.format(type_convex, x, y))
    print('\n========================\n')

'''
x, y is the coord in glp
'''
def test_corner(x, y, type_corner, image, ls):
    if type_corner == 'ur':
        test_ur_corner(x, y, type_corner, image, ls)
    elif type_corner == 'll':
        test_ll_corner(x, y, type_corner, image, ls)
    else:
        raise 'not implmentation'

def test_convex(x, y, type_convex, image, ls):
    if type_convex == 'lr':
        test_lr_convex(x, y, type_convex, image, ls)
    elif type_convex == 'ul':
        test_ul_convex(x, y, type_convex, image, ls)
    else:
        raise 'not implmentation'



def test_outer(x, y, image, ls):
    type_convex = 'outer'
    x1 = x + 512
    y1 = y + 512
    px = (0, 0, y1, x1)
    px0 = (0, 0, y1 - 1, x1 - 1)
    px1 = (0, 0, y1 - 1, x1)
    px2 = (0, 0, y1 - 1, x1 + 1)
    px3 = (0, 0, y1, x1 - 1)
    px4 = (0, 0, y1, x1 + 1)
    px5 = (0, 0, y1 + 1, x1 - 1)
    px6 = (0, 0, y1 + 1, x1)
    px7 = (0, 0, y1 + 1, x1 + 1)
    px8 = (0, 0, y1 - 2, x1 - 2)
    px9 = (0, 0, y1 - 2, x1 + 2)
    px10 = (0, 0, y1 + 2, x1 - 2)
    px11 = (0, 0, y1 + 2, x1 + 2)

    assert image[px] == 0, 'the {} convex of ({},{}) should be {}'.format(type_convex, x, y, 0)
    assert image[px0] ==  0, 'the {} convex of ({},{}) px0 should be {}'.format(type_convex, x, y, 0)
    assert image[px1] ==  0, 'the {} convex of ({},{}) px1 should be {}'.format(type_convex, x, y, 0)
    assert image[px3] ==  0, 'the {} convex of ({},{}) px3 should be {}'.format(type_convex, x, y, 0)
    assert image[px5] ==  0, 'the {} convex of ({},{}) px5 should be {}'.format(type_convex, x, y, 0)
    assert image[px6] ==  0, 'the {} convex of ({},{}) px6 should be {}'.format(type_convex, x, y, 0)
    assert image[px7] ==  0, 'the {} convex of ({},{}) px7 should be {}'.format(type_convex, x, y, 0)
    assert ls[px] == UP_TRUNCATED_D, 'the {} convex levelset of ({},{}) should be {}'.format(type_convex, x, y, UP_TRUNCATED_D)
    # assert ls[px1] == 1, 'the {} convex levelset of ({},{}) px1 should be {}'.format(type_convex, x, y, 1)
    assert ls[px7] ==  UP_TRUNCATED_D, 'the {} convex levelset of ({},{}) px7 should be {}'.format(type_convex, x, y, UP_TRUNCATED_D)
    assert ls[px3] == UP_TRUNCATED_D, 'the {} convex levelset of ({},{}) px3 should be {}'.format(type_convex, x, y, UP_TRUNCATED_D)
    # print('the {} convex ls[px1] is {}, which should be 0'.format(type_convex, ls[px1]))
    # print('the {} convex ls[px8] is {}, which should be -1'.format(type_convex, ls[px8]))
    # print('the {} convex ls[px9] is {}, which should be -sqrt(5)'.format(type_convex, ls[px9]))
    print('check the output to see whether pass all test {} convex of ({},{}) pass all'.format(type_convex, x, y))
    print('\n========================\n')
