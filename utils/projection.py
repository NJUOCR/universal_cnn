import cv2 as cv
import numpy as np
from utils.uimg import auto_bin, save, read, show, replace_color
import os

# 计算横向或者竖向每列非白像素的个数，image:传入的图像; direction：0->列，1->行
def calculate_pixel(image, direction):
    # todo 自适应二值化后简化操作
    # 灰度图，只有0,1
    img_matrix = np.array(image)
    # img_matrix = np.floor(img_matrix / 255)
    # 取反
    img_matrix = np.logical_not(img_matrix)
    img_matrix = img_matrix + 0
    pixel_sum = np.sum(img_matrix, axis=direction)

    return pixel_sum


# 根据传入的参数，生成固定大小的初始值为0的矩阵，
# dimension:拼接的矩阵的长或者宽维度;
# direction：0->列，1->行
def generate_matrix(dimension, direction):
    matrix = []
    # 竖值（right）
    if direction == 0:
        matrix = np.zeros((dimension, 100))  # 每行
    # 水平（left)
    elif direction == 1:
        matrix = np.zeros((100, dimension))  # 每列
    # 全部变成白色
    matrix = np.add(matrix, 255)
    return matrix


# 将空矩阵填充起来,
# image:传入的图像;
# pix_sum:每列或者每行的非白像素个数之和;
# direction：0->列，1->行
def fill_matrix(image, pix_sum, direction):
    img_matrix = np.array(image)
    # 统计相应方向的个数
    num = np.size(img_matrix, direction)
    # temp_matrix：白色背景图片生成
    temp_matrix = generate_matrix(num, direction)

    for i in range(100):
        pix_sum = np.append(pix_sum, 0)
    pix_sum_norm = np.floor(pix_sum / max(pix_sum) * 100)  # 归一化到100
    for i in range(num):  # 填充
        if pix_sum[i] != 0:
            for j in range(int(pix_sum_norm[i])):
                if direction == 1:
                    temp_matrix[99 - j, i] = 0
                elif direction == 0:
                    temp_matrix[i, 99 - j] = 0
    return temp_matrix


# 横向拼接矩阵
def horizontal_merge(image, row_pix):
    img_matrix = np.array(image)
    ve_matrix = fill_matrix(img_matrix, row_pix, 0)
    img_matrix = np.hstack((img_matrix, ve_matrix))
    return img_matrix


# 纵向拼接矩阵
def vertical_merge(image, col_pix):
    img_matrix = np.array(image)
    ho_matrix = fill_matrix(image, col_pix, 1)
    img_matrix = np.vstack((img_matrix, ho_matrix))
    return img_matrix


def project(img, direction='vertical'):
    """
    Do projection.
    :param img: A numpy array. source image.
    :param direction: `vertical` | `horizontal`
    :return: A numpy array with shape (1, )
    """
    # todo 自适应二值化
    return calculate_pixel(img, 1 if direction == 'horizontal' else 0)


def draw_projective_histogram(img, direction='both'):
    """
    1. Copy the input img array
    2. Do padding, on right, bottom or both, according to the `direction`
    3. Draw histogram

    > The original input image will not be changed.
    :param img: A numpy array, the source image.
    :param direction: `vertical` | `horizontal` | `both`(default)
    :return: A numpy array
    """
    temp_matrix = np.array(img)
    per_col = project(temp_matrix, 'vertical')
    per_row = project(temp_matrix, 'horizontal')
    if direction == 'vertical':
        return vertical_merge(temp_matrix, per_col)
    elif direction == 'horizontal':
        return horizontal_merge(temp_matrix, per_row)
    else:
        horizontal_matrix = horizontal_merge(temp_matrix, per_row)
        result_matrix = vertical_merge(horizontal_matrix, per_col)
        return result_matrix


if __name__ == '__main__':
    # 读取图片: cv2.imread(路径,num) 其中num=0，为灰度图像；num=1为彩图

    # 二值化
    # im = auto_bin(im)



    # 获得结果矩阵
    # 获得结果矩阵
    # for src in os.listdir("/home/stone/PycharmProjects/cnn_orientation_correcting/out"):
    # print(dst)
    #     img = cv.imread(os.path.join("/home/stone/PycharmProjects/cnn_orientation_correcting/out", src))
    # 计算每列的非白像素求和
    #     _per_col = calculate_pixel(img, 0)

    # 计算每行的非白像素求和
    idx = 0
    for root, _, files in os.walk('/usr/local/src/data/doc_imgs'):
        for file in files:
            img = read(os.path.join(root, file), read_flag=cv.IMREAD_COLOR)
            img_at_mean = auto_bin(img)
            _per_row = calculate_pixel(img_at_mean, 1)
            _result_matrix = draw_projective_histogram(img_at_mean)
            save(os.path.join('../out', '%d.jpg'%idx), _result_matrix)
            idx += 1

    # save('test3.jpg', _result_matrix)
    # _result_matrix = draw_projective_histogram(im)
    # cv2.imshow('result_image', _result_matrix)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()V4
