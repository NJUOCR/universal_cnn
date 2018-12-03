#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import numpy as np
import cv2 as cv
from utils import uimg
from utils.orientation import fix_orientation
import os
import time


# 表格线的识别
def find_line2(img):
    # 二值化，如果不是灰度图，转成灰度图片
    print(img)
    threshold = uimg.thresh_bin(img)
    # 膨胀
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv.dilate(threshold, kernel, iterations=1)
    # 轮廓检测
    image, contours, hierarchy = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # 找到表格最外层轮廓
    tree = hierarchy[0]
    # 找出了面积最大所在的轮廓; ctr:轮廓
    max_contour = max(contours, key=lambda ctr: cv.contourArea(ctr))
    # 判断文件中是否含有表格，设定一个单元面积cell_size,如果最大轮廓面积小于这个cell_size，认为该图片不含表格
    cell_size = 2
    if cv.contourArea(max_contour) < cell_size:
        return img
    # 找到最大轮廓所在的索引
    else:
        outer_contour_index = contours.index(max_contour)
        cv.drawContours(img, max_contour, -1, (255, 255, 255), 18)
        # 设定一个阈值，估算最小单元
        x, y, w, h = cv.boundingRect(max_contour)
        min_size = w * h * 0.0015
        # 继续进一步找子轮廓
        inter_contour_index = tree[outer_contour_index][2]
    # 轮廓不为空执行的情况
    if contours:
        for j in range(0, len(contours)):
            contour = contours[inter_contour_index]
            if cv.contourArea(contour) > min_size:
                rect = cv.minAreaRect(contour)
                box = cv.boxPoints(rect)
                box = np.int0(box)
                cv.drawContours(img, [box], 0, (255, 255, 255), 5)
            inter_contour_index = tree[inter_contour_index][0]
            if inter_contour_index < 0:
                return img
    else:
        print("contours is none")
    return img


def hou(img):
    # 转为二值化图
    threshold = uimg.thresh_bin(img)
    # gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # edge_img = cv.Canny(gray_img, 50, 200)
    # 膨胀
    # kernel = np.ones((3, 3), np.uint8)
    # dilated = cv.dilate(threshold, kernel, iterations=1)
    lines = cv.HoughLines(threshold, 1, np.pi/180, 300)
    result = img.copy()
    for line in lines:
        rho, theta = line.reshape(2)
        print(rho, theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = rho * a
        y0 = rho * b
        x1 = int(x0 + 1000 *(-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 *(-b))
        y2 = int(y0 - 1000 * a)
        cv.line(result, (x1, x2), (y1, y2), (0, 0, 255), 5)
    return result

# 累计概率霍夫变换,去掉膨胀，加上之后，效果不好见文件夹out_houP_dilate
# canny检测效果也不好，见文件夹out_houP_canny
def houP(img):
    # 转为二值化图
    threshold = uimg.thresh_bin(img)
    lines = cv.HoughLinesP(threshold, 0.01, np.pi / 180, 150, minLineLength=90, maxLineGap=5)
    result = img.copy()
    _lines = lines[:, 0, :]  # 提取为二维
    for x1, y1, x2, y2 in _lines[:]:
        cv.line(result, (x1, y1), (x2, y2), (255, 255, 255), 3)
    return result

if __name__ == '__main__':
    start = time.clock()
    input_path = "H:\\Data\\table_out\\table_data"
    hou_path = "H:\\Data\\2014_0100_208"
    out_hou_1129 = "H:\\Data\\table_out\\2014_0110_208_hou"
    dst_hou_path = os.listdir(hou_path)
    for i, img in enumerate(dst_hou_path):
        img_path = os.path.join("%s\%s" % (hou_path, img))
        print(img_path)
        ori_img = cv.imread(img_path, 1)
        # cour_img = find_line2(ori_img)
        out_img = houP(ori_img)
        uimg.save(os.path.join("%s\%s.jpg" % (out_hou_1129, i)), out_img)
    end = time.clock()
    print("total_time:", end - start)
