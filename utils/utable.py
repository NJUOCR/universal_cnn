#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import  numpy as np
import cv2 as cv
from utils import uimg
from utils.orientation import fix_orientation
import os

def find_line2(img):
    # ori_img = fix_orientation(img)
    # 二值化，如果不是灰度图，转成灰度图片
    # img = uimg.auto_bin(img, otsu=True)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 阈值,小的阈值设置为150
    _, threshold = cv.threshold(gray_img, 150, 255, cv.THRESH_BINARY_INV)

    # 膨胀
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv.dilate(threshold, kernel, iterations=1)
    # 轮廓检测
    # contours:轮廓；hierarchy：等级制度
    image, contours, hierarchy = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # 找到表格最外层轮廓
    tree = hierarchy[0]
    max_size = 0
    outer_contour_index = None
    for i, contour in enumerate(contours):
        size = cv.contourArea(contour)
        if size > max_size:
            max_size = size
            outer_contour_index = i
    # 找出了面积最大所在的轮廓
    max_contour = contours[outer_contour_index]
    # out_img = cv.drawContours(img, max_contour, -1, (0, 0, 255), 20)
    # 用红线标出
    cv.drawContours(img, max_contour, -1, (0, 0, 255), 20)
    # 设定一个阈值，估算最小单元
    x, y, w, h = cv.boundingRect(contour)
    min_size = w * h * 0.015
    # 继续进一步找子轮廓
    inter_contour_index = tree[outer_contour_index][2]
    for j in range(0, len(contours)):
        contour = contours[inter_contour_index]
        if cv.contourArea(contour) > min_size:
            rect = cv.minAreaRect(contour)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            # 用绿线标出
            cv.drawContours(img, [box], 0, (0, 255, 0), 5)
        inter_contour_index = tree[inter_contour_index][0]
        if inter_contour_index < 0:
            return img

    return img

if __name__ == '__main__':

    img_path = "H:\\Data\\table_out\\table_data"
    out_path_1127 = "H:\\Data\\table_out\\out_1127_3"
    dst_path = os.listdir(img_path)
    for i, img in enumerate(dst_path):
        img = os.path.join("%s\%s" % (img_path, img))
        img = cv.imread(img, 1)
        out_img = find_line2(img)
        uimg.save(os.path.join("%s\%s.jpg" % (out_path_1127, i)), out_img)

