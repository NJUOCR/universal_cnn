import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def save(file_path: str, img):
    """
    Do not use `cv2.imwrite` to save image, doesn't work as expected when it comes to Chinese file name
    :param file_path:
    :param img:
    :return:
    """
    cv.imencode('.jpg', img)[1].tofile(file_path)


def read(file_path: str, read_flag=0):
    return cv.imdecode(np.fromfile(file_path, dtype=np.uint8), read_flag)


def show(img):
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


def reverse(img):
    """
    :param img: channels = 1, dtype = numpy.uint8
    :return: val = 255 - val, pixel-wise
    """
    img = 255 - img
    return img


def auto_bin(img):
    """
    # todo ylx:自适应二值化
    :param img:
    :return:
    """
    # 读取图像，并转为灰度图
    img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 自适应二值化
    img_at_mean = cv.adaptiveThreshold(img_grey, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 10)


    return img_at_mean


def replace_color(img, bgr, old_val_bound, new_val):
    """
    # todo ylx:替换颜色
    :param img:
    :param old_color:
    :param new_color:
    :return:
    """
    height, width = img.shape[:2]
    # 转换HSV
    # hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([0, 0, 20])
    upper = np.array([100, 100, 255])
    mask_img = cv.inRange(img, lower, upper)

    # 腐蚀膨胀
    erode_img = cv.erode(mask_img, None, iterations=1)
    dilate_img = cv.dilate(erode_img, None, iterations=1)

    # 遍历替换
    for i in range(height):
        for j in range(width):
            if dilate_img[i, j] == 255:
                img[i, j] = (255, 255, 255)
    save("dst_change_color.jpg", img)
    return img


