import cv2 as cv
import numpy as np


def fix_orientation(img):
    """
    Text orientation correcting.
    :param img: a numpy array with shape (h, w) corresponding to a image, in which the text paragraph may be
                rotated by an angle.
    :return: the output image, in which the text is horizontally oriented.
    """
    # todo 文字倾斜校正
    gray = img if len(img.shape) == 2 else cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    binary_matrix = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 47, 10)
    padded_matrix = image_padded(binary_matrix)
    hough_lines = image_dft(padded_matrix)
    des_angle = calculate_angle(hough_lines, img)
    result_image = image_rotated(des_angle, img)
    return result_image


def image_padded(image):
    img_matrix = np.array(image)
    width = cv.getOptimalDFTSize(np.size(img_matrix, 1))
    height = cv.getOptimalDFTSize(np.size(img_matrix, 0))
    top_bottom = height - np.size(img_matrix, 0)
    left_right = width - np.size(img_matrix, 1)
    new_matrix = cv.copyMakeBorder(img_matrix, 0, top_bottom,
                                   0, left_right, borderType=cv.BORDER_CONSTANT, value=0)
    return new_matrix


def image_dft(image):
    temp_matrix = np.array(image)
    # 傅里叶变换
    forier_matrix = np.fft.fft2(temp_matrix)
    forier_matrix_shift = np.fft.fftshift(forier_matrix)
    forier_matrix_magnitude = np.log(np.abs(forier_matrix_shift))
    # 二值化
    forier_matrix_magnitude = forier_matrix_magnitude.astype(np.uint8)
    ret, threshold_matrix = cv.threshold(forier_matrix_magnitude, 14, 255, cv.THRESH_BINARY)  # 11这个阈值 可能需要根据情况变换
    # cv.imshow("wwq", threshold_matrix)
    # 霍夫直线变换
    lines = cv.HoughLinesP(threshold_matrix, 2, np.pi / 180, 80, minLineLength=0, maxLineGap=100)
    return lines


def calculate_angle(lines, image):
    height, width = image.shape[:2]
    angle = 0.0
    piThresh = np.pi / 90
    pi2 = np.pi / 2
    lenIndex = []
    thetaIndex = []
    # lineIndex = []
    if lines == None:
        return 0.0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        theta = abs(np.arctan2(y2 - y1, x2 - x1))  # 如果不是绝对值  有可能angle是负的  即 p2 比 p1 小
        lines_direction = np.arctan2(y2 - y1, x2 - x1)
        if abs(theta) < piThresh * 5.5 or abs(theta - pi2) < piThresh:
            continue
        else:
            lenIndex.append((x2 - x1) ** 2 + (y2 - y1) ** 2)
            thetaIndex.append(lines_direction)
            # lineIndex.append(line)
    thetaIndex = np.sort(thetaIndex)
    angle = thetaIndex[int(len(thetaIndex) / 2)]
    lines_direction = np.tan(angle)
    if angle >= pi2:
        angle = angle - np.pi
    if angle != pi2:
        anglet = height * np.tan(angle) / width
        angle = np.arctan(anglet)
    angle = angle * (180 / np.pi)
    angle = abs(angle)
    if lines_direction > 0:
        angle = angle - 90
    else:
        angle = 90 - angle
    return angle


def image_rotated(angle, image):
    height, width = image.shape[:2]
    img_matrix = np.array(image)
    center = (width // 2, height // 2)
    temp_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated_matrix = cv.warpAffine(img_matrix, temp_matrix, (width, height), flags=cv.INTER_CUBIC,
                                   borderMode=cv.BORDER_REPLICATE)
    return rotated_matrix


if __name__ == '__main__':
    img = cv.imread('E:/fuliye/imageText_02_R.jpg')
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # padded_matrix = image_padded(gray)
    # hough_lines = image_dft(padded_matrix)
    # des_angle = calculate_angle(hough_lines, img)
    # result_image = image_rotated(des_angle, img)
    # cv.imshow('current_image', img)
    # cv.imshow('des_image', result_image)
    rr = fix_orientation(img)
    cv.imshow("ss", rr)
    cv.waitKey(0)
    cv.destroyAllWindows()
