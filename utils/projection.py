import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def calculate_pixel(image, axis):
    """
    Calculate the sum of **black** pixel
    The input image should be binaried. ie. the value of each pixel should either be `n` or `0`,
    `n` is a non-negative constant.
    :param image: input image
    :param axis: calculate sum along this axis
    :return:
    """
    # 取反
    img_matrix = np.logical_not(image)
    img_matrix = img_matrix + 0.0
    # img_matrix = image // 255
    pixel_sum = np.sum(img_matrix, axis=axis)

    return pixel_sum


def project(img, direction='vertical', smooth=None):
    """
    Do projection.
    :param smooth: None or a tuple containing (`window_size`, `smooth_times`)
    :param img: A numpy array. source image.
    :param direction: `vertical` | `horizontal`
    :return: A numpy array with shape (1, )
    """
    assert direction in ('vertical', 'horizontal')
    assert smooth is None or len(smooth) == 2
    sum_array = calculate_pixel(img, 1 if direction == 'horizontal' else 0)
    if smooth is not None:
        window_size, times = smooth
        kernel = np.ones((window_size,)) / window_size
        for _ in range(times):
            sum_array = np.convolve(sum_array, kernel, mode='same')
        # sum_array = sum_array // 1
    return sum_array


def draw_projective_histogram(img, direction='both', histogram_height=100, histogram_background='white',
                              smooth=None):
    """
    1. Copy the input img array
    2. Do padding, on right, bottom or both, according to the `direction`
    3. Draw histogram

    > The original input image will not be changed.
    :param smooth: a tuple, `(window_size, smooth_times)`. If it is not `None`, several one-dimensional box blur
    will be performed on the `sum_array`, box shape is `(window_size, )`
    :param histogram_background:
    :param histogram_height:
    :param img: A numpy array, the source image.
    :param direction: `vertical` | `horizontal` | `both`(default)
    :return: A numpy array
    """
    background_color = 255 if histogram_background == 'white' else 0
    foreground_color = 255 - background_color

    def sum2histogram(sum_array, histogram_canvas):
        assert len(sum_array.shape) == 1, "a `sum_array` should have only one dimension"
        sum_array = sum_array // 1
        max_val = np.max(sum_array)
        for x, val in enumerate(sum_array):
            histogram_canvas[histogram_height - int(val / max_val * histogram_height):, x] = foreground_color

    img_height, img_width = img.shape
    container_height = img_height + (0 if direction == 'horizontal' else histogram_height)
    container_width = img_width + (0 if direction == 'vertical' else histogram_height)
    container = np.ones((container_height, container_width)) * background_color
    # copy image into container
    container[:img_height, :img_width] = img
    container = container.reshape((*container.shape, 1))

    if direction == 'both':
        vertical_sum = project(img, direction='vertical', smooth=smooth)
        sum2histogram(vertical_sum, container[img_height:, :img_width])
        horizontal_sum = project(img, direction='horizontal', smooth=smooth)
        sum2histogram(horizontal_sum, np.transpose(container[:img_height, img_width:], [1, 0, 2]))
    elif direction == 'vertical':
        vertical_sum = project(img, direction='vertical', smooth=smooth)
        sum2histogram(vertical_sum, container[img_height:, :img_width])
    elif direction == 'horizontal':
        horizontal_sum = project(img, direction='horizontal', smooth=smooth)
        sum2histogram(horizontal_sum, np.transpose(container[:img_height, img_width:], [1, 0, 2]))
    return container


def get_splitter_horizontal(sum_array):
    splitters = []
    for i in range(1, len(sum_array) - 21):
        left, cur, right = sum_array[i - 1:i + 2]
        cur_index = i
        if cur == left == right:
            continue
        if cur <= min(left, right):
            for j in range(i, i + 20):
                if (abs(sum_array[j] - cur)) >= 1:
                    cur_index = j - 1
                    break
            # if cur + 1.5 > left:
            #     cur_index = i - 1
            # if cur + 1.5 > right:
            #     cur_index = i + 1
            splitters.append(cur_index)
    return splitters


def get_splitter(sum_array):
    splitters = []
    for i in range(1, len(sum_array) - 1):
        left, cur, right = sum_array[i - 1:i + 2]
        if cur == left == right:
            continue
        if cur <= min(left, right):
            splitters.append(i)
    return splitters


def get_splitter_zero(sum_array):
    splitters = []
    for i in range(1, len(sum_array) - 3):
        left, cur, right, right1, right2 = sum_array[i - 1:i + 4]
        if left > 0 and cur == 0 and right == right1 == 0:
            splitters.append(i)
    return splitters


# 画直方图
def draw_images(sum_array):
    plt.plot(sum_array, range(sum_array.shape[0]))
    plt.gca().invert_yaxis()
    plt.show()


def extract_peek_array(array_vals, minimun_val=20, minimum_range=2):
    start_i = None
    end_i = None
    peek_ranges = []
    for i, val in enumerate(array_vals):
        if val > minimun_val and start_i is None:
            start_i = i
        elif val >= minimun_val and start_i is not None:
            pass
        elif val < minimun_val and start_i is not None:
            end_i = i
            if end_i - start_i >= minimum_range:
                peek_ranges.append((start_i, end_i))
            start_i = None
            end_i = None
        elif val <= minimun_val and start_i is None:
            pass
        else:
            raise ValueError("cannot pass this case")
    print(peek_ranges)
    return peek_ranges


def draw_line(sum_array, img):
    peek_ranges = extract_peek_array(sum_array)
    line_seg = np.copy(img)
    for i, peek_ranges in enumerate(peek_ranges):
        x = 0
        y = peek_ranges[0]
        w = line_seg.shape[1]
        h = peek_ranges[1] - y
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        cv.rectangle(line_seg, pt1, pt2, 255)
    return line_seg
    # cv.imshow('line image', line_seg)
    # cv.waitKey(0)
