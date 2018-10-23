import numpy as np


# 计算横向或者竖向每列非白像素的个数，image:传入的图像; direction：0->列，1->行
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
    img_matrix = img_matrix + 0
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
        kernel = np.ones((window_size,))/window_size
        for _ in range(times):
            sum_array = np.convolve(sum_array, kernel, mode='same')
        sum_array = sum_array // 1
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
