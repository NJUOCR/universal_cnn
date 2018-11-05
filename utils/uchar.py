import numpy as np


def get_bounds(img, foreground_color='black'):
    """
    Get the minimum rectangle box containing the text. This method assumes that:
        1. there is no noise in the given image
    :param foreground_color:
    :param img: input image, of which the pixel value should either be `0` or `255`
    :return: top, left, bottom, right
    > FYI: image_height = bottom - top, image_width = right - left
    """
    assert foreground_color in ('white', 'black')
    h, w = img.shape
    background_vertical = np.zeros((h,), np.uint8) if foreground_color == 'white' else np.ones((h,), np.uint8) * 255
    background_horizontal = np.zeros((w,), np.uint8) if foreground_color == 'white' else np.ones((w,), np.uint8) * 255
    for left in range(w):
        if not (img[:, left] == background_vertical).all():
            break
    else:
        left = None

    for right in range(w - 1, -1, -1):
        if not (img[:, right] == background_vertical).all():
            break
    else:
        right = None

    for top in range(h):
        if not (img[top, :] == background_horizontal).all():
            break
    else:
        top = None

    for bottom in range(h - 1, -1, -1):
        if not (img[bottom, :] == background_horizontal).all():
            break
    else:
        bottom = None

    return top, left, bottom, right


def to_size(img, height, width):
    pass
