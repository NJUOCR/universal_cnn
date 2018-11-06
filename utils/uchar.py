import numpy as np
import cv2 as cv
import utils.uimg as uimg


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
    # print(h, w)
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

    print(top, left, bottom, right)
    return top, left, bottom, right


# make size to [64, 64]
def to_size(img, height = 64, width = 64):
    top, left, bottom, right = get_bounds(img, 'black')
# 如果小于64,进行边缘补齐
    new_left = 0
    new_right = 0
    new_bottom = 0
    new_top = 0
    img_height = bottom - top
    img_width = right - left
    if img_height <= height:
        new_top = (height - img_height)//2
        new_bottom = height - img_height - new_top
    elif img_width <= width:
        new_left = (right - left)//2
        new_right = width - new_left - img_width
    # out_img = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=0)
    out_img = cv.copyMakeBorder(img, new_top, new_bottom, new_left, new_right, cv.BORDER_CONSTANT, value=0)
    # cv.imshow("src", out_img)
    # print(new_top, new_bottom, new_left, new_right)
    uimg.save("out_img4.jpg", out_img)
    return out_img

if __name__ == '__main__':
    img = uimg.read('/home/stone/PycharmProjects/universal_cnn/4.jpg')
    # get_bounds(img)
    to_size(img, 64, 64)