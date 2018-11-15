import utils.projection as proj
import utils.uchar as uchar
import numpy as np


class TextLine:

    def __init__(self, line_img):
        self.img = line_img
        self.char_splitters = []

    def get_char_splitters(self):
        # vertical_smooth = max(5, (lower - upper) // 6), 6
        vertical_smooth = (0, 0)
        vertical_sum_array = proj.project(self.img, direction='vertical', smooth=vertical_smooth)
        self.char_splitters = proj.get_splitter_zero(vertical_sum_array)
        # char_splitters = proj.get_splitter(vertical_sum_array)
        return self.char_splitters

    def get_char_obj_list(self, force=False):
        splitters = self.get_char_splitters() if len(self.char_splitters) == 0 or force else self.char_splitters
        return [TextChar(self.img[:, l:r]) for l, r in zip(splitters, splitters[1:])]


class TextChar:

    def __init__(self, char_img):
        self.img = char_img
        self.content_top = None
        self.content_bottom = None
        self.content_left = None
        self.content_right = None

    def box_content(self):
        """
        尝试使用矩形框出文字范围
        """
        self.content_top, self.content_left, self.content_bottom, self.content_right = uchar.get_bounds(self.img)

    def has_content(self):
        return None not in (self.content_top, self.content_left, self.content_bottom, self.content_right)

    def get_content_height(self):
        return self.content_bottom - self.content_top if self.has_content() else None

    def get_content_width(self):
        return self.content_right - self.content_left if self.has_content() else None

    def get_content_center(self):
        return int((self.content_bottom - self.content_top) // 2), int((self.content_right - self.content_left) // 2)

    def get_barycenter(self, direction: str = 'vertical', foreground_color: str = 'black')->float or None:
        """
        计算前景色像素的重心
        :param direction: 方向分量 'vertical | horizontal' # todo 'both'
        :param foreground_color: 文字的颜色 `black | white`
        :return: 以图片左上角为原点，返回重心的相对偏移，取值范围[0,1]
        """
        assert direction in ('vertical', 'horizontal')
        if not self.has_content():
            return None

        sum_array = proj.project(self.img, smooth=(0, 0),
                                 direction='vertical' if direction == 'horizontal' else 'horizontal')
        total_weight = np.sum(sum_array)
        total_leverage = 0
        for dst, weight in enumerate(sum_array):
            total_leverage += dst*weight
        absolute_barycenter = total_leverage / total_weight
        relative_barycenter = absolute_barycenter / sum_array.shape[0]
        return relative_barycenter

    def fit_resize(self, new_height, new_width):
        pass
