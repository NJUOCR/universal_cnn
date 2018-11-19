import re
from typing import List

import cv2 as cv
import numpy as np

import utils.projection as proj
import utils.uchar as uchar
import utils.uimg as uimg
from utils.orientation import fix_orientation

# 汉字，不包含汉字的标点符号
ptn = re.compile('[\u4e00-\u9fa5]')


def is_chinese(c):
    return ptn.match(c) and True


class TextChar:

    def __init__(self, char_img, drawing_copy=None):
        self.img = char_img
        self.drawing_copy = drawing_copy
        self.content_top = None
        self.content_bottom = None
        self.content_left = None
        self.content_right = None

        self.is_half = None
        self.c, self.p = None, 0

        self.__valid = False

        self.__box_content()

    def __box_content(self):
        """
        尝试使用矩形框出文字范围
        """
        self.content_top, self.content_left, self.content_bottom, self.content_right = uchar.get_bounds(
            self.img[1:-1, 1:-1])

    def has_content(self):
        return None not in (self.content_top, self.content_left, self.content_bottom, self.content_right)

    def get_content_height(self):
        return self.content_bottom - self.content_top if self.has_content() else None

    def get_content_width(self):
        return self.content_right - self.content_left if self.has_content() else None

    def get_content_center(self):
        if not self.has_content():
            return None
        center_y, center_x = int((self.content_bottom + self.content_top) // 2), int(
            (self.content_right + self.content_left) // 2)
        if self.drawing_copy is not None:
            cv.rectangle(self.drawing_copy, (center_x - 1, center_y - 1), (center_x + 1, center_y + 1), 180,
                         thickness=3)
        return center_y, center_x

    def get_barycenter(self, direction: str = 'vertical', foreground_color: str = 'black') -> float or None:
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
            total_leverage += dst * weight
        absolute_barycenter = total_leverage / total_weight
        relative_barycenter = absolute_barycenter / sum_array.shape[0]
        return relative_barycenter

    def get_width(self):
        return self.img.shape[1]

    def get_height(self):
        return self.img.shape[0]

    def fit_resize(self, new_height, new_width):
        return uchar.to_size(self.img, new_height, new_width)

    def half(self, set_to=None):
        if set_to in (True, False):
            self.is_half = set_to
        return self.is_half

    def set_result(self, c, p=0):
        self.c = c
        self.p = round(p, 2)
        self.valid(set_to=True)

    def valid(self, set_to=None):
        if set_to in (True, False):
            self.__valid = set_to
        return self.__valid


class TextLine:

    def __init__(self, line_img, idx, drawing_copy=None):
        self.img = line_img
        self.idx = idx
        # self.char_splitters = []
        self.drawing_copy = drawing_copy

        self.__text_chars = []
        self.std_width = None

        # linear regression parameters
        self.a = None
        self.b = None

    def get_char_splitters(self):
        """

        :return:
        """
        # vertical_smooth = max(5, (lower - upper) // 6), 6
        vertical_smooth = (0, 0)
        vertical_sum_array = proj.project(self.img, direction='vertical', smooth=vertical_smooth)
        char_splitters = proj.get_splitter_end(vertical_sum_array)
        # char_splitters = proj.get_splitter(vertical_sum_array)
        return char_splitters

    def split(self, force=False) -> List[TextChar]:
        """
        Split the line image into char images. If this function has been invoked before, do nothing.
        :param force: force to do splitting, even if `split` has been invoked.
        :return:
        """
        if not force and len(self.__text_chars) > 0:
            return self.__text_chars
        else:
            splitters = self.get_char_splitters()
            self.__text_chars = list(
                filter(lambda tc: tc.has_content(),
                       [TextChar(self.img[:, l:r],
                                 drawing_copy=self.drawing_copy[:, l:r] if self.drawing_copy is not None else None) for
                        l, r in
                        zip(splitters, splitters[1:])]
                       )
            )

            if self.drawing_copy is not None:
                self.drawing_copy[:, splitters] = 120

            return self.__text_chars

    def calculate_meanline_regression(self) -> bool:
        """
        The regression function towards the center points of text contents in char images
        :return: `True` if successful else `False`
        """
        self.split(force=False)
        offset = 0
        centers = []
        for char in self.__text_chars:
            c_y, c_x = char.get_content_center()
            centers.append((c_y, c_x + offset))
            offset += char.get_width()

        n = len(centers)
        if n < 2:
            return False
        _x = _y = _xy = _x2 = 0
        for yi, xi in centers:
            _x += xi
            _y += yi
            _xy += xi * yi
            _x2 += xi ** 2
        self.a = (_x * _y - n * _xy) / (_x * _x - n * _x2)
        self.b = (_y - self.a * _x) / n
        # print("y = %.4f * x + %.4f" % (self.a, self.b))
        if self.drawing_copy is not None:
            y1, x1 = centers[0]
            p1 = x1, self.regression_fn(x1)
            y2, x2 = centers[-1]
            p2 = x2, self.regression_fn(x2)
            # print(p1, p2)
            cv.line(self.drawing_copy, p1, p2, 100, thickness=1)
        return True

    def get_drawing_copy(self):
        self.drawing_copy[[0, -1], :] = 180
        return self.drawing_copy

    def regression_fn(self, x: float) -> int:
        return int(self.a * x + self.b)

    def get_relative_standard_width(self):
        cnt = {}
        # for h in map(lambda c: round(c.get_content_width() / self.get_line_height(), 1),
        for h in map(lambda c: round(c.get_width() / self.get_line_height(), 1),
                     self.__text_chars):
            if h not in cnt:
                cnt[h] = 0
            cnt[h] += 1

        # 先选出占比最大的两个宽度，这是因为当一行中的标点符号和数字太多时，宽度众数不是中文宽度，而是数字宽度
        public_num_2 = sorted(cnt.items(), key=lambda i: i[1], reverse=True)[:2]
        # 从两个众数中，选择宽度大的那一个，作为标准相对宽度
        self.std_width = max(map(lambda x: x[0], public_num_2))
        return self.std_width

    def get_line_height(self):
        return self.img.shape[0]

    def mark_half(self):
        """
        Mark chars that take only half of the relative standard character width, aka. `std_width`.
        These chars are possibly digits, english letters or punctuation signs.

        *If the predication result of a 'half char' is a chinese character, it indicates that an
         `over split` case might has happened, that is, an image containing a single chinese character
        has been splited into two or more images for predicting. We will fix such cases in the following
        steps*
        :return:
        """
        if self.std_width is None:
            self.get_relative_standard_width()

        half_thresh = self.std_width * 0.65
        for c in self.__text_chars:
            if round(c.get_width() / self.get_line_height(), 1) < half_thresh:
                c.half(set_to=True)
        return self

    def merge_components(self):
        """
        **precondition**: `TextPage.set_result()` has been invoked, and predication(inferring) results have set
        **target**: the `char` that takes only half a `std_width` but predication result is a chinese character

        > A chinese character takes one full `std_width`
        :return:
        """
        def merge_score(tgt: TextChar, nbr: TextChar):
            if nbr is None:
                return 0

            score = 0
            add_width = tgt.get_width() + nbr.get_width()
            if nbr.half() and is_chinese(nbr.c):
                score += 1
            elif add_width / self.std_width > 1.1:
                score = 0
            else:
                score += 1 / abs(add_width - self.std_width)
            return score

        def best_merge(idx_list: list) -> list:
            left = idx_list[0] - 1
            right = idx_list[-1] + 1

            if left > 0:
                pass

        for i, char in enumerate(self.__text_chars):
            if is_chinese(char.c) and char.half():
                # 预测出是汉字但只占半个字符位置
                char.drawing_copy[[-1, -2, -3, -4], :-4] = 20, 20, 180
                left_nbr = self.get_chars()[i - 1] if i > 0 else None
                right_nbr = self.get_chars()[i + 1] if i < len(self.get_chars()) - 2 else None
                left_score = merge_score(char, left_nbr)
                right_score = merge_score(char, right_nbr)
                # todo tomorrow's work


    def get_chars(self):
        return self.split(force=False)


class TextPage:

    def __init__(self, page_img, idx, drawing_copy=None):
        self.img = page_img
        self.idx = idx
        self.drawing_copy = drawing_copy

        self.__text_lines = []

    def auto_bin(self):
        self.img = uimg.auto_bin(self.img, otsu=True)
        return self

    def fix_orientation(self):
        self.img = fix_orientation(self.img)
        return self

    def split(self, force=False) -> List[TextLine]:
        if not force and len(self.__text_lines) > 0:
            return self.__text_lines
        else:
            # horizontal_smooth = (0, 0)
            horizontal_smooth = min(15, (self.img.shape[0] * self.img.shape[1]) // 100000), 9
            horizontal_sum_array = proj.project(self.img, direction='horizontal', smooth=horizontal_smooth)
            # line_splitters = proj.get_splitter_end(horizontal_sum_array)
            line_splitters = proj.get_splitter_horizontal(horizontal_sum_array)

            if self.drawing_copy is not None:
                self.drawing_copy[line_splitters, :] = 180

            for line_id, (upper, lower) in enumerate(zip(line_splitters, line_splitters[1:])):
                line_img = self.img[upper:lower, :]
                text_line = TextLine(line_img, line_id,
                                     drawing_copy=self.drawing_copy[upper:lower,
                                                  :] if self.drawing_copy is not None else None)
                self.__text_lines.append(text_line)
        return self.__text_lines

    def get_drawing_copy(self):
        return self.drawing_copy

    def get_lines(self) -> List[TextLine]:
        return self.split(force=False)

    def make_infer_input(self, height=64, width=64):
        char_imgs = []
        for line in self.split(force=False):
            for char in line.split(force=False):
                char_imgs.append(char.fit_resize(height, width))
        return char_imgs

    def set_result(self, results):
        ptr = 0
        for line in self.get_lines():
            for char in line.get_chars():
                c, p = results[ptr]
                char.set_result(c, p=p)
                ptr += 1

    def format_result(self, with_p=False) -> str:
        buff = []
        for line in self.get_lines():
            for char in line.get_chars():
                buff.append(str(char.c) if not with_p else str((char.c, char.p)))
            buff.append('\n')
        return ''.join(buff)
