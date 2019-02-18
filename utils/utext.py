import re
from math import sqrt
from collections import deque
from typing import List, Tuple

import cv2 as cv
import numpy as np

import utils.projection as proj
import utils.uchar as uchar
import utils.uimg as uimg
from utils.orientation import fix_orientation

HALF_WIDTH_THRESH_FACTOR = 0.8
MAX_MERGE_WIDTH = 1.35
DEFAULT_NOISE_P_THRESH = 0.5

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

        self.merged = False

        self.is_half = None
        self._c, self.p = None, 0
        self.logs = []

        # The predication for this char is possibly wrong. If `__valid is False`, this char has been merged.
        self.__valid = False
        self.__location = None

        if char_img is not None:
            self.__box_content()

    def __box_content(self):
        """
        Find a rectangle to box the text region

        > If it is an empty image, the 4 values would be `None`
        :return: None
        """
        self.content_top, self.content_left, self.content_bottom, self.content_right = uchar.get_bounds(
            self.img)

    def has_content(self) -> bool:
        """
        If an image has content (aka. text in our context), none of the 4 values should be `None`
        :return: True | False
        """
        return None not in (self.content_top, self.content_left, self.content_bottom, self.content_right)

    def get_content_height(self) -> int or None:
        return self.content_bottom - self.content_top if self.has_content() else None

    def get_content_width(self) -> int or None:
        return self.content_right - self.content_left if self.has_content() else None

    def get_content_center(self) -> Tuple[int] or None:
        """
        Find the center point of the content box (text region).
        *This is mainly used for making **the regression line**,
        which is used for distinguishing `’` from `,` and so on*
        :return: the content center
        """
        if not self.has_content():
            return None
        center_y, center_x = int((self.content_bottom + self.content_top) // 2), int(
            (self.content_right + self.content_left) // 2)
        if self.drawing_copy is not None:
            cv.rectangle(self.drawing_copy, (center_x - 1, center_y - 1), (center_x + 1, center_y + 1), 180,
                         thickness=3)
        return center_y, center_x

    def get_barycenter(self, direction: str = 'vertical') -> float or None:
        """
        ** NOT IN USE **
        calculate the barycenter of content (text)
        :param direction: 'vertical | horizontal' # todo 'both'
        :return: the relative offset to the top-left point of the image
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

    def get_width(self) -> int:
        return self.img.shape[1]

    def get_height(self) -> int:
        return self.img.shape[0]

    def get_padding(self, which: str) -> int:
        assert which in ('left', 'right', 'top', 'bottom')
        if which == 'left':
            return self.content_left
        if which == 'right':
            return self.get_width() - self.content_right
        if which == 'top':
            return self.content_top
        if which == 'bottom':
            return self.get_width() - self.content_top

    def fit_resize(self, new_height, new_width):
        """
        Resize the char image to fit the tensorflow model input.
        This function does not literally resize image to the new size, it briefly following principles below:
        1. Original images cannot be scaled up, but can be scaled down: if the image is too small, it will not be
        recognized by tf model, even if scaled into a larger one
        2. Images with smaller size should be padded into a larger one
        3. Both of the width and height axis should be scaled at the same ratio
        4. A binaryzation will be performed in the end

        > A white image will return if there is a failure
        :param new_height:
        :param new_width:
        :return: A numpy array
        """
        resized_img = uchar.to_size(self.img, new_height, new_width)
        return resized_img if resized_img is not None else (np.ones((new_height, new_width), dtype=np.uint8) * 255)

    def half(self, set_to=None):
        """
        getter and setter
        :param set_to: if it is `True` or `False`, set `self.is_half` accordingly
        :return: whether this char takes half a standard width
        """
        if set_to in (True, False):
            self.is_half = set_to
        return self.is_half

    def set_result(self, c, p=0):
        """
        Set the value of char of this TextChar obj.

        > This function is similar to `set_content_text`, but is used to set values from predication.
        In hence, it has a `p` parameter.
        :param c:
        :param p:
        :return:
        """
        self.set_content_text(c, msg='prediction result')
        self.p = round(p, 2)
        self.valid(set_to=True)

    def valid(self, set_to: bool = None) -> bool:
        """
        Setter and Getter
        :param set_to:
        :return:
        """
        if set_to in (True, False):
            self.__valid = set_to
        return self.__valid

    def location(self, set_to: str = None) -> str:
        """
        Set or get the location of text content.
        :param set_to: if this is `roof` or `floor`, set the `__location` value correspondingly
        :return: this `__location` value
        """
        assert set_to in (None, 'roof', 'floor')
        if set_to in ('roof', 'floor'):
            self.__location = set_to
        return self.__location

    def draw(self, y: tuple, x: tuple, color: tuple):
        """
        Draw auxiliary figures on the `drawing_copy`
        :param y: tuple: (start, end)
        :param x: tuple: (start, end)
        :param color: tuple: (blue, green, red)
        :return:
        """
        if self.drawing_copy is not None:
            self.drawing_copy[y[0]:y[1], x[0]:x[1]] = color

    def set_content_text(self, c, msg='none'):
        """
        Set value of char to this TextChar obj.

        > the value can be inferred by tf model, or rectified from a previous value.
        :param c: value to be set
        :param msg: the reason for setting this value, used for debugging.
        :return:
        """
        self.logs.append({
            'msg': msg,
            'last': self._c,
            'current': c
        })
        self._c = c

    @property
    def c(self):
        """
        Read-only
        :return: the value of char to this TextChar obj.
        """
        return self._c


class TextLine:

    def __init__(self, line_img, idx, drawing_copy=None):
        """

        :param line_img: the original line image
        :param idx: line id
        :param drawing_copy: A numpy array, copy of `line_img`, auxiliary marks should be drew on
        `drawing_copy` not the original image ( in this case, `line_img`)
        """
        self.img = line_img
        self.idx = idx
        # self.char_splitters = []
        self.drawing_copy = drawing_copy

        self.__text_chars = []
        self.std_width = None

        # linear regression parameters
        self.a = None
        self.b = None

        self.__empty = True

        self.__merged_from = []
        self.__merged_char = []

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
        The regression function towards the center points of text contents in char images.
        ** chars that are `invalid` or takes half a `std_width` will be neglected**
        :return: `True` if success else `False`
        """
        self.split(force=False)
        offset = 0
        centers = []
        for char in self.get_chars(only_valid=False):
            if char.valid() and not char.half():
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
            x1 = 0
            p1 = x1, self.regression_fn(x1)
            x2 = self.get_drawing_copy().shape[1] - 1
            p2 = x2, self.regression_fn(x2)
            # print(p1, p2)
            cv.line(self.drawing_copy, p1, p2, 100, thickness=1)
        return True

    def get_drawing_copy(self):
        self.drawing_copy[[0, -1], :] = 180
        return self.drawing_copy

    def regression_fn(self, x: float) -> int:
        return int(self.a * x + self.b)

    def get_relative_standard_width(self, only_valid_char=True):
        assert not self.empty() or not only_valid_char, """
        Assertion failed, one possible reason: Which char is valid or not is unclear before `set_results()`
        """
        cnt = {}
        for h in map(lambda c: self.__relative_width(c.get_content_width()),
                     # for h in map(lambda c: self.__relative_width(c.get_width()),
                     self.get_chars(only_valid=only_valid_char)):
            if h not in cnt:
                cnt[h] = 0
            cnt[h] += 1
        # 先选出占比最大的两个宽度，这是因为当一行中的标点符号和数字太多时，宽度众数不是中文宽度，而是数字宽度
        # 注意：频次为小于2的不应当称为`众数`，它们只是离群点
        public_num_2 = sorted(
            filter(lambda i: i[1] > 1, cnt.items()),
            key=lambda i: i[1], reverse=True
        )[:2]

        if len(public_num_2) == 0:
            self.empty(set_to=True)
            self.std_width = False
            return self.std_width
        # 从两个众数中，选择宽度大的那一个，作为标准相对宽度
        self.std_width = max(map(lambda x: x[0], public_num_2))

        if self.drawing_copy is not None:
            self.drawing_copy[5:-5, :int(self.std_width * self.get_line_height())] = 50, 50, 50
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

        if self.std_width is False:
            return self

        half_thresh = self.std_width * HALF_WIDTH_THRESH_FACTOR
        for c in self.get_chars(only_valid=True):
            if c.valid() and self.__relative_width(c.get_width()) < half_thresh:
                c.half(set_to=True)
                c.draw((-5, -1), (5, -5), (20, 200, 20))
        return self

    def mark_location(self):
        """
        Newly merged chars would not be marked, because regression line is computed by
        the first-time-splited chars
        :return:
        """
        if self.a is None or self.b is None:
            self.calculate_meanline_regression()

        offset = 0
        for char in self.get_chars():
            center_y, center_x = char.get_content_center()
            y_ = self.regression_fn(center_x + offset)
            char.location(set_to='roof' if center_y < y_ else 'floor')
            offset += char.get_width()

    def merge_components(self):
        """
        **precondition**: `TextPage.set_result()` has been invoked, and predication(inferring) results have set
        **target**: the `char` that takes only half a `std_width` but predication result is a chinese character

        > A chinese character takes one full `std_width`
        :return:
        """
        if self.std_width is False:
            return

        def merge_score(cur_indices: deque, nbr: TextChar, which: str = ''):
            assert which in ('left', 'right')
            # cur_width = self.__relative_width(sum(map(lambda idx: self.get_chars()[idx].get_width(), cur_indices)))
            cur_width = sum(map(lambda idx: self.get_char_at(idx).get_width(), cur_indices))
            cur_width -= self.get_char_at(cur_indices[0]).get_padding('left')
            cur_width -= self.get_char_at(cur_indices[-1]).get_padding('right')
            cur_width = self.__relative_width(cur_width)
            if nbr is None or nbr.merged is True:
                return -1

            score = 0
            if nbr.half() and is_chinese(nbr.c):
                score += 1

            # margin
            cur_left = self.get_char_at(cur_indices[0])
            margin_left = cur_left.get_padding('left') + nbr.get_padding('right')
            cur_right = self.get_char_at(cur_indices[-1])
            margin_right = cur_right.get_padding('right') + nbr.get_padding('left')
            distance = margin_right if which == 'right' else margin_left
            score += 1. / (1e-4 + self.__relative_width(distance))

            # width
            # score += (1. / (cur_width + self.__relative_width(nbr.get_width())))
            add_width = cur_width + self.__relative_width(nbr.get_content_width() + distance)
            score += (1. / (1e-4 + add_width))

            if add_width / self.std_width > MAX_MERGE_WIDTH:
                score = -1
            return score

        def best_merge(indices: deque) -> deque:
            left_nbr = self.get_chars()[indices[0] - 1] if indices[0] - 1 >= 0 else None
            right_nbr = self.get_chars()[indices[-1] + 1] if indices[-1] + 1 < len(self.get_chars()) else None
            left_score = merge_score(indices, left_nbr, which='left')
            right_score = merge_score(indices, right_nbr, which='right')
            if left_score > 0 or right_score > 0:
                if left_score > right_score:
                    indices.appendleft(indices[0] - 1)
                else:
                    indices.append(indices[-1] + 1)
                # fixme python 递归可能会很慢，需要检查这一步是否花费太长时间
                return best_merge(indices)
            else:
                # 左右都无法合并
                for index in indices:
                    self.get_chars()[index].merged = True
                return indices

        for i, char in enumerate(self.get_chars(only_valid=False)):
            if not char.valid():
                continue
            if is_chinese(char.c) and char.half():
                # 预测出是汉字但只占半个字符位置
                char.draw((-15, -10), (None, -4), (20, 20, 180))

                if self.get_char_at(i).merged is True:
                    continue
                merged_indices = best_merge(deque([i]))

                if len(merged_indices) > 1:
                    self.__merged_from.append(tuple(merged_indices))
                    merged_img = np.concatenate(
                        list(
                            map(lambda _char_idx: self.get_chars(only_valid=False)[_char_idx].img,
                                merged_indices)
                        ),
                        axis=1
                    )
                    self.__merged_char.append(TextChar(merged_img))

                    for m_char_idx in merged_indices:
                        self.get_chars(only_valid=False)[m_char_idx].valid(set_to=False)
                        self.get_chars(only_valid=False)[m_char_idx].draw((0, 3), (None, None), (180, 20, 20))
                    self.get_chars(only_valid=False)[merged_indices[0]].draw((None, None), (0, 2), (180, 20, 30))
                    self.get_chars(only_valid=False)[merged_indices[-1]].draw((None, -20), (-3, -1), (180, 20, 30))
                else:
                    self.get_char_at(i).merged = False

    def __relative_width(self, pixel_width):
        return round(pixel_width / self.get_line_height(), 1)

    def get_chars(self, only_valid: bool = False, replace_merged: bool = False)->List[TextChar]:
        """
        Get char objects.
        > ** ATTENTION: merged chars are excluded**
        > To get merged chars, use `get_merged_chars()`
        :param replace_merged: if it is `True`, the chars that merged into a new one is suppressed,
        meanwhile the new one will be inserted.
        :param only_valid: return chars that are valid. See `TextChar.valid(set_to=None)`
        :return: list of TextChar objects.
        """
        if replace_merged is True:
            whole = self.split(force=False).copy()
            for indices, m_char in zip(self.get_merged_indices(), self.get_merged_chars()):
                for idx in indices:
                    whole[idx] = None
                whole[indices[0]] = m_char
            return list(filter(lambda char: char is not None and ((not only_valid) or char.valid()), whole))
        else:
            return list(filter(lambda char: (not only_valid) or char.valid(), self.split(force=False)))

    def get_char_at(self, i, only_valid=False):
        return self.get_chars(only_valid=only_valid)[i]

    def get_merged_indices(self) -> List[tuple]:
        return self.__merged_from

    def get_merged_chars(self) -> List[TextChar]:
        return self.__merged_char

    def filter_by_p(self, p_thresh=DEFAULT_NOISE_P_THRESH):
        for char in self.get_chars(only_valid=False):
            if char.p < p_thresh:
                char.valid(set_to=False)
        self.empty(set_to=len(self.get_chars(only_valid=True)) == 0)

    def empty(self, set_to: bool = None) -> bool:
        if set_to in (True, False):
            self.__empty = set_to
        return self.__empty


class TextPage:

    def __init__(self, page_img, idx, drawing_copy=None):
        self.img = page_img
        self.idx = idx
        self.drawing_copy = drawing_copy

        self.__text_lines = []

    def remove_lines(self, origin_size=None):
        h, w = self.img.shape[:2] if origin_size is None else origin_size
        threshold = int(sqrt(h ** 2 + w ** 2) // 20)
        gap = int(w // 300 + 2)
        min_length = int(sqrt(h ** 2 + w ** 2) // 30)
        self.img = 255 - self.img
        print("[Hough Args] threshold=%d, gap=%d, min_length=%d" % (threshold, gap, min_length))
        lines = cv.HoughLinesP(self.img, 1, np.pi / 180, threshold,
                               minLineLength=min_length, maxLineGap=gap)
        _lines = lines[:, 0, :]  # 提取为二维
        for x1, y1, x2, y2 in _lines[:]:
            cv.line(self.img, (x1, y1), (x2, y2), 0, 1)
        self.img = 255 - self.img
        return self

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
            horizontal_smooth = (0, 0)
            # horizontal_smooth = min(15, (self.img.shape[0] * self.img.shape[1]) // 100000), 9
            horizontal_sum_array = proj.project(self.img, direction='horizontal', smooth=horizontal_smooth)
            line_splitters = proj.get_splitter_end(horizontal_sum_array)
            # line_splitters = proj.get_splitter_horizontal(horizontal_sum_array)

            if self.drawing_copy is not None:
                self.drawing_copy[line_splitters, :] = 180

            for line_id, (upper, lower) in enumerate(zip(line_splitters, line_splitters[1:])):
                line_img = self.img[upper:lower, :]
                line_drawing_copy = self.drawing_copy[upper:lower, :] if self.drawing_copy is not None else None
                text_line = TextLine(line_img, line_id, drawing_copy=line_drawing_copy)
                self.__text_lines.append(text_line)
        return self.__text_lines

    def get_drawing_copy(self):
        return self.drawing_copy

    def get_lines(self, ignore_empty: bool = False) -> List[TextLine]:
        return list(filter(lambda line: not (ignore_empty and line.empty()), self.split(force=False)))

    def make_infer_input_1(self, height=64, width=64):
        """
        This function is for **first round** inferring, it will not include the merged chars
        :param height:
        :param width:
        :return:
        """
        char_imgs = []
        for line in self.split(force=False):
            for char in line.split(force=False):
                char_imgs.append(char.fit_resize(height, width))
        return char_imgs

    def make_infer_input_2(self, height=64, width=64):
        char_imgs = []
        for line in self.get_lines(ignore_empty=True):
            for m_char in line.get_merged_chars():
                char_imgs.append(m_char.fit_resize(height, width))
        return char_imgs

    def set_result_1(self, results):
        ptr = 0
        for line in self.get_lines():
            for char in line.get_chars():
                c, p = results[ptr]
                char.set_result(c, p=p)
                ptr += 1
            line.filter_by_p(p_thresh=DEFAULT_NOISE_P_THRESH)

    def set_result_2(self, results):
        ptr = 0
        for line in self.get_lines(ignore_empty=True):
            for char in line.get_merged_chars():
                c, p = results[ptr]
                char.set_result(c, p=p)
                ptr += 1

    def iterate(self, window_size):
        assert window_size % 2 == 1
        r = window_size // 2
        compressed = [TextChar(None) for _ in range(r)]
        for line in self.get_lines(ignore_empty=True):
            compressed += line.get_chars(only_valid=False, replace_merged=True)
        compressed += [TextChar(None) for _ in range(r)]
        for i in range(r, len(compressed) - r):
            yield compressed[i - r: i + r + 1]

    def format_result(self, p_thresh=DEFAULT_NOISE_P_THRESH) -> str:
        buff = []
        for line in self.get_lines(ignore_empty=True):
            line_buff = []
            for char in line.get_chars():
                line_buff.append(char.c if char.p > p_thresh else '')
            for m_indices, m_char in zip(line.get_merged_indices(), line.get_merged_chars()):
                line_buff[m_indices[0]:m_indices[-1]] = ['' for _ in m_indices[:-1]]
                line_buff[m_indices[-1]] = m_char.c if m_char.p > p_thresh else ''
            buff.append(''.join(line_buff))
        return '\n'.join(buff)

    def format_json(self, p_thresh=DEFAULT_NOISE_P_THRESH) -> list:
        buff = []
        for line in self.get_lines(ignore_empty=True):
            line_buff = []
            for char in line.get_chars():
                line_buff.append({
                    'c': char.c,
                    'logs': char.logs,
                    'merged': char.merged,
                    'p': '%.1f' % char.p,
                    'under_thresh': bool(char.p < p_thresh)
                })
            for offset, (m_indices, m_char) in enumerate(zip(line.get_merged_indices(), line.get_merged_chars())):
                line_buff.insert(m_indices[-1] + 1 + offset, {
                    'c': m_char.c,
                    'logs': m_char.logs,
                    'merged': False,
                    'p': '%.1f' % m_char.p,
                    'under_thresh': bool(m_char.p < p_thresh)
                })
            buff.append(line_buff)
        return buff

    def format_markdown(self, p_thresh=DEFAULT_NOISE_P_THRESH):
        buff = []
        for line in self.get_lines(ignore_empty=True):
            line_buff = []
            for char in line.get_chars():
                line_buff.append(str(char.c) if char.p > p_thresh else "")
            for m_indices, m_char in zip(line.get_merged_indices(), line.get_merged_chars()):
                line_buff[m_indices[0]] = '~~`' + line_buff[m_indices[0]]
                line_buff[m_indices[-1]] = line_buff[m_indices[-1]] + '`~~'
                line_buff.insert(m_indices[-1] + 1, ("**%s**" % m_char.c) if m_char.p > p_thresh else '')
            buff.append(''.join(line_buff))
        return '\n\n'.join(buff)

    def format_html(self, tplt: str, p_thresh=DEFAULT_NOISE_P_THRESH):
        buff = []
        for line in self.get_lines(ignore_empty=True):
            line_buff = []
            for char in line.get_chars():
                line_buff.append(("<span>%s</span>" % char.c) if char.p > p_thresh else "")
            for offset, (m_indices, m_char) in enumerate(zip(line.get_merged_indices(), line.get_merged_chars())):
                line_buff.insert(m_indices[-1] + 1 + offset,
                                 ("<strong>%s</strong>" % m_char.c) if m_char.p > p_thresh else '')
                line_buff[m_indices[0] + offset] = '<code class="inline"><del>' + line_buff[m_indices[0] + offset]
                line_buff[m_indices[-1] + offset] = line_buff[m_indices[-1] + offset] + '</del></code>'
            buff.append("<p>%s</p>" % ''.join(line_buff))
        return tplt.replace('%REPLACE%', '\n'.join(buff))

    def filter_by_p(self, p_thresh=DEFAULT_NOISE_P_THRESH):
        for line in self.get_lines():
            for char in line.get_chars():
                if char.p < p_thresh:
                    char.valid(set_to=False)

    def mark_char_location(self):
        for line in self.get_lines(ignore_empty=True):
            line.mark_location()
