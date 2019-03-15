import threading
from typing import Tuple

import cv2 as cv

import utils.rectification as rct
import utils.uimg as uimg
from data import SingleCharData
from main import Main
from utils.utext import TextPage


class Processor(object):
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self, charmap_path, aliasmap_path, ckpt_dir, h, w, num_class, batch_size):
        self.main = Main().load(w, h, num_class=num_class, ckpt_dir=ckpt_dir)
        self.data = SingleCharData(h, w).load_char_map(charmap_path).load_alias_map(aliasmap_path)
        self.batch_size = batch_size

    def __new__(cls, charmap_path, aliasmap_path, ckpt_dir, h, w, num_class, batch_size):
        if Processor._instance is None:
            with Processor._instance_lock:
                if Processor._instance is None:
                    Processor._instance = object.__new__(cls)
        return Processor._instance

    def _process(self, page_path: str, p_thresh: float, auxiliary_img: str,
                 box: Tuple[float, float, float, float] = None, remove_lines: bool = False) -> TextPage:
        print(page_path)
        src = uimg.read(page_path, 1)
        _page = TextPage(src, 0)
        # 1.
        _page.auto_bin()
        if box is not None:
            x1, y1, x2, y2 = box
            x1 = int(x1 * src.shape[1])
            y1 = int(y1 * src.shape[0])
            x2 = int(x2 * src.shape[1])
            y2 = int(y2 * src.shape[0])
            if x1 > x2: x1, x2 = x2, x1
            if y1 > y2: y1, y2 = y2, y1
            region_img = _page.img[y1:y2, x1:x2]
        else:
            region_img = _page.img
        region = TextPage(region_img, 0, drawing_copy=None)

        if remove_lines:
            region.remove_lines(origin_size=src.shape[:2])

        if auxiliary_img is not None and auxiliary_img != '':
            region.drawing_copy = cv.cvtColor(region.img.copy(), cv.COLOR_GRAY2BGR)

            # 2.
            # fixme 低分辨率旋转校正报错
            # page.fix_orientation()

            # 3. & 4.
            region.split()

        # 5. & 6.
        self.data.set_images(region.make_infer_input_1()).init_indices()
        results = self.main.infer(infer_data=self.data, batch_size=self.batch_size)

        # 7.
        region.set_result_1(results)
        region.filter_by_p(p_thresh=p_thresh)
        for line in region.get_lines(ignore_empty=True):
            line.mark_half()
            # line.calculate_meanline_regression()
            line.merge_components()

        # 8.
        self.data.set_images(region.make_infer_input_2()).init_indices()
        results2 = self.main.infer(infer_data=self.data, batch_size=self.batch_size)
        region.set_result_2(results2)
        region.mark_char_location()

        rct.rectify_by_location(region.iterate(1))
        rct.rectify_5(region.iterate(5))

        if auxiliary_img is not None:
            uimg.save(auxiliary_img, region.drawing_copy)

        # if auxiliary_html is not None:
        #     with open(auxiliary_html, 'w', encoding='utf-8') as f:
        #         f.write(page.format_html(tplt))

        return region

    def get_json_result(self, page_path: str, p_thresh: float, auxiliary_img: str,
                        box: Tuple[float, float, float, float] = None, remove_lines=False):
        page = self._process(page_path, p_thresh, auxiliary_img, box=box, remove_lines=remove_lines)
        return page.format_json(p_thresh=p_thresh)

    def get_text_result(self, page_path: str, p_thresh: float, auxiliary_img: str,
                        box: Tuple[float, float, float, float] = None, remove_lines=False):
        page = self._process(page_path, p_thresh, auxiliary_img, box=box, remove_lines=remove_lines)
        return page.format_result(p_thresh=p_thresh)

    def get_verbose_result(self, page_path: str, p_thresh: float, auxiliary_img: str,
                           box: Tuple[float, float, float, float] = None, remove_lines=False):
        page = self._process(page_path, p_thresh, auxiliary_img, box=box, remove_lines=remove_lines)
        return page.format_verbose(p_thresh=p_thresh)


if __name__ == '__main__':
    path = "doc_imgs/2015南立刑初字第0001号_枉法裁判罪84页.pdf/img-0228.jpg"

    proc = Processor("/usr/local/src/data/stage2/all_4190/all_4190.json",
                     "/usr/local/src/data/stage2/all_4190/aliasmap.json",
                     '/usr/local/src/data/stage2/all_4190/ckpts',
                     64, 64, 4190, 64)
    res = proc.get_text_result(path, 0.9, '/usr/local/src/data/results/auxiliary.png')
    print(res)
