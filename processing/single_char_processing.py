import threading
import cv2 as cv
import utils.uimg as uimg
import processing.rectification as rct
from data import SingleCharData
from main import Main
from processing.tplt import tplt
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

    def _process(self, page_path: str, p_thresh: float, auxiliary_img: str, auxiliary_html: str)->TextPage:
        print(page_path)
        page = TextPage(uimg.read(page_path, 1), 0, drawing_copy=None)

        # 1.
        page.auto_bin()

        if auxiliary_img is not None:
            page.drawing_copy = cv.cvtColor(page.img.copy(), cv.COLOR_GRAY2BGR)

        # 2.
        page.fix_orientation()

        # 3. & 4.
        page.split()

        # 5. & 6.
        self.data.set_images(page.make_infer_input_1()).init_indices()
        results = self.main.infer(infer_data=self.data, batch_size=self.batch_size)

        # 7.
        page.set_result_1(results)
        page.filter_by_p(p_thresh=p_thresh)
        for line in page.get_lines(ignore_empty=True):
            line.mark_half()
            # line.calculate_meanline_regression()
            line.merge_components()

        # 8.
        self.data.set_images(page.make_infer_input_2()).init_indices()
        results2 = self.main.infer(infer_data=self.data, batch_size=self.batch_size)
        page.set_result_2(results2)
        page.mark_char_location()

        rct.rectify_by_location(page.iterate(1))
        rct.rectify_3(page.iterate(3))

        if auxiliary_img is not None:
            uimg.save(auxiliary_img, page.drawing_copy)

        if auxiliary_html is not None:
            with open(auxiliary_html, 'w', encoding='utf-8') as f:
                f.write(page.format_html(tplt))

        return page

    def get_json_result(self, page_path: str, p_thresh: float, auxiliary_img: str, auxiliary_html: str):
        page = self._process(page_path, p_thresh, auxiliary_img, auxiliary_html)
        return page.format_json(p_thresh=p_thresh)

    def get_text_result(self, page_path: str, p_thresh: float, auxiliary_img: str, auxiliary_html: str):
        page = self._process(page_path, p_thresh, auxiliary_img, auxiliary_html)
        return page.format_result(p_thresh=p_thresh)

if __name__ == '__main__':
    path = "doc_imgs/2015南立刑初字第0001号_枉法裁判罪84页.pdf/img-0228.jpg"

    proc = Processor("/usr/local/src/data/stage2/all_4190/all_4190.json",
                     "/usr/local/src/data/stage2/all_4190/aliasmap.json",
                     '/usr/local/src/data/stage2/all_4190/ckpts',
                     64, 64, 4190, 64)
    res = proc.process(path, 0.9, '/usr/local/src/data/results/auxiliary.png', '/usr/local/src/data/results/auxiliary.html')
    print(res)
