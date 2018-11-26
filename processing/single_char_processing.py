import os

import cv2 as cv

import utils.uimg as uimg
from data import SingleCharData
from main import Main
from processing.tplt import tplt
from utils.utext import TextPage


def process(page_img, charmap_path: str, aliasmap_path: str, ckpt_dir: str,
            img_height, img_width, num_class, p_thresh, auxiliary_img, auxiliary_html) -> str:
    """

    1. auto binarization
    2. orientation correcting
    3. text line segmentation
    4. char segmentation
    5. char image fit-resize
    6. infer
    7. adjust char segmentation
    8. infer
    :param page_img:
    :param charmap_path:
    :param aliasmap_path:
    :param ckpt_dir:
    :param img_height:
    :param img_width:
    :param num_class:
    :param p_thresh:
    :param auxiliary_img:
    :param auxiliary_html:
    :return:
    """
    page = TextPage(page_img, 0, drawing_copy=None)

    # 1.
    page.auto_bin()

    if auxiliary_img is not False:
        page.drawing_copy = cv.cvtColor(page.img.copy(), cv.COLOR_GRAY2BGR)

    # 2.
    page.fix_orientation()

    # 3. & 4.
    page.split()

    main = Main()

    # 5. & 6.
    data = SingleCharData(img_height, img_width) \
        .load_char_map(charmap_path) \
        .load_alias_map(aliasmap_path) \
        .set_images(page.make_infer_input_1()) \
        .init_indices()
    results = main.infer(infer_data=data, input_width=img_width, input_height=img_height,
                         num_class=num_class, ckpt_dir=ckpt_dir)

    # 7.
    page.set_result_1(results)
    page.filter_by_p(p_thresh=p_thresh)
    for line in page.get_lines(ignore_empty=True):
        line.mark_half()
        # line.calculate_meanline_regression()
        line.merge_components()

    # 8.
    data.set_images(page.make_infer_input_2()).init_indices()
    results2 = main.infer(infer_data=data, input_width=img_width, input_height=img_height,
                          num_class=num_class, ckpt_dir=ckpt_dir)
    page.set_result_2(results2)

    if auxiliary_img is not False:
        uimg.save(auxiliary_img, page.drawing_copy)

    if auxiliary_html is not False:
        with open(auxiliary_html, 'w', encoding='utf-8') as f:
            f.write(page.format_html(tplt))

    return page.format_result(p_thresh=p_thresh)


if __name__ == '__main__':
    path = "doc_imgs/2015南立刑初字第0001号_枉法裁判罪84页.pdf/img-0228.jpg"
    if os.path.isfile(path):
        page_img_path = path
        # page_img_path = "doc_imgs/2014东刑初字第0100号_诈骗罪208页.pdf/img-0296.jpg"
        txt = process(uimg.read(page_img_path, read_flag=1),
                      charmap_path="/usr/local/src/data/stage2/all/all.json",
                      aliasmap_path="/usr/local/src/data/stage2/all/aliasmap.json",
                      ckpt_dir='/usr/local/src/data/stage2/all/ckpts',
                      img_height=64, img_width=64, num_class=4184, p_thresh=0.9,
                      auxiliary_img='/usr/local/src/data/results/auxiliary.jpg',
                      auxiliary_html='/usr/local/src/data/results/auxiliary.html'
                      )
        print(txt)
