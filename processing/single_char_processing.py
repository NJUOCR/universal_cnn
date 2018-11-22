import cv2 as cv
import os
import utils.uimg as uimg
from data import SingleCharData
from main import Main
from utils.utext import TextPage


def process(page_img, draw=False, filename='single_pld_prob_result'):
    """
    1. auto binarization
    2. orientation correcting
    3. text line segmentation
    4. char segmentation
    5. char image fit-resize
    6. infer
    :param draw:
    :param page_img:
    :return:
    """
    page = TextPage(page_img, 0, drawing_copy=None)

    # 1.
    page.auto_bin()

    if draw:
        page.drawing_copy = cv.cvtColor(page.img.copy(), cv.COLOR_GRAY2BGR)

    # 2.
    page.fix_orientation()

    # 3. & 4.
    page.split()

    main = Main()

    # 5. & 6.
    data = SingleCharData(64, 64) \
        .load_char_map("/usr/local/src/data/stage2/all/all.json") \
        .load_alias_map("/usr/local/src/data/stage2/all/aliasmap.json") \
        .set_images(page.make_infer_input_1()) \
        .init_indices()
    results = main.infer(infer_data=data, input_width=64, input_height=64,
                         num_class=4184, ckpt_dir='/usr/local/src/data/stage2/all/ckpts')
    page.set_result_1(results)
    page.filter_by_p(p_thresh=0.9)
    for line in page.get_lines(ignore_empty=True):
        line.mark_half()
        # line.calculate_meanline_regression()
        line.merge_components()
    uimg.save('/usr/local/src/data/results/%s.jpg' % filename, page.drawing_copy)
    data.set_images(page.make_infer_input_2()).init_indices()
    results2 = main.infer(infer_data=data, input_width=64, input_height=64,
                          num_class=4184, ckpt_dir='/usr/local/src/data/stage2/all/ckpts')
    page.set_result_2(results2)

    with open('/usr/local/src/data/results/%s.html' % filename, 'w', encoding='utf-8') as f:
        with open('./processing/tplt.html') as tplt:
            formatted = page.format_html(tplt.read())
        # f.write(page.format_result(with_p=False))
            f.write(formatted)
    return page


if __name__ == '__main__':
    path = "doc_imgs/2015南立刑初字第0001号_枉法裁判罪84页.pdf/img-0029.jpg"
    if os.path.isfile(path):
        page_img_path = path
        # page_img_path = "doc_imgs/2014东刑初字第0100号_诈骗罪208页.pdf/img-0296.jpg"
        process(uimg.read(page_img_path, read_flag=1), draw=True, filename='test')

