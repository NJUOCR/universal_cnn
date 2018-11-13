import utils.uimg as uimg
import utils.uchar as uchar
import utils.projection as proj
from data import SingleCharData
from main import Main
from utils.uchar import contains_text
from utils.orientation import fix_orientation


def preprocess(page_img, draw=False):
    """
    1. auto binarization
    2. orientation correcting
    3. text line segmentation
    4. char segmentation
    5. char image fit-resize
    :param page_img:
    :return:
    """
    # 1.
    bin_img = uimg.auto_bin(page_img, otsu=True)

    # 2.
    ori_img = fix_orientation(bin_img)

    # 3.
    horizontal_smooth = min(15, (ori_img.shape[0] * ori_img.shape[1]) // 100000), 9
    horizontal_sum_array = proj.project(ori_img, direction='horizontal', smooth=horizontal_smooth)
    line_splitters = proj.get_splitter_horizontal(horizontal_sum_array)
    if draw:
        ori_img[line_splitters, :] = 180
    # 4.
    chars = []
    lines = []
    for line_id, (upper, lower) in enumerate(zip(line_splitters, line_splitters[1:])):
        line_img = ori_img[upper:lower, :]
        # todo 去除空行
        # vertical_smooth = max(5, (lower - upper) // 6), 6
        vertical_smooth = (0, 0)
        vertical_sum_array = proj.project(line_img, direction='vertical', smooth=vertical_smooth)
        char_splitters = proj.get_splitter_zero(vertical_sum_array)
        # char_splitters = proj.get_splitter(vertical_sum_array)
        if draw:
            line_img[:, char_splitters] = 180
        for char_id, (left, right) in enumerate(zip(char_splitters, char_splitters[1:])):
            char_img = line_img[:, left:right]

            # 5.
            # resized_char_img = uimg.fit_resize(char_img, 64, 64)
            resized_char_img = uchar.to_size(char_img, 64, 64)
            if resized_char_img is None or not contains_text(resized_char_img, 64):
                continue
            chars.append(resized_char_img)
            lines.append(line_id)
    return chars, lines, ori_img


if __name__ == '__main__':
    page_img_path = "doc_imgs/2014东刑初字第0100号_诈骗罪208页.pdf/img-0020.jpg"
    _chars, _lines, _tiles_img = preprocess(uimg.read(page_img_path, 1), draw=True)

    main = Main()
    data = SingleCharData(64, 64, 3900).load_char_map("label_maps/single_char_1107.json").set_images(_chars).init_indices()
    results = main.infer(infer_data=data, input_width=64, input_height=64,
                         num_class=3900, ckpt_dir='./ckpts/single_char_prob')
    cur_line = -1

    uimg.save('/usr/local/src/data/results/pre_result.jpg', _tiles_img)
    f = open('/usr/local/src/data/results/pre_result.txt', 'w', encoding='utf-8')
    for pred, line_idx in zip(results, _lines):
        if line_idx != cur_line:
            f.write('\n')
            cur_line = line_idx
        if pred[0] == '醫':
            continue
        f.write("(%s,%.2f)" % pred + '\t')
    f.close()
