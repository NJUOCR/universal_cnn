import os

import cv2 as cv
from progressbar import ProgressBar

import utils.projection as proj
import utils.uimg as uimg


def batch_generate_projection(src_root="/usr/local/src/data/doc_imgs",
                              histogram_root='/usr/local/src/data/histograms_split'):
    with ProgressBar() as bar:
        idx = 0
        for directory in os.listdir(src_root):
            dst_dir = os.path.join(histogram_root, directory)
            if not os.path.isdir(dst_dir):  os.mkdir(dst_dir)
            for file in os.listdir(os.path.join(src_root, directory)):
                img = uimg.read(os.path.join(src_root, directory, file), cv.IMREAD_COLOR)
                assert img is not None, "Read fails"
                img = uimg.auto_bin(img)
                img = uimg.erode_img(img, 10, 10)

                sum_array = proj.project(img, direction='horizontal', smooth=(15, 9))
                spts = proj.get_splitter(sum_array)
                img = proj.draw_projective_histogram(img, direction='horizontal')
                img[spts, :] = 180
                uimg.save(os.path.join(dst_dir, file), img)

                idx += 1
                bar.update(idx)


def generate_projection(src_root="/usr/local/src/data/doc_imgs",
                              block_split='/usr/local/src/data/block_split'):
    with ProgressBar() as bar:
        idx = 0
        for directory in os.listdir(src_root):
            dst_dir = os.path.join(block_split, directory)
            if not os.path.isdir(dst_dir): os.makedirs(dst_dir)
            for file in os.listdir(os.path.join(src_root, directory)):
                input_img = uimg.read(os.path.join(src_root, directory, file), cv.IMREAD_COLOR)
                assert input_img is not None, "Read fails"
                img = uimg.auto_bin(input_img)

                horizontal_sum_array = proj.project(img, direction='horizontal', smooth=(15, 9))
                line_histo_img = proj.draw_projective_histogram(img, direction='horizontal', smooth=(15, 9))

                line_splitters = proj.get_splitter(horizontal_sum_array)
                img[line_splitters, :] = 0
                for upper, lower in zip(line_splitters, line_splitters[1:]):
                    print('lower=', lower)
                    print('upper=', upper)
                    line_img = img[upper:lower, :]
                    vertical_sum_array = proj.project(line_img, direction='vertical', smooth=(max(5, (lower-upper)//6), 6))
                    char_splitter = proj.get_splitter(vertical_sum_array)
                    line_img[:,char_splitter] = 0
                uimg.save(os.path.join(dst_dir, file), img)

                idx += 1
                bar.update(idx)

# 切分字，进行边缘补齐，并存储
def generate_block_split(src_root="/usr/local/src/data/doc_imgs",
                              block_split='/usr/local/src/data/block_split'):
    with ProgressBar() as bar:
        idx = 0
        for directory in os.listdir(src_root):
            dst_dir = os.path.join(block_split, directory)
            if not os.path.isdir(dst_dir): os.makedirs(dst_dir)
            for file in os.listdir(os.path.join(src_root, directory)):
                input_img = uimg.read(os.path.join(src_root, directory, file), cv.IMREAD_COLOR)
                assert input_img is not None, "Read fails"
                img = uimg.auto_bin(input_img)
                horizontal_smooth = min(15, (img.shape[0]*img.shape[1]) // 100000)
                horizontal_sum_array = proj.project(img, direction='horizontal', smooth=(horizontal_smooth, 9))
                # print('horizontal_sum_array=', horizontal_sum_array)
                line_histo_img = proj.draw_projective_histogram(img, direction='horizontal', smooth=(15, 9))
                line_splitters = proj.get_splitter_horizontal(horizontal_sum_array)
                # img[line_splitters, :] = 0
                total_num = 0
                num = 0
                for upper, lower in zip(line_splitters, line_splitters[1:]):
                    # print('lower-upper', lower-upper)
                    line_img = img[upper:lower, :]
                    vertical_sum_array = proj.project(line_img, direction='vertical', smooth=(max(5, (lower-upper)//6), 6))
                    # print('vertical_sum_array=', vertical_sum_array)
                    # if vertical_sum_array < 2:
                    char_splitter = proj.get_splitter(vertical_sum_array)
                    # line_img[:,char_splitter] = 0
                    for left, right in zip(char_splitter, char_splitter[1:]):
                        if lower - upper > 30 and num < 6:
                            split_word_img = img[upper:lower, left:right]
                            resize_img = uimg.fit_resize(split_word_img, 64, 64)
                            pad_img = uimg.pad_to(resize_img, 64, 64, 255)
                            uimg.save(os.path.join(dst_dir, file)+"%d_.jpg" % total_num, pad_img)
                            total_num += 1
                    num += 1
                # 保存单个字

                uimg.save(os.path.join(block_split, file), img)
                idx += 1
                bar.update(idx)

# def generate_projection(src):
#     input_img = uimg.read(src, cv.IMREAD_COLOR)
#     assert input_img is not None, "Read fails"
#     img = uimg.auto_bin(input_img)
#     horizontal_sum_array = proj.project(img, direction='horizontal', smooth=(15, 9))
#     line_histo_img = proj.draw_projective_histogram(img, direction='horizontal', smooth=(15, 9))
#
#     line_splitters = proj.get_splitter(horizontal_sum_array)
#     img[line_splitters, :] = 0
#     for upper, lower in zip(line_splitters, line_splitters[1:]):
#         line_img = img[upper:lower, :]
#         vertical_sum_array = proj.project(line_img, direction='vertical', smooth=(15, 9))
#         char_splitter = proj.get_splitter(vertical_sum_array)
#         line_img[:,char_splitter] = 0
#
#     return img


if __name__ == '__main__':
    # batch_generate_projection()
    src_root = "/usr/local/src/data/poor_imgs_src"
    block_split = '/usr/local/src/data/poor_imgs_block'
    generate_block_split( src_root,block_split)

    # uimg.save('char.jpg', generate_projection('../img-0005.jpg'))
    # img = uimg.read('../img-0005.jpg')
    # img = uimg.erode_img(img, 10, 10)
    # uimg.save('test2.jpg', img)