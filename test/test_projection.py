import os

import cv2 as cv
from progressbar import ProgressBar

import utils.projection as proj
import utils.uimg as uimg


def batch_generate_projection(src_root="/usr/local/src/data/doc_imgs", histogram_root='/usr/local/src/data/histograms'):
    with ProgressBar() as bar:
        idx = 0
        for directory in os.listdir(src_root):
            dst_dir = os.path.join(histogram_root, directory)
            if not os.path.isdir(dst_dir):  os.mkdir(dst_dir)
            for file in os.listdir(os.path.join(src_root, directory)):
                img = uimg.read(os.path.join(src_root, directory, file), cv.IMREAD_COLOR)
                assert img is not None, "Read fails"
                img = uimg.auto_bin(img)
                img = proj.draw_projective_histogram(img, direction='both')
                uimg.save(os.path.join(dst_dir, file), img)

                idx += 1
                bar.update(idx)


def generate_projection(src, dst, smooth):
    input_img = uimg.read(src, cv.IMREAD_COLOR)
    assert input_img is not None, "Read fails"
    img = uimg.auto_bin(input_img)
    a = proj.project(img, direction='horizontal', smooth=smooth)
    img = proj.draw_projective_histogram(img, direction='both', smooth=smooth)
    uimg.save(dst, img)
    return a


if __name__ == '__main__':
    with open('conv/project_sum.txt', 'w', encoding='utf-8') as f:
        for smooth_time in [0, 6, 9]:
            for smooth_window in [15, 25]:
                array = generate_projection('./img-0008.jpg',
                # array = generate_projection('./img-0420.jpg',
                                            './conv/projection-img-0008_W%d_T%d.jpg' % (smooth_window, smooth_time),
                                            (smooth_window, smooth_time))
                f.write('window:%d,times:%d ' % (smooth_window, smooth_time))
                f.write('\t'.join(map(lambda e: str(e), array)))
                f.write('\n')
