import os
import re
import json
import cv2 as cv
import numpy as np
from progressbar import ProgressBar


class AbstractData:
    def __init__(self, height, width, num_class):
        self.height, self.width = height, width
        self.images = None
        self.labels = None
        self.indices = None
        self.predictions = None
        self.num_class = num_class

        self.label_map = {}
        self.label_map_reverse = {}

        self.batch_ptr = 0

    def load_char_map(self, file_path):
        print('Loading char map from `%s` ...\t' % file_path, end='')
        with open(file_path, encoding='utf-8') as f:
            self.label_map = json.load(f)
        for k, v in self.label_map.items():
            self.label_map_reverse[v] = k
        print('[done]')
        return self

    def dump_char_map(self, file_path):
        print('Generating char map to `%s` ...\t' % file_path, end='')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.label_map, f, ensure_ascii=False)
        print('[done]')
        return self

    def clear_char_map(self):
        self.label_map_reverse = {}
        self.label_map = {}
        return self

    def read(self, src_root, size=None, make_char_map=False):
        """

        :param src_root: the root directory containing all the data to be read
        :param size: the total number of the data set
        :param make_char_map: whether to make a new charmap
        :return:
        """
        print('loading data...%s from %s ' % ('' if size is None else ("[%d]" % size), src_root))
        images = []
        labels = []
        with ProgressBar(max_value=size) as bar:
            for parent_dir, _, filenames in os.walk(src_root):
                for filename in filenames:
                    lbl = self.filename2label(filename)
                    if make_char_map and lbl not in self.label_map:
                        next_idx = len(self.label_map)
                        self.label_map[lbl] = next_idx
                        self.label_map_reverse[next_idx] = lbl
                    labels.append(self.label_map[lbl])
                    images.append(
                        cv.imdecode(np.fromfile(os.path.join(parent_dir, filename)), 0)
                        .astype(np.float32)
                        .reshape((self.height, self.width, 1)) / 255.
                    )
                    bar.update(bar.value+1)
        self.images = np.array(images)
        self.labels = np.array(labels)
        return self

    def filename2label(self, filename: str):
        # return filename.split('_')[-1].split('.')[0]
        raise Exception('filename2label not implement')

    def shuffle_indices(self):
        samples = self.size()
        self.indices = np.random.permutation(samples)
        self.batch_ptr = 0
        return self

    def init_indices(self):
        samples = self.size()
        self.indices = np.arange(0, samples, dtype=np.int32)
        self.batch_ptr = 0
        return self

    def next_batch(self, batch_size):
        start, end = self.batch_ptr, self.batch_ptr + batch_size
        end = end if end <= len(self.indices) else len(self.indices)
        if start >= self.size():
            return None
        else:
            indices = [self.indices[i] for i in range(start, end)]
            self.batch_ptr = end
            return self.images[indices], self.labels[indices]

    def unmap(self, src):
        if isinstance(src, list):
            return list(map(lambda el: self.label_map_reverse[el], src))
        else:
            return self.label_map_reverse[src]

    def get(self):
        return self.images, self.labels

    def size(self):
        return self.images.shape[0]

    def get_imgs(self):
        return self.images

    def init_pred_buff(self):
        self.predictions = []

    def buff_pred(self, pred):
        self.predictions += pred


class RotationData(AbstractData):
    def filename2label(self, filename: str):
        _ = filename.split('.')[:-1]
        basename = '.'.join(_)
        angle = round(float(basename.split('_')[1]))
        return angle + self.num_class // 2


class SingleCharData(AbstractData):
    ptn = re.compile("\d+_(\w+)\.(?:jpg|png|jpeg)")

    def filename2label(self, filename: str):
        return SingleCharData.ptn.search(filename).group(1)
