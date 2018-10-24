import os
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

        self.batch_ptr = 0

    def read(self, src_root):
        print('loading data...')
        images = []
        labels = []
        with ProgressBar() as bar:
            i = 0
            for parent_dir, _, filenames in os.walk(src_root):
                for filename in filenames:
                    labels.append(self.filename2label(filename))
                    images.append(
                        np.reshape(
                            cv.imdecode(np.fromfile(os.path.join(parent_dir, filename)), 0),
                            (self.height, self.width, 1)
                        )
                    )
                    i += 1
                    bar.update(i)
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

    def init_indices(self):
        samples = self.size()
        self.indices = np.arange(0, samples, dtype=np.int32)
        self.batch_ptr = 0

    def next_batch(self, batch_size):
        start, end = self.batch_ptr, self.batch_ptr + batch_size
        end = end if end <= len(self.indices) else len(self.indices)
        if start >= self.size():
            return None
        else:
            indices = [self.indices[i] for i in range(start, end)]
            self.batch_ptr = end
            return self.images[indices], self.labels[indices]

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
    def filename2label(self, filename: str):
        return filename[:-4].split("_")[1]