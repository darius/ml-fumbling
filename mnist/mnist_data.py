"""
MNIST data: load from files into Dataset structs.
Based on python-mnist or was it mnist-python?
"""

from array import array
from dataclasses import dataclass
import os
import struct

import numpy as np

DATA_PATH = 'data'

class MNISTException(Exception):
    pass

@dataclass
class Dataset:
    "A collection of examples and corresponding labels."
    examples: object   # No specific type because numpy.typing is only in numpy >= 1.20
    labels: object

    def pairs(self):
        return zip(self.examples, self.labels)

    def __len__(self):          # XXX right?
        return len(self.examples)

    def slice(self, lo, hibound):         # TODO __getitem__ method instead
        return Dataset(self.examples[lo:hibound],
                       self.labels[lo:hibound])

    def shuffled(self, rng):
        perm = np.arange(len(self))
        rng.shuffle(perm)
        return Dataset(self.examples[perm],
                       self.labels[perm])

class MNIST(object):
    def __init__(self, path=DATA_PATH):
        self.path = path
        self.train_img_fname = 'train-images-idx3-ubyte'
        self.train_lbl_fname = 'train-labels-idx1-ubyte'
        self.test_img_fname = 't10k-images-idx3-ubyte'
        self.test_lbl_fname = 't10k-labels-idx1-ubyte'
        self.training_set = None
        self.validation_set = None
        self.test_set = None

    @property
    def training(self):
        self.load_training()
        return self.training_set

    @property
    def validation(self):
        self.load_training()
        return self.validation_set

    @property
    def test(self):
        self.load_testing()
        return self.test_set

    def load_training(self):
        if self.training_set is not None: return
        examples, labels = self.load(os.path.join(self.path, self.train_img_fname),
                                     os.path.join(self.path, self.train_lbl_fname))
        assert len(examples) == len(labels)
        n = 50000
        self.training_set   = Dataset(examples[:n], labels[:n])
        self.validation_set = Dataset(examples[n:], labels[n:])

    def load_testing(self):
        if self.test_set is not None: return
        examples, labels = self.load(os.path.join(self.path, self.test_img_fname),
                                     os.path.join(self.path, self.test_lbl_fname))
        assert len(examples) == len(labels)
        self.test_set = Dataset(examples, labels)

    def load(self, path_img, path_lbl):
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got {}'.format(magic))

            labels = array("B", file.read())
            # TODO try np.array() with the dtype argument -- not sure exactly what
            #labels = np.array(file.read())
            #print(labels.dtype)
            
        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got {}'.format(magic))

            image_data = array("B", file.read())

        examples = np.array(image_data).reshape((size, rows*cols))  # TODO reshape((size, rows, cols)) or something?
        labels = np.array(labels)
        return examples, labels

    @classmethod
    def display(cls, img, width=28, threshold=200):
        chars = ['.@'[intensity > threshold] for intensity in img]
        return '\n'.join(''.join(chars[i:i+width])
                         for i in range(0, len(chars), width))
