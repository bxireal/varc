from os.path import join

from dataset import DatasetFromFolder
from ldc import CelebADataset


def get_training_set(root_dir):
    txt_file = join("/data/JXR/VSRC/dataset/data/train.txt")
    return CelebADataset(txt_file)


def get_test_set(root_dir):
    test_dir = join(root_dir, "test")
    return DatasetFromFolder(test_dir)
