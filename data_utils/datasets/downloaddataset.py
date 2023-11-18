import tensorflow_datasets as tfds
import os

# 设置数据集存储基路径为 ~/dataset
data_dir = os.path.expanduser('~/dataset')

# 下载 Caltech101 数据集
caltech101_ds = tfds.load('caltech101', split='train', as_supervised=True, data_dir=data_dir)

# 下载 Oxford Flowers 102 数据集
flowers102_ds = tfds.load('oxford_flowers102', split='train', as_supervised=True, data_dir=data_dir)

# 下载 Oxford-IIIT Pet 数据集
pets_ds = tfds.load('oxford_iiit_pet', split='train', as_supervised=True, data_dir=data_dir)

# 下载 SUN397 数据集
sun397_ds = tfds.load('sun397', split='train', as_supervised=True, data_dir=data_dir)
