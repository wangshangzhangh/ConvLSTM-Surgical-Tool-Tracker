import tensorflow as tf
import numpy as np
import pickle
import tensorflow as tf
import tensorflow as tf
import os
import cv2  # OpenCV库，用于图像处理

class CholecDataset():
    def __init__(self, file_paths, file_labels, transform=None,
                 loader=cv2.imread):
        self.file_paths = file_paths
        self.file_labels_1 = file_labels[:, range(7)]
        self.file_labels_2 = file_labels[:, -1]
        self.transform = transform
        # self.target_transform=target_transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        labels_1 = self.file_labels_1[index]
        labels_2 = self.file_labels_2[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)
        return imgs, labels_1, labels_2

    def __len__(self):
        return len(self.file_paths)
class DataLoader:
    def __init__(self, paths, labels, num_each, batch_size, num_classes, img_height, img_width):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.img_height = img_height
        self.img_width = img_width

        self.dataset = CholecDataset(paths, labels)
        self.num_each = num_each
        self.num_batches = len(self.dataset) // batch_size
        self.image_paths = paths       
        self.tool_labels = self.dataset.file_labels_1
        
    def _parse_function(self, image_path, tool_label):
        # 读取图片
        image = cv2.imread(image_path.numpy().decode('utf-8'))
        image = cv2.resize(image, (self.img_width, self.img_height))
        image = image.astype(np.float32) / 255.0  # 归一化
        # print("tool_label", tool_label)
        # 将标签转换为独热编码
        # tool_label = tf.one_hot(tool_label, depth=self.num_classes)
        return image, tool_label
    
    def get_iterator(self):
        # 将数据转换为TensorFlow张量
        image_paths = tf.constant(self.image_paths)
        tool_labels = tf.constant(self.tool_labels, dtype=tf.float32)
        print("image_paths", image_paths)

        # 创建数据集
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, tool_labels))
        dataset = dataset.map(lambda image_path, tool_label: 
            (tf.py_function(self._parse_function, [image_path, tool_label], 
            [tf.float32, tf.float32])), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        dataset = dataset.batch(self.batch_size)

        # 保存数据集到文件
        if not os.path.exists('data'):
            os.makedirs('data')
        dataset_filename = 'data/dataset.tfrecord'
        tf.data.experimental.save(dataset, dataset_filename)
        print(f"Dataset saved to {dataset_filename}")

        return dataset.make_initializable_iterator()

class DataLoader:
    def __init__(self, paths, labels, num_each, batch_size, num_classes, img_height, img_width):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.img_height = img_height
        self.img_width = img_width

        self.dataset = CholecDataset(paths, labels)
        self.num_each = num_each
        self.num_batches = len(self.dataset) // batch_size
        self.image_paths = paths       
        self.tool_labels = self.dataset.file_labels_1
        
    def _parse_function(self, image_path, tool_label):
        # 读取图片
        image = cv2.imread(image_path.numpy().decode('utf-8'))
        image = cv2.resize(image, (self.img_width, self.img_height))
        image = image.astype(np.float32) / 255.0  # 归一化
        # print("tool_label", tool_label)
        # 将标签转换为独热编码
        # tool_label = tf.one_hot(tool_label, depth=self.num_classes)
        return image, tool_label
    
    def get_iterator(self):
        # 将数据转换为TensorFlow张量
        image_paths = tf.constant(self.image_paths)
        tool_labels = tf.constant(self.tool_labels, dtype=tf.float32)
        print("image_paths", image_paths)

        # 创建数据集
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, tool_labels))
        dataset = dataset.map(lambda image_path, tool_label: 
            (tf.py_function(self._parse_function, [image_path, tool_label], 
            [tf.float32, tf.float32])), num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.batch(self.batch_size)
        return dataset.make_initializable_iterator()

