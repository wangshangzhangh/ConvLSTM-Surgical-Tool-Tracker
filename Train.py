import pickle
import tensorflow as tf
import numpy as np
from model import Model  # 导入刚刚的模型定义
from data_loader import DataLoader  # 假设你有一个数据加载器
import os
import multiprocessing

# 获取可用GPU列表
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置TensorFlow只在需要时分配内存
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # 打印异常
        print(e)

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 强制TensorFlow使用CPU

# 参数设置
num_epochs = 160  # 训练轮数
batch_size = 16  # 批次大小
num_classes = 7  # 类别数，根据实际情况调整
learning_rate = 1e-4  # 学习率

# 准备数据
root_dir = '/home/hp/ProcessingData/surgical_tool_detection'
data_file = os.path.join(root_dir, 'train_val_test_paths_labels.pkl') 
img_height = 480 
img_width = 854

# 设置模型保存路径
model_save_path = os.path.join(root_dir, 'saved_models')
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# 加载数据
train_test_paths_labels = []
with open(data_file, 'rb') as f:
    train_test_paths_labels = pickle.load(f)

train_test_paths_labels = np.array(train_test_paths_labels)

# 分离train\val\test数据
train_paths = np.array(train_test_paths_labels[0])
val_paths = np.array(train_test_paths_labels[1])
test_paths = np.array(train_test_paths_labels[2])

train_labels = np.array(train_test_paths_labels[3])
val_labels = np.array(train_test_paths_labels[4])
test_labels = np.array(train_test_paths_labels[5])

train_num_each = np.array(train_test_paths_labels[6])
val_num_each = np.array(train_test_paths_labels[7])
test_num_each = np.array(train_test_paths_labels[8])

train_labels_sum = np.sum(train_labels, axis=0)
val_labels_sum = np.sum(val_labels, axis=0)
test_labels_sum = np.sum(test_labels, axis=0)

# Wc 是每个类别的权重，形状为 [num_classes]
total_labels_sum = train_labels_sum + val_labels_sum + test_labels_sum
new_vector = total_labels_sum[:7]
median = np.median(new_vector)
Wc = median / new_vector

# 创建数据加载器
data_loader_train = DataLoader(train_paths, train_labels, train_num_each, batch_size, num_classes, img_height, img_width)  # 根据实际情况调整
data_loader_val = DataLoader(val_paths, val_labels, val_num_each, batch_size, num_classes, img_height, img_width)  # 根据实际情况调整
data_loader_test = DataLoader(test_paths, test_labels, test_num_each, batch_size, num_classes, img_height, img_width)  # 根据实际情况调整

iterator_train = data_loader_train.get_iterator()
iterator_val = data_loader_val.get_iterator()
iterator_test = data_loader_test.get_iterator()

next_element_train = iterator_train.get_next()
next_element_val = iterator_val.get_next()
next_element_test = iterator_test .get_next()

# 设置TensorFlow会话
config = tf.ConfigProto()
num_processes = multiprocessing.cpu_count()
config.inter_op_parallelism_threads = num_processes  # 根据你的CPU核心数调整
config.intra_op_parallelism_threads = num_processes  # 根据你的CPU核心数调整

# 创建模型
images_placeholder = tf.placeholder(tf.float32, shape=[None, img_height, img_width, 3])  # 根据实际图像尺寸调整
model = Model(images=images_placeholder, num_classes=num_classes)
logits, lhmaps = model.build_model()

# 定义损失函数和优化器
labels_placeholder = tf.placeholder(tf.float32, shape=[None, num_classes])  # 标签占位符
# labels_placeholder = tf.one_hot(labels_placeholder, depth=num_classes)  # 独热编码

# 假设 y_true 是真实的标签，形状为 [batch_size, num_classes]
# y_pred 是模型预测的输出，形状为 [batch_size, num_classes]，通过sigmoid函数处理后表示为概率
y_true = labels_placeholder
y_pred = tf.sigmoid(logits)

print("Logits: {}".format(logits))
print("Lhmaps: {}".format(lhmaps))  
print("Y_true: {}".format(y_true))
print("Y_pred: {}".format(y_pred))  
# 计算加权交叉熵损失
weighted_cross_entropy_loss = -tf.reduce_mean(
    Wc * y_true * tf.log(y_pred + 1e-8) + (1 - y_true) * tf.log(1 - y_pred + 1e-8)
)

optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(weighted_cross_entropy_loss)
saver = tf.train.Saver(max_to_keep=5)  # 保存最近的5个模型

# 开始训练
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator_train.initializer)
    print("data_loader_train.num_batches", data_loader_train.num_batches)
    for epoch in range(num_epochs):
        total_loss = 0
        for _ in range(data_loader_train.num_batches):
            images_batch, labels_batch = sess.run(next_element_train)
            feed_dict = {images_placeholder: images_batch, labels_placeholder: labels_batch}
            _, batch_loss = sess.run([optimizer, weighted_cross_entropy_loss], feed_dict=feed_dict)
            total_loss += batch_loss
            print("Batch Loss: {}".format(batch_loss))
        
        # 验证
        total_val_loss = 0
        for _ in range(data_loader_val.num_batches):
            images_batch_val, labels_batch_val = sess.run(next_element_val)
            feed_dict_val = {images_placeholder: images_batch_val, labels_placeholder: labels_batch_val}
            val_loss = sess.run(weighted_cross_entropy_loss, feed_dict=feed_dict_val)
            total_val_loss += val_loss

        print("Epoch: {}, Training Loss: {}, Validation Loss: {}".format(epoch, total_loss, total_val_loss))

        # 在这里添加保存模型的代码
        save_path = saver.save(sess, model_save_path + '/model.ckpt', global_step=epoch)
        print("Model saved in path: %s" % save_path)


    # 测试
    # total_test_loss = 0
    # for _ in range(data_loader_test.num_batches):
    #     images_batch_test, labels_batch_test = sess.run(next_element_test)
    #     feed_dict_test = {images_placeholder: images_batch_test, labels_placeholder: labels_batch_test}
    #     test_loss = sess.run(loss, feed_dict=feed_dict_test)
    #     total_test_loss += test_loss

    # print("Testing Loss: {}".format(total_test_loss))