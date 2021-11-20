import h5py
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils.vis_utils import plot_model
import V1_utils

import keras.backend as K
K.set_image_data_format('channels_last')
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
#os.environ["TF_GPU_ALLOCATOR"] = 'cuda_malloc_async'
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


def modelV1(input_shape):
    """
    实现一个检测CME 爆发的模型

    参数：
        input_shape - 输入的数据的维度
    返回：
        model - 创建的Keras的模型

    """

    # 你可以参考和上面的大纲
    X_input = Input(input_shape)

    # 使用0填充：X_input的周围填充0
    X = ZeroPadding2D((3, 3))(X_input)

    # 对X使用 CONV -> BN -> RELU 块
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # 最大值池化层
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # 降维，矩阵转化为向量 + 全连接层
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # 创建模型，讲话创建一个模型的实体，我们可以用它来训练、测试。
    model = Model(inputs=X_input, outputs=X, name='modelV1')

    return model


if __name__ == "__main__":
    xtrain_orig, ytrain, xtest_orig, ytest, classes = V1_utils.load_dataset()
    # Normalize image vectors
    X_train = xtrain_orig / 255.
    X_test = xtest_orig / 255.
    # Reshape
    Y_train = ytrain.T
    Y_test = ytest.T

    # 创建一个模型实体
    model_v1 = modelV1(X_train.shape[1:])
    # 编译模型
    model_v1.compile("adam", "binary_crossentropy", metrics=['accuracy'])
    # 训练模型
    model_v1.fit(X_train, Y_train, epochs=40, batch_size=1)
    # 评估模型
    preds = model_v1.evaluate(X_test, Y_test, batch_size=1, verbose=1, sample_weight=None)
    print("误差值 = " + str(preds[0]))
    print("准确度 = " + str(preds[1]))
    print("hhh")


