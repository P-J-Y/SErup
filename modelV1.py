import tensorflow as tf
import h5py
import numpy as np
import tensorflow.keras.applications.resnet50
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from tensorflow.python.keras.utils.vis_utils import model_to_dot
from tensorflow.python.keras.utils.vis_utils import plot_model
import V1_utils
import tensorflow.keras.backend as K


K.set_image_data_format('channels_last')
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
# os.environ["TF_GPU_ALLOCATOR"] = 'cuda_malloc_async'
# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
# config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
# session = tf.compat.v1.Session(config=config)

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image



def data_generator(xdata, ydata, batch_size):
    batches = (np.shape(xdata)[0] + batch_size - 1) // batch_size
    while True:
        for i in range(batches):
            X = xdata[i * batch_size:(i + 1) * batch_size]
            Y = ydata[i * batch_size:(i + 1) * batch_size]
            yield (X, Y)


def val_generator(xdata, ydata, batch_size):
    batches = (np.shape(xdata)[0] + batch_size - 1) // batch_size
    for i in range(batches):
        X = xdata[i * batch_size:(i + 1) * batch_size]
        Y = ydata[i * batch_size:(i + 1) * batch_size]
        yield (X, Y)

def pre_generator(xdata,batch_size):
    batches = (np.shape(xdata)[0] + batch_size - 1) // batch_size
    for i in range(batches):
        X = xdata[i * batch_size:(i + 1) * batch_size]
        yield X


###################### Training #####################

if __name__ == "__main__":
    xtrain_orig, ytrain, xtest_orig, ytest, classes = V1_utils.load_dataset()
    #xtrain_orig,ytrain,classes = V1_utils.load_dataset_tot('data/data60/data60tot.h5')
    #xtrain_orig, ytrain, classes = V1_utils.load_dataset_tot()
    # Normalize image vectors
    X_train = xtrain_orig / 255.
    X_test = xtest_orig/255.
    #X_test = xtest_orig / 255.
    # Reshape
    Y_train = ytrain.T
    Y_test = ytest.T
    #Y_test = ytest.T
    #imgSize1,imgSize2,nchannel = np.shape(X_train)[1:]

    # 创建一个模型实体
    #model_v1 = V1_utils.modelResnet(X_train.shape[1:])
    #model_v1 = V1_utils.modelVgg19(X_train.shape[1:])

    lambda_l2 = 0.05
    lr = 0.001

    model_v1 = V1_utils.modelV1(X_train.shape[1:],lambda_l2=lambda_l2)
    # 编译模型（在锁层以后操作）
    opt = tensorflow.keras.optimizers.Adam(lr=lr)

    model_v1.compile(loss="binary_crossentropy", metrics=['accuracy'],optimizer=opt)
    # 训练模型
    batch_size = 16
    steps_per_epoch = (np.shape(X_train)[0] + batch_size - 1) // batch_size
    metrics = V1_utils.Metrics(test_data=(X_test[::20], Y_test[::20]),train_data=(X_train[::100],Y_train[::100]))

    from sklearn.utils import class_weight
    import pandas as pd
    class_weight = class_weight.compute_class_weight(class_weight='balanced',
                                                     classes=classes,
                                                     y=Y_train[:,0])
    cw = dict(enumerate(class_weight))

    model_v1.fit_generator(generator=data_generator(X_train, Y_train, batch_size),
                           steps_per_epoch=steps_per_epoch,
                           epochs=8,
                           verbose=1,
                           validation_data=(X_test[::20], Y_test[::20]),
                           callbacks=[metrics],
                           class_weight=cw,
                           )

    #model_v1.fit(X_train, Y_train, epochs=10, batch_size=8,validation_data=(X_test[::2],Y_test[::2]),callbacks = [metrics],)
    # 评估模型
    batch_size_test = 4
    preds = model_v1.evaluate_generator(generator=val_generator(X_test, Y_test, batch_size_test),
                                        verbose=1)
    #警告，不用管 UserWarning: `Model.evaluate_generator` is deprecated and will be removed in a future version.Please use `Model.evaluate`, which supports generators.

    # preds = model_v1.evaluate(X_test, Y_test, batch_size=8, verbose=1, sample_weight=None)
    print("误差值 = " + str(preds[0]))
    print("准确度 = " + str(preds[1]))
    cvres = model_v1.predict_generator(pre_generator(X_test,4),verbose=1)
    cvf1s = V1_utils.fmeasure(Y_test,cvres)
    print(cvf1s)
    print("hhh")
