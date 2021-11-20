import numpy as np
import h5py
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
import tensorflow.keras.applications.resnet50
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def creat_dataset():
    datas = {}
    for i in range(8):
        filename = 'data/data60to30/dataset{}.npz'.format(i + 1)
        dataset = np.load(filename)
        datas["pos{}".format(i)] = dataset['pos'].astype(np.float32)
        datas["neg{}".format(i)] = dataset['neg'].astype(np.float32)

    postrain = datas["pos0"]
    negtrain = datas["neg0"]
    for i in range(1, 6):
        postrain = np.concatenate((postrain, datas["pos{}".format(i)]), axis=0)
        negtrain = np.concatenate((negtrain, datas["neg{}".format(i)]), axis=0)
    postest = np.concatenate((datas["pos6"], datas["pos7"]), axis=0)
    negtest = np.concatenate((datas["neg6"], datas["neg7"]), axis=0)

    xtrain_orig = np.concatenate((postrain, negtrain), axis=0)
    xtest_orig = np.concatenate((postest, negtest), axis=0)
    ytrain = np.append(np.ones(np.shape(postrain)[0], dtype=int),
                       np.zeros(np.shape(negtrain)[0], dtype=int))
    ytest = np.append(np.ones(np.shape(postest)[0], dtype=int),
                      np.zeros(np.shape(negtest)[0], dtype=int))

    randTrainPerm = np.random.choice(np.arange(len(ytrain)), size=len(ytrain), replace=False)
    xtrain_orig = xtrain_orig[randTrainPerm]
    ytrain = ytrain[randTrainPerm]
    classes = [0,1]
    file = h5py.File('data/data60to30/data60to30.h5', 'w')
    file.create_dataset('xtrain_orig', data=xtrain_orig)
    file.create_dataset('ytrain', data=ytrain)
    file.create_dataset('xtest_orig', data=xtest_orig)
    file.create_dataset('ytest', data=ytest)
    file.create_dataset('classes', data=classes)
    file.close()
    # return xtrain_orig,ytrain,xtest_orig,ytest,classes


def load_dataset(filename='data/data60to30/data60to30.h5'):
    file = h5py.File(filename, 'r')
    xtrain_orig = np.array(file['xtrain_orig'][:])
    ytrain = np.array(file['ytrain'][:])
    ytrain = ytrain.reshape(1,len(ytrain))
    xtest_orig = np.array(file['xtest_orig'][:])
    ytest = np.array(file['ytest'][:])
    ytest = ytest.reshape(1,len(ytest))
    classes = np.array(file['classes'][:])
    return xtrain_orig, ytrain, xtest_orig, ytest, classes

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

def modelVgg19(input_shape):
    """
    实现一个检测CME 爆发的模型

    参数：
        input_shape - 输入的数据的维度
    返回：
        model - 创建的Keras的模型

    """
    X_input = Input(input_shape)
    X = Conv2D(3, (1, 1), strides=(1, 1))(X_input)
    keras_vgg19 = VGG19(include_top=False, weights="imagenet", input_shape=X.shape[1:])
    vgg19_flower = Sequential()
    vgg19_flower.add(Flatten(input_shape=keras_vgg19.output_shape[1:]))
    vgg19_flower.add(Dense(512, activation="relu"))
    vgg19_flower.add(Dropout(0.5))
    vgg19_flower.add(Dense(128, activation="relu"))
    vgg19_flower.add(Dropout(0.5))
    vgg19_flower.add(Dense(1, activation='sigmoid'))
    predictons = vgg19_flower(keras_vgg19(X))

    model_vgg19 = Model(inputs=X_input, outputs=predictons)
    for layer in keras_vgg19.layers:
        layer.trainable = False
    keras_vgg19.layers[-1].trainable = True
    keras_vgg19.layers[-2].trainable = True
    keras_vgg19.layers[-3].trainable = True
    keras_vgg19.layers[-4].trainable = True
    keras_vgg19.layers[-5].trainable = True
    keras_vgg19.layers[-6].trainable = True
    for x in model_vgg19.trainable_weights:
        print(x.name)
    print('\n')
    for x in model_vgg19.non_trainable_weights:
        print(x.name)
    print('\n')

    model_vgg19.summary()

    return model_vgg19

def modelResnet(input_shape):
    """
    实现一个检测CME 爆发的模型

    参数：
        input_shape - 输入的数据的维度
    返回：
        model - 创建的Keras的模型

    """

    X_input = Input(input_shape)
    X = Conv2D(3, (1, 1), strides=(1, 1))(X_input)
    keras_resnet = tensorflow.keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet", input_shape=X.shape[1:])
    resnet_flower = Sequential()
    resnet_flower.add(Flatten(input_shape=keras_resnet.output_shape[1:]))
    resnet_flower.add(Dense(512, activation="relu"))
    resnet_flower.add(Dropout(0.5))
    resnet_flower.add(Dense(128, activation="relu"))
    resnet_flower.add(Dropout(0.5))
    resnet_flower.add(Dense(1, activation='sigmoid'))
    predictons = resnet_flower(keras_resnet(X))

    model_resnet = Model(inputs=X_input, outputs=predictons)
    for layer in keras_resnet.layers:
        layer.trainable = False
    keras_resnet.layers[-1].trainable = True
    keras_resnet.layers[-2].trainable = True
    keras_resnet.layers[-3].trainable = True
    keras_resnet.layers[-4].trainable = True
    keras_resnet.layers[-5].trainable = True
    keras_resnet.layers[-6].trainable = True
    for x in model_resnet.trainable_weights:
        print(x.name)
    print('\n')
    for x in model_resnet.non_trainable_weights:
        print(x.name)
    print('\n')

    model_resnet.summary()

    return model_resnet


if __name__ == '__main__':
    #creat_dataset()
    xtrain_orig, ytrain, xtest_orig, ytest, classes = load_dataset()
    print("test down")

