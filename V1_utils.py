import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
import tensorflow.keras.applications.resnet50
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow.keras.regularizers as tfkreg

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

def creat_dataset_tot():
    datas = {}
    for i in range(8):
        filename = 'data/data60to30/dataset{}.npz'.format(i + 1)
        dataset = np.load(filename)
        datas["pos{}".format(i)] = dataset['pos'].astype(np.float32)
        datas["neg{}".format(i)] = dataset['neg'].astype(np.float32)

    postrain = datas["pos0"]
    negtrain = datas["neg0"]
    for i in range(1, 8):
        postrain = np.concatenate((postrain, datas["pos{}".format(i)]), axis=0)
        negtrain = np.concatenate((negtrain, datas["neg{}".format(i)]), axis=0)

    xtrain_orig = np.concatenate((postrain, negtrain), axis=0)
    ytrain = np.append(np.ones(np.shape(postrain)[0], dtype=int),
                       np.zeros(np.shape(negtrain)[0], dtype=int))

    randTrainPerm = np.random.choice(np.arange(len(ytrain)), size=len(ytrain), replace=False)
    xtrain_orig = xtrain_orig[randTrainPerm]
    ytrain = ytrain[randTrainPerm]
    classes = [0,1]
    file = h5py.File('data/data60to30/data60to30tot.h5', 'w')
    file.create_dataset('xtrain_orig', data=xtrain_orig)
    file.create_dataset('ytrain', data=ytrain)
    file.create_dataset('classes', data=classes)
    file.close()
    # return xtrain_orig,ytrain,xtest_orig,ytest,classes

def creat_dataset_single():
    dataset = np.load("data/data60/dataset_60.npz")
    pos = dataset['pos'].astype(np.float32)
    neg = dataset['neg'].astype(np.float32)
    xtrain_orig = np.concatenate((pos,neg),axis=0)
    ytrain = np.append(np.ones(np.shape(pos)[0], dtype=int),
                       np.zeros(np.shape(neg)[0], dtype=int))
    randTrainPerm = np.random.choice(np.arange(len(ytrain)), size=len(ytrain), replace=False)
    xtrain_orig = xtrain_orig[randTrainPerm]
    ytrain = ytrain[randTrainPerm]
    classes = [0, 1]
    file = h5py.File('data/data60/data60tot.h5', 'w')
    file.create_dataset('xtrain_orig', data=xtrain_orig)
    file.create_dataset('ytrain', data=ytrain)
    file.create_dataset('classes', data=classes)
    file.close()

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

def load_dataset_tot(filename='data/data60to30/data60to30tot.h5'):
    file = h5py.File(filename, 'r')
    xtrain_orig = np.array(file['xtrain_orig'][:])
    ytrain = np.array(file['ytrain'][:])
    ytrain = ytrain.reshape(1,len(ytrain))
    classes = np.array(file['classes'][:])
    return xtrain_orig, ytrain, classes

def modelV1(input_shape,lambda_l2=0.1):
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
    #X = ZeroPadding2D((3, 3))(X_input)

    # 对X使用 CONV -> BN -> RELU 块
    X = Conv2D(96, (11, 11), strides=(4, 4), name='conv0',
               kernel_regularizer=tfkreg.l2(lambda_l2),
               )(X_input,)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # 最大值池化层
    X = MaxPooling2D((3, 3), name='max_pool0',strides=(2,2))(X)
    X = Dropout(0.3)(X)
    #X = Dropout(0.5)(X)

    # 使用0填充：X_input的周围填充0
    X = ZeroPadding2D((2, 2))(X)

    # 对X使用 CONV -> BN -> RELU 块
    X = Conv2D(256, (5, 5), strides=(1, 1), name='conv1',
               kernel_regularizer=tfkreg.l2(lambda_l2),
               )(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)

    # 最大值池化层
    X = MaxPooling2D((3, 3), name='max_pool1',strides=(2,2))(X)
    X = Dropout(0.3)(X)
    #X = Dropout(0.5)(X)

    # 对X使用 CONV -> BN -> RELU 块
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(384, (3, 3), strides=(1, 1), name='conv2',
               kernel_regularizer=tfkreg.l2(lambda_l2),
               )(X)
    X = BatchNormalization(axis=3, name='bn2')(X)
    X = Activation('relu')(X)

    # 对X使用 CONV -> BN -> RELU 块
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(384, (3, 3), strides=(1, 1), name='conv3',
               kernel_regularizer=tfkreg.l2(lambda_l2),
               )(X)
    X = BatchNormalization(axis=3, name='bn3')(X)
    X = Activation('relu')(X)
    X = Dropout(0.3)(X)

    # 对X使用 CONV -> BN -> RELU 块
    X = Conv2D(256, (3, 3), strides=(1, 1), name='conv4',
               kernel_regularizer=tfkreg.l2(lambda_l2),
               )(X)
    X = BatchNormalization(axis=3, name='bn4')(X)
    X = Activation('relu')(X)

    # 最大值池化层
    X = MaxPooling2D((3, 3), name='max_pool2',strides=(2,2))(X)
    # X = Dropout(0.5)(X)

    # 降维，矩阵转化为向量 + 全连接层
    X = Flatten()(X)
    #X = Dense(2048, activation='relu', name='fc1',kernel_regularizer=tfkreg.l2(lambda_l2))(X)
    X = Dense(1, activation='sigmoid', name='fc2',
              kernel_regularizer=tfkreg.l2(lambda_l2),
              )(X)
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
    #X = Conv2D(3, (1, 1), strides=(1, 1))(X_input)
    keras_vgg19 = VGG19(include_top=False, weights="imagenet", input_shape=input_shape)
    vgg19_flower = Sequential()
    vgg19_flower.add(Flatten(input_shape=keras_vgg19.output_shape[1:]))
    vgg19_flower.add(Dense(256, activation="relu"))
    vgg19_flower.add(Dropout(0.5))
    vgg19_flower.add(Dense(1, activation='sigmoid'))
    predictons = vgg19_flower(keras_vgg19(X_input))

    model_vgg19 = Model(inputs=X_input, outputs=predictons)
    for layer in keras_vgg19.layers:
        layer.trainable = False
    keras_vgg19.layers[-1].trainable = True
    keras_vgg19.layers[-2].trainable = True
    keras_vgg19.layers[-3].trainable = True
    keras_vgg19.layers[-4].trainable = True


    # for i in range(0,12):
    #     keras_vgg19.layers[i].trainable = False
    #
    #
    #
    # for x in model_vgg19.trainable_weights:
    #     print(x.name)
    # print('\n')
    # for x in model_vgg19.non_trainable_weights:
    #     print(x.name)
    # print('\n')

    # model_vgg19.summary()

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
    # for x in model_resnet.trainable_weights:
    #     print(x.name)
    # print('\n')
    # for x in model_resnet.non_trainable_weights:
    #     print(x.name)
    # print('\n')
    #
    # model_resnet.summary()

    return model_resnet



from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


class Metrics(Callback):

    def __init__(self, test_data,train_data):
        # Should be the label encoding of your classes
        self.test_data = test_data
        self.train_data = train_data

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.train_f1s = []
        self.train_recalls = []
        self.train_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.test_data[0]))).round()
        val_targ = self.test_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        train_predict = (np.asarray(self.model.predict(self.train_data[0]))).round()
        train_targ = self.train_data[1]
        _train_f1 = f1_score(train_targ, train_predict)
        _train_recall = recall_score(train_targ, train_predict)
        _train_precision = precision_score(train_targ, train_predict)
        self.train_f1s.append(_train_f1)
        self.train_recalls.append(_train_recall)
        self.train_precisions.append(_train_precision)
        print("— val_f1: % f — val_precision: % f — val_recall % f \n" % (_val_f1, _val_precision, _val_recall))
        print("— train_f1: % f — train_precision: % f — train_recall % f" % (_train_f1, _train_precision, _train_recall))
        return


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    y_true = np.float32(y_true)
    y_pred = np.float32(y_pred)
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1)





if __name__ == '__main__':
    #creat_dataset()
    #creat_dataset_tot()
    #creat_dataset_single()
    #xtrain_orig, ytrain, xtest_orig, ytest, classes = load_dataset()
    #xtrain_orig, ytrain, classes = load_dataset_tot('data/data60/data60tot.h5')
    model = modelV1([256,256,6])
    model.summary()
    print("test down")

