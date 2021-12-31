import h5py
import numpy as np
import tensorflow
import tensorflow.keras.backend as K
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D,MaxPooling2D, Dropout
import tensorflow.keras.regularizers as tfkreg
from tensorflow.keras.models import Model,Sequential
import matplotlib as mpl
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from matplotlib.pyplot import imshow

import V1_utils


def preprocessing(fileName='C:/Users/jy/Documents/fields/py/SErup/data/v2/v2_2/test.h5',):
    file = h5py.File(fileName)
    x_orig = np.array(file['x'])
    y_orig = np.array(file['y'])
    x = x_orig / 255.
    y = y_orig.reshape(len(y_orig),1)
    randperm = np.random.choice(np.arange(len(y)), size=len(y), replace=False)
    x = x[randperm]
    y = y[randperm]
    return x,y

def data_generator(xdata, ydata, batch_size, cycle=True, givey=True):
    batches = (np.shape(xdata)[0] + batch_size - 1) // batch_size
    if cycle:#用于训练
        while True:
            for i in range(batches):
                X = xdata[i * batch_size:(i + 1) * batch_size]
                Y = ydata[i * batch_size:(i + 1) * batch_size]
                yield (X, Y)
    elif givey:#用于开发及测试
        for i in range(batches):
            X = xdata[i * batch_size:(i + 1) * batch_size]
            Y = ydata[i * batch_size:(i + 1) * batch_size]
            yield (X, Y)
    else:#用于预测
        for i in range(batches):
            X = xdata[i * batch_size:(i + 1) * batch_size]
            yield X



############# Models ##################
def modelV2(input_shape,params):
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
               kernel_regularizer=tfkreg.l2(params['lambda_l2']),
               )(X_input)
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
               kernel_regularizer=tfkreg.l2(params['lambda_l2']),
               )(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)

    # 最大值池化层
    X = MaxPooling2D((3, 3), name='max_pool1',strides=(2,2))(X)
    #X = Dropout(0.3)(X)
    #X = Dropout(0.5)(X)

    # 对X使用 CONV -> BN -> RELU 块
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(384, (3, 3), strides=(1, 1), name='conv2',
               kernel_regularizer=tfkreg.l2(params['lambda_l2']),
               )(X)
    X = BatchNormalization(axis=3, name='bn2')(X)
    X = Activation('relu')(X)

    # 对X使用 CONV -> BN -> RELU 块
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(384, (3, 3), strides=(1, 1), name='conv3',
               kernel_regularizer=tfkreg.l2(params['lambda_l2']),
               )(X)
    X = BatchNormalization(axis=3, name='bn3')(X)
    X = Activation('relu')(X)
    #X = Dropout(0.3)(X)

    # 对X使用 CONV -> BN -> RELU 块
    X = Conv2D(256, (3, 3), strides=(1, 1), name='conv4',
               kernel_regularizer=tfkreg.l2(params['lambda_l2']),
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
              kernel_regularizer=tfkreg.l2(params['lambda_l2']),
              )(X)
    # 创建模型，讲话创建一个模型的实体，我们可以用它来训练、测试。
    model = Model(inputs=X_input, outputs=X, name='modelV2')

    return model

def model_vgg16(input_shape,params):

    X_input = Input(input_shape)
    #X = Conv2D(3, (1, 1), strides=(1, 1))(X_input)
    keras_vgg16 = VGG16(include_top=False, weights="imagenet", input_shape=input_shape,pooling='max')
    vgg16_flower = Sequential()
    vgg16_flower.add(Flatten(input_shape=keras_vgg16.output_shape[1:]))
    vgg16_flower.add(Dropout(0.3))
    vgg16_flower.add(Dense(256, activation="relu"
                           ,kernel_regularizer=tfkreg.l2(params['lambda_l2']),))
    #vgg16_flower.add(Dropout(0.5))
    vgg16_flower.add(Dense(1, activation='sigmoid',kernel_regularizer=tfkreg.l2(params['lambda_l2'])))
    predictons = vgg16_flower(keras_vgg16(X_input))

    model_vgg16 = Model(inputs=X_input, outputs=predictons)

    # for layer in keras_vgg16.layers:
    #     layer.trainable = False
    # keras_vgg16.layers[-1].trainable = True
    # keras_vgg16.layers[-2].trainable = True
    # keras_vgg16.layers[-3].trainable = True
    # keras_vgg16.layers[-4].trainable = True


    for i in range(0,4):
        keras_vgg16.layers[i].trainable = False

    # for x in model_vgg16.trainable_weights:
    #     print(x.name)
    # print('\n')
    # for x in model_vgg16.non_trainable_weights:
    #     print(x.name)
    # print('\n')

    #model_vgg16.summary()

    return model_vgg16

def model_inception3(input_shape,params):
    # inception3 比2加入了BN
    X_input = Input(input_shape)
    # X = Conv2D(3, (1, 1), strides=(1, 1))(X_input)
    keras_InceptionV3 = InceptionV3(include_top=False, weights="imagenet", input_shape=input_shape,pooling='max')
    InceptionV3_flower = Sequential()
    InceptionV3_flower.add(Flatten(input_shape=keras_InceptionV3.output_shape[1:]))
    InceptionV3_flower.add(Dense(256, activation="relu",kernel_regularizer=tfkreg.l2(params['lambda_l2'])))
    InceptionV3_flower.add(Dense(1, activation='sigmoid',kernel_regularizer=tfkreg.l2(params['lambda_l2'])))
    predictons = InceptionV3_flower(keras_InceptionV3(X_input))

    model_InceptionV3 = Model(inputs=X_input, outputs=predictons)
    # for layer in keras_InceptionV3.layers:
    #     layer.trainable = False
    # keras_InceptionV3.layers[-1].trainable = True
    # keras_InceptionV3.layers[-2].trainable = True
    # keras_InceptionV3.layers[-3].trainable = True
    # keras_InceptionV3.layers[-4].trainable = True

    for i in range(0, 4):
        keras_InceptionV3.layers[i].trainable = False
    #
    #
    #
    # for x in model_vgg16.trainable_weights:
    #     print(x.name)
    # print('\n')
    # for x in model_vgg16.non_trainable_weights:
    #     print(x.name)
    # print('\n')

    #model_InceptionV3.summary()

    return model_InceptionV3

def model_mobile2(input_shape,params):
    #3好像还更好，但是好像库里没有？
    X_input = Input(input_shape)
    # X = Conv2D(3, (1, 1), strides=(1, 1))(X_input)
    keras_MobileNetV2 = MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape, pooling='max')
    MobileNetV2_flower = Sequential()
    MobileNetV2_flower.add(Flatten(input_shape=keras_MobileNetV2.output_shape[1:]))
    MobileNetV2_flower.add(Dense(256, activation="relu",kernel_regularizer=tfkreg.l2(params['lambda_l2'])))
    MobileNetV2_flower.add(Dense(1, activation='sigmoid',kernel_regularizer=tfkreg.l2(params['lambda_l2'])))
    predictons = MobileNetV2_flower(keras_MobileNetV2(X_input))

    model_MobileNetV2 = Model(inputs=X_input, outputs=predictons)
    # for layer in keras_MobileNetV2.layers:
    #     layer.trainable = False
    # keras_MobileNetV2.layers[-1].trainable = True
    # keras_MobileNetV2.layers[-2].trainable = True
    # keras_MobileNetV2.layers[-3].trainable = True
    # keras_MobileNetV2.layers[-4].trainable = True

    for i in range(0, 4):
        keras_MobileNetV2.layers[i].trainable = False
    #
    #
    #
    # for x in model_vgg16.trainable_weights:
    #     print(x.name)
    # print('\n')
    # for x in model_vgg16.non_trainable_weights:
    #     print(x.name)
    # print('\n')

    #model_MobileNetV2.summary()

    return model_MobileNetV2

if __name__ == '__main__':
    K.set_image_data_format('channels_last')
    classes = [0, 1]
    xtrain,ytrain = preprocessing(fileName='C:/Users/jy/Documents/fields/py/SErup/data/v2/v2_2/train.h5')
    xdev,ydev = preprocessing(fileName='C:/Users/jy/Documents/fields/py/SErup/data/v2/v2_2/dev.h5')

    ################################# hyperopt model #####################################
    from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
    from tensorflow.keras.callbacks import EarlyStopping

    # 需要的参数
    # lr 1e-5 to 1e-1
    # lambda_l2 1e-5 to 1e-1
    # batch_size 2 to 16

    space = {
        'lr': hp.loguniform('lr', -9, 0),
        'lambda_l2': hp.loguniform('lambda_l2', -9, 0),
        'batch_size': hp.choice('batch_size', [8,])
    }

    f1 = 0
    workidx=5
    print('work {}'.format(workidx))
    maxtrailnum = 50
    def trainAmodel(params):
        global xtrain, xdev, ytrain, ydev
        global f1, workidx
        print('Params testing: ', params)
        model_v1 = model_inception3(xtrain.shape[1:], params)
        opt = tensorflow.keras.optimizers.Adam(lr=params['lr'])
        model_v1.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)
        steps_per_epoch = (np.shape(xtrain)[0] + params['batch_size'] - 1) // params['batch_size']
        # metrics = V1_utils.Metrics(test_data=(X_test[::10], Y_test[::10]), train_data=(X_train[::100], Y_train[::100]))

        from sklearn.utils import class_weight
        import pandas as pd
        class_weight = class_weight.compute_class_weight(class_weight='balanced',
                                                         classes=classes,
                                                         y=ytrain[:, 0])
        cw = dict(enumerate(class_weight))
        early_stopping = EarlyStopping(monitor='val_loss', patience=8, min_delta=0.8, mode='min')

        history = model_v1.fit_generator(generator=data_generator(xtrain, ytrain, params['batch_size']),
                                         steps_per_epoch=steps_per_epoch,
                                         epochs=30,
                                         verbose=0,
                                         validation_data=(xdev[::40], ydev[::40]),
                                         callbacks=[early_stopping],
                                         # callbacks=[metrics],
                                         class_weight=cw,
                                         )
        # 评估模型
        batch_size_test = 4
        preds = model_v1.evaluate_generator(generator=data_generator(xdev, ydev, batch_size_test, cycle=False),verbose=0)
        print("误差值 = " + str(preds[0]))
        print("准确度 = " + str(preds[1]))
        cvres = model_v1.predict_generator(data_generator(xdev,None,4,cycle=False,givey=False), verbose=0)
        cvf1s, cache = V1_utils.fmeasure(ydev, cvres)
        p, r = cache
        print("f1 = {}, precision = {}, recall = {}".format(cvf1s, p, r))
        if cvf1s > f1:
            f1 = cvf1s
            model_v1.save('model/v2_2/model_v2_{}.h5'.format(workidx))
            print(f1)
            plt.figure()
            plt.plot(history.history['loss'], 'b', label='Training loss')
            plt.plot(history.history['val_loss'], 'r', label='Validation val_loss')
            plt.title('Traing and Validation loss')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.savefig('figure/log/loss_v2_2_{}.jpg'.format(workidx))
        return {
            'loss': -cvf1s,
            'status': STATUS_OK
        }


    trials = Trials()
    best = fmin(trainAmodel, space, algo=tpe.suggest, max_evals=maxtrailnum, trials=trials)

    filename = 'model/v2_2/log_v2_{}.npz'.format(workidx)
    np.savez(filename, trials=trials, best=best)

    print('best')
    print(best)

    trialNum = len(trials.trials)
    l2s = np.zeros(trialNum)
    lrs = np.zeros(trialNum)
    losses = np.zeros(trialNum)
    bzs = np.zeros(trialNum)
    for trialidx in range(trialNum):
        thevals = trials.trials[trialidx]['misc']['vals'] #如果是从文件中读取，这一行改成trials[]，即不需要后面那个.trails,下面losses那一行同理
        l2s[trialidx] = thevals['lambda_l2'][0]
        lrs[trialidx] = thevals['lr'][0]
        bzs[trialidx] = (thevals['batch_size'][0] + 1)
        losses[trialidx] = -trials.trials[trialidx]['result']['loss']

    plt.figure()
    # plt.scatter(np.log(lrs), np.log(l2s), c=bzs, s=losses * 100, cmap=mpl.colors.ListedColormap(
    #     ["darkorange", "gold", "lawngreen", "lightseagreen"]
    # ))
    plt.scatter(np.log(lrs), np.log(l2s), c=losses, )
    plt.xlabel('ln[lr]')
    plt.ylabel('ln[λ]')
    plt.title('f1')
    cb = plt.colorbar()
    # cb.set_label('log2[BatchSize]', labelpad=-1)
    plt.savefig('model/v2_2/hyparams_v2_{}.jpg'.format(workidx))
    print('done')

    ###############


    # v2_1 works: #1 test #2 model_v2 #3 vgg-16 keep few layers #4 inception 4 layers #5 mobile net 4 layers#6-8 三个 4层 不要dropout 而是正则化（batchsize控制到16）
    # v2_2 dell # 这电脑上的batch 5 之后是8