#加上earlyStop！！！

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

#################################### load dataset ################################################
    #xtrain_orig, ytrain, xtest_orig, ytest, classes = V1_utils.load_dataset(filename='data/data24hr_1hr/data24hr_1hr.h5')
    xtrain_orig, ytrain, xtest_orig, ytest, classes = V1_utils.load_dataset(filename='data/data60to30/data60to30.h5')
    #xtrain_orig,ytrain,classes = V1_utils.load_dataset_tot('data/data60/data60tot.h5')
    #xtrain_orig, ytrain, classes = V1_utils.load_dataset_tot()
    # Normalize image vectors
    # X_train = xtrain_orig[:,:,:,[1,5,6]] / 255.
    # X_test = xtest_orig[:,:,:,[1,5,6]] / 255.
    X_train = xtrain_orig / 255.
    X_test = xtest_orig / 255.
    #X_test = xtest_orig / 255.
    # Reshape
    Y_train = ytrain.T
    Y_test = ytest.T
    #Y_test = ytest.T
    #imgSize1,imgSize2,nchannel = np.shape(X_train)[1:]

    # 创建一个模型实体
    #model_v1 = V1_utils.modelResnet(X_train.shape[1:])
    #model_v1 = V1_utils.modelVgg19(X_train.shape[1:])

############################################# modelV1 ######################################

    # lambda_l2 = 0.04
    # lr = 0.00005
    #
    # model_v1 = V1_utils.modelV1(X_train.shape[1:],lambda_l2=lambda_l2)
    # # plot_model(model_v1, to_file='model.png')
    # # SVG(model_to_dot(model_v1).create(prog='dot', format='svg'))
    # # 编译模型（在锁层以后操作）
    # opt = tensorflow.keras.optimizers.Adam(lr=lr)
    #
    # model_v1.compile(loss="binary_crossentropy", metrics=['accuracy'],optimizer=opt)
    # # 训练模型
    # batch_size = 16
    # steps_per_epoch = (np.shape(X_train)[0] + batch_size - 1) // batch_size
    # metrics = V1_utils.Metrics(test_data=(X_test[::10], Y_test[::10]),train_data=(X_train[::100],Y_train[::100]))
    #
    # from sklearn.utils import class_weight
    # import pandas as pd
    # class_weight = class_weight.compute_class_weight(class_weight='balanced',
    #                                                  classes=classes,
    #                                                  y=Y_train[:,0])
    # cw = dict(enumerate(class_weight))
    #
    # history=model_v1.fit_generator(generator=data_generator(X_train, Y_train, batch_size),
    #                        steps_per_epoch=steps_per_epoch,
    #                        epochs=30,
    #                        verbose=1,
    #                        validation_data=(X_test[::20], Y_test[::20]),
    #                        callbacks=[metrics],
    #                        class_weight=cw,
    #                        )
    # plt.figure()
    # plt.plot(history.history['loss'], 'b', label='Training loss')
    # plt.plot(history.history['val_loss'], 'r', label='Validation val_loss')
    # plt.title('Traing and Validation loss')
    # plt.legend()
    # plt.savefig('figure/log/loss.jpg')
    #
    # #model_v1.fit(X_train, Y_train, epochs=10, batch_size=8,validation_data=(X_test[::2],Y_test[::2]),callbacks = [metrics],)
    # # 评估模型
    # batch_size_test = 4
    # preds = model_v1.evaluate_generator(generator=val_generator(X_test, Y_test, batch_size_test),
    #                                     verbose=1)
    # #警告，不用管 UserWarning: `Model.evaluate_generator` is deprecated and will be removed in a future version.Please use `Model.evaluate`, which supports generators.
    #
    # # preds = model_v1.evaluate(X_test, Y_test, batch_size=8, verbose=1, sample_weight=None)
    # print("误差值 = " + str(preds[0]))
    # print("准确度 = " + str(preds[1]))
    # cvres = model_v1.predict_generator(pre_generator(X_test,4),verbose=1)
    # cvf1s,cache = V1_utils.fmeasure(Y_test,cvres)
    # p,r = cache
    # print("f1 = {}, precision = {}, recall = {}".format(cvf1s,p,r))
    # print("hhh")


################################# hyperopt model #####################################
from hyperopt import hp, STATUS_OK, Trials, fmin, tpe

#需要的参数
#lr 1e-5 to 1e-1
#lambda_l2 1e-5 to 1e-1
#batch_size 2 to 16

space = {
    'lr':hp.loguniform('lr',-9,-4.5),
    'lambda_l2':hp.loguniform('lambda_l2',-4,0),
    'batch_size':hp.choice('batch_size',[16,])
}

f1 = 0
def trainAmodel(params):
    global X_train,X_test,Y_test,Y_train
    global f1
    print('Params testing: ', params)
    model_v1 = V1_utils.modelV1(X_train.shape[1:],params)
    opt = tensorflow.keras.optimizers.Adam(lr=params['lr'])
    model_v1.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)
    steps_per_epoch = (np.shape(X_train)[0] + params['batch_size'] - 1) // params['batch_size']
    #metrics = V1_utils.Metrics(test_data=(X_test[::10], Y_test[::10]), train_data=(X_train[::100], Y_train[::100]))

    from sklearn.utils import class_weight
    import pandas as pd
    class_weight = class_weight.compute_class_weight(class_weight='balanced',
                                                     classes=classes,
                                                     y=Y_train[:, 0])
    cw = dict(enumerate(class_weight))

    history = model_v1.fit_generator(generator=data_generator(X_train, Y_train, params['batch_size']),
                                     steps_per_epoch=steps_per_epoch,
                                     epochs=30,
                                     verbose=0,
                                     validation_data=(X_test[::20], Y_test[::20]),
                                     #callbacks=[metrics],
                                     class_weight=cw,
                                     )
    # 评估模型
    batch_size_test = 4
    preds = model_v1.evaluate_generator(generator=val_generator(X_test, Y_test, batch_size_test),
                                        verbose=0)
    print("误差值 = " + str(preds[0]))
    print("准确度 = " + str(preds[1]))
    cvres = model_v1.predict_generator(pre_generator(X_test, 4), verbose=0)
    cvf1s, cache = V1_utils.fmeasure(Y_test, cvres)
    p, r = cache
    print("f1 = {}, precision = {}, recall = {}".format(cvf1s, p, r))
    if cvf1s>f1:
        f1 = cvf1s
        model_v1.save('model/v1/model_v1_4.h5')
        print(f1)
        plt.figure()
        plt.plot(history.history['loss'], 'b', label='Training loss')
        plt.plot(history.history['val_loss'], 'r', label='Validation val_loss')
        plt.title('Traing and Validation loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig('figure/log/loss.jpg')
    return {
        'loss':-cvf1s,
        'status':STATUS_OK
    }

trials = Trials()
best = fmin(trainAmodel, space, algo=tpe.suggest, max_evals=50, trials=trials)

filename= 'model/v1/log_v1_4.npz'
np.savez(filename,trials=trials,best=best)

print('best')
print(best)

trialNum = len(trials.trials)
l2s = np.zeros(trialNum)
lrs = np.zeros(trialNum)
losses = np.zeros(trialNum)
for trialidx in range(trialNum):
    thevals = trials.trials[trialidx]['misc']['vals']
    l2s[trialidx] = thevals['lambda_l2'][0]
    lrs[trialidx] = thevals['lr'][0]
    losses[trialidx] = -trials.trials[trialidx]['result']['loss']

plt.figure()
plt.scatter(np.log(lrs),np.log(l2s),c=losses,cmap='jet')
plt.xlabel('ln[lr]')
plt.ylabel('ln[λ]')
plt.title('f1')
plt.colorbar()
plt.savefig('model/v1/hyparams_v1_4.jpg')