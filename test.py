# from sunpy.net import jsoc
# from sunpy.net import attrs as a
# client = jsoc.JSOCClient()
# response = client.search(a.Time('2014-01-01T00:00:00', '2014-01-01T00:10:00'),
#                          a.jsoc.Series('hmi.sharp_720s'), a.jsoc.Notify("2101110617@pku.edu.cn"))
# res = client.fetch(response)
# #requests = client.request_data(response)
#
# print(response)
import os
import shutil
import datetime

import cv2
import h5py
import tensorflow
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
import pandas as pd
from sunpy.coordinates import frames
from tensorflow.python.keras.models import Model

import V1_utils
import getDataset
import json
import numpy as np
from sunpy.net.helioviewer import HelioviewerClient
from sunpy.map import Map
from sunpy.physics.differential_rotation import diff_rot, solar_rotate_coordinate
from ffmpy3 import FFmpeg
import matplotlib as mpl

def getCmeSunWithArIndex(cmeTstart,
                         cmeCoord,
                         arInfo,
                         time_earlier1=80,
                         time_earlier2=20,
                         freq='2min',
                         observatory="SDO",
                         instrument="AIA",
                         measurement="193",
                         mapFileDir=os.getcwd() + "\\figure\\cme\\mapcache\\",
                         submapFileDir=os.getcwd() + "\\figure\\cme\\submapcache\\"):  # cme时间：根据CME catalog；ar信息：根据HEK搜索出比较大的,最好是直接输入AR的完整信息；输出：gif+活动区信息列表
    '''
    获取事件的图片，包括全日面图片和互动区图片
    :param cmeTstart:
    :param cmeCoord:
    :param arInfo:
    :param time_earlier1:
    :param time_earlier2:
    :param freq:
    :param observatory:
    :param instrument:
    :param measurement:
    :param mapFileDir:
    :param submapFileDir:
    :return:
    '''
    # 记得在CME爆发时刻打个记号（图上标记一下，这样判断的时候才能判断出是哪个CME，免得搞错时间
    hv = HelioviewerClient()

    def getMap(t, observatory, instrument, measurement):
        file = hv.download_jp2(t,
                               observatory=observatory,
                               instrument=instrument,
                               measurement=measurement)
        themap = Map(file)
        return themap

    def getCmeSunFrame(t,
                       arInfo,
                       idx,
                       cmeCoord,
                       observatory=observatory,
                       instrument=instrument,
                       measurement=measurement,
                       mapFileDir=mapFileDir):
        themap = getMap(t, observatory=observatory, instrument=instrument, measurement=measurement)
        timeStrForFig = t.strftime("%Y-%m-%d %H:%M:%S")
        ax = plt.subplot(projection=themap)
        # ax = plt.plot()
        im = themap.plot()
        # Prevent the image from being re-scaled while overplotting.
        ax.set_autoscale_on(False)
        aridxs = []
        for aridx in range(arInfo['ar_num']):
            if arInfo["ar_tstarts"][aridx] <= t: # 如果这个时间这个AR还没有产生，就不考虑这个AR
                aridxs.append(aridx)
        if aridxs:
            for i in range(arInfo["ar_num"]):

                if t > arInfo["ar_tends"][i] or t < arInfo["ar_tstarts"][i]:
                    continue
                theAR_coord = arInfo["ar_coords"][i]
                transAR_coord = theAR_coord.transform_to(themap.coordinate_frame)
                theRotated_coord = solar_rotate_coordinate(transAR_coord, time=t)
                if np.isnan(theRotated_coord.Tx.arcsec):
                    # theRotated_coord = theAR_coord
                    continue

                # transRotated_coord = theRotated_coord.transform_to(themap.coordinate_frame)
                ax.plot_coord(theRotated_coord, 'x', label=i)
                if arInfo["ar_coordsyss"][i] == "UTC-HPC-TOPO":
                    thebl = SkyCoord(theRotated_coord.transform_to(theAR_coord.frame).Tx - arInfo["ar_widths"][i] / 2,
                                     theRotated_coord.transform_to(theAR_coord.frame).Ty - arInfo["ar_heights"][i] / 2,
                                     frame=theAR_coord.frame).transform_to(themap.coordinate_frame)
                elif arInfo["ar_coordsyss"][i] == "UTC-HGS-TOPO":
                    thebl = SkyCoord(theRotated_coord.transform_to(theAR_coord.frame).lon - arInfo["ar_widths"][i] / 2,
                                     theRotated_coord.transform_to(theAR_coord.frame).lat - arInfo["ar_heights"][i] / 2,
                                     frame=theAR_coord.frame)
                themap.draw_quadrangle(thebl, width=arInfo["ar_widths"][i], height=arInfo["ar_heights"][i])
                # ==============
                # ax.scatter_coord(theRotated_coord.Tx.arcsec,theRotated_coord.Ty.arcsec)

            # Set title.
            if t < cmeTstart:
                ax.set_title(timeStrForFig)
            else:
                ax.set_title("{} CME".format(timeStrForFig))
            ax.plot_coord(cmeCoord, "r+", label="cme")
            plt.legend()
            plt.savefig("{}{}.png".format(mapFileDir, idx), dpi=600)


    # ims = []
    fig = plt.figure()
    tstart = cmeTstart - datetime.timedelta(minutes=time_earlier1)
    tend = cmeTstart - datetime.timedelta(minutes=time_earlier2)
    ts = list(pd.date_range(tstart, tend, freq=freq))
    idx = 0
    for t in ts:
        getCmeSunFrame(t, arInfo, idx, cmeCoord, observatory=observatory, instrument=instrument,
                       measurement=measurement)
        idx += 1


def checkAevent(cmelist,
                cmeStartTimes,
                cmeTstart,
                arInfo,
                model,
                time_earlier1=80,
                time_earlier2=20,
                freq='2min',
                observatorys=("SDO", "SDO", "SDO", "SDO", "SDO", "SDO", "SDO",),
                instruments=("AIA", "AIA", "AIA", "AIA", "AIA", "HMI", "HMI"),
                measurements=("94", "171", "193", "211", "304", "magnetogram", 'continuum'),
                imgSize=256,
                cmeidx=681,
                filmChannel=2,
                fmt="%Y-%m-%dT%H:%MZ",
                fileDir=os.getcwd() + "\\figure\\test\\",
                ):
    hv = HelioviewerClient()

    def getCme(t):
        cmeinfos = np.array(cmelist)[[(i>t) & (i<(t+datetime.timedelta(hours=24))) for i in cmeStartTimes]]
        return cmeinfos

    def getMap(t, observatory, instrument, measurement):
        file = hv.download_jp2(t,
                               observatory=observatory,
                               instrument=instrument,
                               measurement=measurement)
        themap = Map(file)
        return themap

    def pad2square(submapData):
        '''

        :param submapData: 2 dims array
        :return:
        '''
        dataShape = np.shape(submapData)
        assert len(dataShape)==2, "data dims error"
        bigSize = max(dataShape)
        smallSize = min(dataShape)
        if bigSize==smallSize:
            return submapData
        bigAxis = dataShape.index(bigSize)
        #smallAxis = dataShape.index(smallSize)
        p1 = (bigSize-smallSize)//2
        p2 = bigSize-smallSize-p1
        bigPad = (0,0)
        smallPad = (p1,p2)
        if bigAxis == 0:
            thePad = (bigPad,smallPad)
        else:
            thePad = (smallPad,bigPad)
        res = np.pad(submapData,thePad,'constant',constant_values=(0,0))
        return res

    def getSubmap(t,
                  arInfo,
                  Nchannels,
                  idx=0,
                  mapFileDir=fileDir + "mapcache\\",):
        #get maps
        aridxs = []
        for aridx in range(arInfo['ar_num']):
            if arInfo["ar_tstarts"][aridx] <= t: # 如果这个时间这个AR还没有产生，就不考虑这个AR
                aridxs.append(aridx)
        if aridxs:
            themaps = []
            for channelIdx in range(Nchannels):
                observatory = observatorys[channelIdx]
                instrument = instruments[channelIdx]
                measurement = measurements[channelIdx]
                themap = getMap(t, observatory, instrument, measurement)
                themaps.append(themap)
            timeStrForFig = t.strftime("%Y-%m-%d %H:%M:%S")
            ax = plt.subplot(projection=themaps[filmChannel])
            # ax = plt.plot()
            im = themaps[filmChannel].plot()
            # Prevent the image from being re-scaled while overplotting.
            ax.set_autoscale_on(False)
            # get submaps
            for aridx in aridxs:
                if (arInfo["ar_tstarts"][aridx] > t) or (arInfo["ar_tends"][aridx] < t):
                    continue

                cmeArCoord = arInfo["ar_coords"][aridx]
                theRotated_arc = solar_rotate_coordinate(cmeArCoord.transform_to(frames.Helioprojective),
                                                         time=t).transform_to(cmeArCoord.frame)
                if np.isnan(theRotated_arc.lon.value):
                    print("nan rotated AR center")
                    continue
                    # theRotated_arc = cmeArCoord


                width = arInfo["ar_widths"][aridx]
                height = arInfo["ar_heights"][aridx]
                bottom_left = SkyCoord(theRotated_arc.lon - width / 2,
                                       theRotated_arc.lat - height / 2,
                                       frame=cmeArCoord.frame)
                # theRotated_bl = solar_rotate_coordinate(bottom_left, time=t)
                themaps[filmChannel].draw_quadrangle(bottom_left, width=width, height=height)

                aData = np.zeros((imgSize, imgSize, Nchannels),"single")
                for channelIdx in range(Nchannels):
                    thesubmap = themaps[channelIdx].submap(bottom_left,
                                                           width=width,
                                                           height=height)
                    thedata = thesubmap.data
                    thedata = pad2square(thedata)
                    dst_size = (imgSize, imgSize)
                    thedata = cv2.resize(thedata, dst_size, interpolation=cv2.INTER_AREA)
                    aData[:, :, channelIdx] = thedata
                yhat = model.predict(np.reshape(aData,(1,imgSize,imgSize,Nchannels))/255.)
                ax.plot_coord(theRotated_arc, 'x', label=yhat[0][0])
            # Set title.
            ax.set_title(timeStrForFig)
            cmeinfos = getCme(t)
            for cmeinfo in cmeinfos:
                CeCoordStr = cmeinfo["sourceLocation"]
                CE_coord = getDataset.getCmeCoord(getDataset.breakCoordStr(CeCoordStr))
                if CE_coord is None:
                    continue
                ax.plot_coord(CE_coord, "r+", label="cme")
            plt.legend()
            plt.savefig("{}{}.png".format(mapFileDir, idx), dpi=600)
        else:
            print("no ar at this time")

    def getMeasurementFilm(
                           arInfo,
                           measurement,
                           fileDir,
                           ):
        '''
        实现获取某一个波段的视频,程序会在filmDir中创建几个cache文件夹（最后会删除），用于临时存放图片
        :param ts:
        :param arInfo:
        :param cmeCoord:
        :param observatory:
        :param instrument:
        :param measurement:
        :param filmDir: end with "\\" or "/"
        :return:
        '''
        try:
            os.mkdir(fileDir + "mapcache")
        except FileExistsError:
            shutil.rmtree(fileDir + "mapcache")
            os.mkdir(fileDir + "mapcache")

        tstart = cmeTstart - datetime.timedelta(minutes=time_earlier1)
        tend = cmeTstart - datetime.timedelta(minutes=time_earlier2)
        ts = list(pd.date_range(tstart, tend, freq=freq))
        idx = 0
        for t in ts:
            getSubmap(t,
                      arInfo,
                      len(measurements),
                      idx=idx,
                      mapFileDir=fileDir + "mapcache\\", )
            # ims.append(im)
            idx += 1


        ffin1 = FFmpeg(inputs={fileDir + "mapcache\\" + '%d.png': '-y -r 4'},
                       outputs={fileDir + "CME{}Film{}.mp4".format(measurement, cmeidx): None})
        # print(ffin.cmd)
        ffin1.run()

        shutil.rmtree(fileDir + "mapcache")
        # os.mkdir(pic_path)

    # film_name = current_name + "\\figure\\film\cme{}film{}.mp4".format(measurement, cmeidx)

    #193
    measurement = "193"
    getMeasurementFilm(arInfo,
                       measurement,
                       fileDir,
                       )

def preprocessing(fileName='E:/GithubLocal/SErup/data/v2/v2_1/test.h5',seed=False):
    file = h5py.File(fileName)
    x_orig = np.array(file['x'])
    y_orig = np.array(file['y'])
    x = x_orig / 255.
    y = y_orig.reshape(len(y_orig),1)
    if seed:
        np.random.seed(233)
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




if __name__ == '__main__':
    ##################### conv filter ######################
    from keras import backend as K
    import tensorflow as tf

    # tf.compat.v1.disable_eager_execution()


    # 将浮点图像转换成有效图像
    def deprocess_image(x):
        # 对张量进行规范化
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1
        x += 0.5
        x = np.clip(x, 0, 1)
        # 转化到RGB数组
        x *= 255
        x = np.clip(x, 0, 255).astype('uint8')
        return x


    # 可视化滤波器
    def kernelvisual(model, layer_target=1, inner_layers=1, num_iterate=100):
        # 图像尺寸和通道
        img_height, img_width, num_channels = K.int_shape(model.input)[1:4]
        num_out = K.int_shape(model.layers[layer_target].layers[inner_layers].output)[-1]

        plt.suptitle('[%s] convnet filters visualizing' % model.layers[layer_target].layers[inner_layers].name)

        print('第%d层有%d个通道' % (inner_layers, num_out))
        for i_kernal in range(num_out):
            input_img = model.input
            # 构建一个损耗函数，使所考虑的层的第n个滤波器的激活最大化，-1层softmax层
            if layer_target == -1:
                loss = K.mean(model.output[:, i_kernal])
            else:
                loss = K.mean(model.layers[layer_target].layers[inner_layers].output[:, :, :, i_kernal])  # m*28*28*128
            # 计算图像对损失函数的梯度
            grads = K.gradients(loss, input_img)[0]
            # 效用函数通过其L2范数标准化张量
            grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
            # 此函数返回给定输入图像的损耗和梯度
            iterate = K.function([input_img], [loss, grads])
            # 从带有一些随机噪声的灰色图像开始
            np.random.seed(0)
            # 随机图像
            # input_img_data = np.random.randint(0, 255, (1, img_height, img_width, num_channels))  # 随机
            # input_img_data = np.zeros((1, img_height, img_width, num_channels))   # 零值
            input_img_data = np.random.random((1, img_height, img_width, num_channels)) * 20 + 128.  # 随机灰度
            input_img_data = np.array(input_img_data, dtype=float)
            failed = False
            # 运行梯度上升
            print('####################################', i_kernal + 1)
            loss_value_pre = 0
            # 运行梯度上升num_iterate步
            for i in range(num_iterate):
                loss_value, grads_value = iterate([input_img_data])
                if i % int(num_iterate / 5) == 0:
                    print('Iteration %d/%d, loss: %f' % (i, num_iterate, loss_value))
                    print('Mean grad: %f' % np.mean(grads_value))
                    if all(np.abs(grads_val) < 0.000001 for grads_val in grads_value.flatten()):
                        failed = True
                        print('Failed')
                        break
                    if loss_value_pre != 0 and loss_value_pre > loss_value:
                        break
                    if loss_value_pre == 0:
                        loss_value_pre = loss_value
                    # if loss_value > 0.99:
                    #     break
                input_img_data += grads_value * 1  # e-3
            img_re = deprocess_image(input_img_data[0])
            if num_channels == 1:
                img_re = np.reshape(img_re, (img_height, img_width))
            else:
                img_re = np.reshape(img_re, (img_height, img_width, num_channels))
            plt.subplot(np.ceil(np.sqrt(num_out)), np.ceil(np.sqrt(num_out)), i_kernal + 1)
            plt.imshow(img_re)  # , cmap='gray'
            plt.axis('off')

        plt.show()


    model = tensorflow.keras.models.load_model('model/v2/model_v2_7.h5')
    # kernelvisual(model, 1,1)
    #model.summary()

    fileName = 'E:/GithubLocal/SErup/data/v2/v2_1/test.h5'
    # xtest, ytest = preprocessing(fileName=fileName)
    xtest, ytest = preprocessing(fileName=fileName,seed=True)
    # plt.figure()
    # plt.imshow(xtest[1,:,:,0],cmap='gray')


    model = tensorflow.keras.models.load_model('model/v2/model_v2_7.h5')
    layeridx=299 # 1 4 7 11 14 18 21 22 28 29 30 31 41 44 45 51 52 53 54... 299
    #testres = model.predict_generator(data_generator(xtest, None, 4, cycle=False, givey=False), verbose=0)
    intermediate_layer_model = Model(inputs=model.layers[1].input,
                                     outputs=model.layers[1].layers[layeridx].output)

    #############output###############
    # intermediate_output = intermediate_layer_model.predict(np.reshape(xtest[20],[1,256,256,3]))
    # num_out = np.shape(intermediate_output)[3]
    # plt.figure()
    # plt.suptitle(model.layers[1].layers[layeridx].name)
    # for plti in range(num_out):
    #     plt.subplot(np.ceil(np.sqrt(num_out)), np.ceil(np.sqrt(num_out)), plti+1)
    #     plt.imshow(intermediate_output[0,:,:,plti])#,cmap='gray')
    #     plt.axis('off')
    # #plt.title(model.layers[1].layers[layeridx].name)
    # #plt.show()
    # plt.savefig("figure/v2/v2_1/convRes/conv{}.png".format(layeridx),dpi=600)


    ##################input##################
    # plt.figure()
    # plt.suptitle(input)
    # for plti in range(3):
    #     plt.subplot(np.ceil(np.sqrt(3)), np.ceil(np.sqrt(3)), plti + 1)
    #     plt.imshow(np.reshape(xtest[20],[1,256,256,3])[0, :, :, plti])  # ,cmap='gray')
    #     plt.axis('off')
    # # plt.title(model.layers[1].layers[layeridx].name)
    # # plt.show()
    # plt.savefig("figure/v2/v2_1/convRes/input.png".format(layeridx), dpi=600)

    ############# filter ####################
    import tensorflow as tf
    from tf_keras_vis.activation_maximization import ActivationMaximization


    img = visualize_activation(model=intermediate_layer_model, filter_indices=0)
    print('dome')
    ##########################CME films########################

    # CEidx = 55
    # cmelistpath = 'data/cmelist.json'
    # file = open(cmelistpath, 'r', encoding='utf-8')
    # cmelist = json.load(file)
    # cmeStartTimes = [datetime.datetime.strptime(cmei["startTime"], "%Y-%m-%dT%H:%MZ") for cmei in cmelist]
    # ar_search_t1 = 600
    # ar_search_t2 = 20
    # film_t1 = 600
    # film_t2 = 0
    # freq = '15min'
    # ar_threshold = (100, 6)
    # film_path = "figure\\test\\"
    # for CEidx in range(40, 120, 30):
    #
    #     theCmeInfo = cmelist[CEidx]
    #     try:
    #         theArInfo, cache = getDataset.getArInfoWithCmeInfo(theCmeInfo,
    #                                                            time_earlier1=ar_search_t1,
    #                                                            time_earlier2=ar_search_t2,
    #                                                            ar_threshold=ar_threshold,
    #                                                            fmt="%Y-%m-%dT%H:%MZ")
    #     except AssertionError:
    #         continue
    #
    #
    #     CEtstart = cache
    #
    #     # dists = getDataset.getDists(theArInfo, CE_coord, CEtstart)
    #     # minDist, minidx, matchFlag = getDataset.arCmeMatch(dists, theArInfo)
    #
    #     model = tensorflow.keras.models.load_model('model/v2/model_v2_7.h5')
    #
    #
    #     checkAevent(cmelist,
    #                 cmeStartTimes,
    #                 CEtstart,
    #                 theArInfo,
    #                 model,
    #                 time_earlier1=film_t1,
    #                 time_earlier2=film_t2,
    #                 freq=freq,
    #                 observatorys=("SDO", "SDO", "SDO"),
    #                 instruments=("HMI", "AIA", "AIA"),
    #                 measurements=("magnetogram", "193", "1700"),
    #                 imgSize=256,
    #                 cmeidx=CEidx,
    #                 filmChannel=1,
    #                 fmt="%Y-%m-%dT%H:%MZ",
    #                 fileDir=os.getcwd() + "\\figure\\test\\v2_1\\film\\",
    #                 )
    #
    #     print("hhh")

################### test set performance #######################
        # fileName = 'C:/Users/jy/Documents/fields/py/SErup/data/v2/v2_1/test.h5'
        # # xtest, ytest = preprocessing(fileName=fileName)
        # xtest, ytest = preprocessing(fileName=fileName)
        # plt.figure()
        # plt.imshow(xtest[1,:,:,0],cmap='gray')
        #
        #
        # model = tensorflow.keras.models.load_model('model/v2/model_v2_7.h5')
        # testres = model.predict_generator(data_generator(xtest, None, 4, cycle=False, givey=False), verbose=0)
        # testy = testres>=0.9
        # TP = sum((testy==True) & (ytest==True))
        # FP = sum((testy==True) & (ytest==False))
        # TN = sum((testy==False) & (ytest==False))
        # FN = sum((testy==False) & (ytest==True))
        # print("TP={}, FP={}, TN={}, FN={}".format(TP,FP,TN,FN))
        # testf1s, cache = V1_utils.fmeasure(ytest, testres)
        # p, r = cache
        # print("f1 = {}, precision = {}, recall = {}".format(testf1s, p, r))


########################

#     data = np.load('model/v1/log_v1_2.npz',allow_pickle=True)
#     trials = data['trails']
#     best = data['best']
#
#     trialNum = trials.shape[0]
#     l2s = np.zeros(trialNum)
#     lrs = np.zeros(trialNum)
#     bzs = np.zeros(trialNum)
#     losses = np.zeros(trialNum)
#     for trialidx in range(trialNum):
#         thevals = trials[trialidx]['misc']['vals']
#         l2s[trialidx] = thevals['lambda_l2'][0]
#         lrs[trialidx] = thevals['lr'][0]
#         bzs[trialidx] = (thevals['batch_size'][0]+1)
#         losses[trialidx] = -trials[trialidx]['result']['loss']
#
#     plt.figure()
#     plt.scatter(np.log(lrs), np.log(l2s), c=bzs, s=losses*100, cmap=mpl.colors.ListedColormap(
#     ["darkorange", "gold", "lawngreen", "lightseagreen"]
# ))
#     plt.xlabel('ln[lr]')
#     plt.ylabel('ln[λ]')
#     plt.title('f1')
#     cb = plt.colorbar()
#     cb.set_label('log2[BatchSize]', labelpad=-1)
#     plt.savefig('model/v1/hyparams_v1_2.jpg')



##################



