# 这个模块是getdataset的升级，首先保存cme和ar的数据库，然后用交叉匹配，匹配出match的cme与ar；
# 正样本就是匹配的结果，而负样本由长时间无cme爆发的ar得到，这是考虑到cme爆发频繁的时候，“负样本”很可能到背面爆发，实则是正样本
# sharp的数据中ar的持续时间一般最长是4hr，超过了会截断，所以会看到一个cme同时match了多个ar，其实是同一个，只不过是不同的时间段
# sunpy好像有bug 有些连续谱的数据读出来有问题（有nan有且检查不出来），1000个ar间隔30分钟，应该有8000个样本左右，但是这样估计会少很多
# 应该处理一下AR位置太边缘，否则数据太少了
import json
import datetime
import os
import time

import cv2
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from sunpy.map import Map
from sunpy.net import Fido
from sunpy.net import attrs as a
import sys
import numpy as np
import h5py
import astropy.units as u
from sunpy.net.helioviewer import HelioviewerClient
from sunpy.physics.differential_rotation import solar_rotate_coordinate

from getDataset import getCmeCoord, breakCoordStr
import matplotlib.pyplot as plt

def arCoord(arx, ary, art, frame="heliographic_stonyhurst", observer="earth"):
    theAR_coord = SkyCoord(arx, ary,
                           frame=frame,
                           obstime=art,
                           observer=observer)
    return theAR_coord


def creatArDataset(event_type="AR", threshold=6, new=False, ):
    '''
    hmi ar 数据库 "UTC-HGS-TOPO": u.deg
    这程序还有问题，可能不会修复了，你要用arinfo直接加载arinfo就行
    :param event_type:
    :return:
    '''
    if new:
        path = 'data/cmefile.json'
        file = open(path, 'r', encoding='utf-8')
        cmeSourceCatalog = json.load(file)
        fmt = "%Y-%m-%dT%H:%MZ"
        t1 = datetime.datetime.strptime(cmeSourceCatalog[0]["startTime"], fmt)
        t2 = datetime.datetime.strptime(cmeSourceCatalog[-1]["startTime"], fmt)
        ARresults = Fido.search(a.Time(t1, t2), a.hek.EventType(event_type),
                                a.hek.FRM.Identifier == "HMI Active Region Patch",
                                # (a.hek.BoundBox.C1UR - a.hek.BoundBox.C1LL) >= threshold,
                                # (a.hek.BoundBox.C2UR - a.hek.BoundBox.C2LL) >= threshold,
                                )
        np.savez("data/arfile.npz", ARresults=ARresults)
        ARresults = ARresults['hek']
    else:
        ARresults = np.load("data/arfile.npz", allow_pickle=True)['ARresults']

    # AR_coordsyss = ARresults['hek']['event_coordsys']
    # AR_coordunitstr = ARresults['hek']['event_coordunit']
    AR_xs = np.array(ARresults['event_coord1'])[0]
    AR_ys = np.array(ARresults['event_coord2'])[0]
    AR_LL_xs = np.array(ARresults['boundbox_c1ll'])[0]
    AR_LL_ys = np.array(ARresults['boundbox_c2ll'])[0]
    AR_UR_xs = np.array(ARresults['boundbox_c1ur'])[0]
    AR_UR_ys = np.array(ARresults['boundbox_c2ur'])[0]
    AR_widths = AR_UR_xs - AR_LL_xs
    AR_heights = AR_UR_ys - AR_LL_ys
    AR_scales = np.min([AR_widths, AR_heights], axis=0)
    bigArIdx = (np.array(AR_scales) >= threshold)
    assert np.sum(
        bigArIdx) != 0, "there are only small ARs during this period, choose another event or choose a smaller threshold 没有合适的活动区，换下一个CME，别浪费时间"
    fmt = "%Y-%m-%dT%H:%M:%S"
    AR_tstarts = np.array(
        [datetime.datetime.strptime(ARresults['event_starttime'][0, i], fmt) for i in range(len(AR_LL_xs))])
    AR_tends = np.array(
        [datetime.datetime.strptime(ARresults['event_endtime'][0, i], fmt) for i in range(len(AR_LL_xs))])
    AR_ts = AR_tstarts + np.divide(np.subtract(AR_tends, AR_tstarts), 2)
    arInfo = {"ar_xs": AR_xs[bigArIdx],
              "ar_ys": AR_ys[bigArIdx],
              "ar_tstarts": AR_tstarts[bigArIdx],
              "ar_tends": AR_tends[bigArIdx],
              "ar_widths": AR_widths[bigArIdx],
              "ar_heights": AR_heights[bigArIdx],
              "ar_scales": AR_scales[bigArIdx],
              "ar_ts": AR_ts[bigArIdx],
              "ar_num": [sum(bigArIdx), ],
              "ar_coordsys": [ARresults['event_coordsys'][0][0], ],
              "ar_coordunit": [ARresults['event_coordunit'][0][0], ],
              }
    # hv = HelioviewerClient()
    # cframes = []
    AR_coords = []
    for i in range(arInfo["ar_num"][0]):
        # cfile = hv.download_jp2(arInfo["ar_ts"][i], observatory="SDO", instrument="HMI", measurement="magnetogram")
        # cmap = Map(cfile)
        # cframe = cmap.coordinate_frame
        # cframes.append(cframe)
        theAR_coord = arCoord(arInfo["ar_xs"][i] * u.deg, arInfo["ar_ys"][i] * u.deg, arInfo["ar_ts"][i])
        # theAR_coord = SkyCoord(arInfo["ar_xs"][i] * u.arcsec, arInfo["ar_ys"][i] * u.arcsec, frame=cframe)
        AR_coords.append(theAR_coord)
    # arInfo["ar_cframes"] = cframes
    arInfo["ar_coords"] = AR_coords
    # np.savez("data/arinfo.npz", arinfo=arInfo)
    return arInfo


def getarlist(fmt="%Y-%m-%dT%H:%M:%S"):
    '''
    把ar 整理成便于存储的形式
    :return:
    '''
    arinfo = creatArDataset()
    ar_tstarts = arinfo["ar_tstarts"]
    arinfo["ar_tstarts"] = np.array(
        [ar_tstarts[i].timestamp() for i in range(arinfo['ar_num'][0])])
    ar_tends = arinfo["ar_tends"]
    arinfo["ar_tends"] = np.array(
        [ar_tends[i].timestamp() for i in range(arinfo['ar_num'][0])])
    ar_ts = arinfo["ar_ts"]
    arinfo["ar_ts"] = np.array(
        [ar_ts[i].timestamp() for i in range(arinfo['ar_num'][0])])
    arinfo.pop("ar_coords")
    file = h5py.File('data/arlist.h5', 'w')
    dt = h5py.special_dtype(vlen=str)
    file.create_dataset('ar_xs', data=arinfo['ar_xs'])
    file.create_dataset('ar_ys', data=arinfo['ar_ys'])
    file.create_dataset('ar_tstarts', data=arinfo['ar_tstarts'])
    file.create_dataset('ar_tends', data=arinfo['ar_tends'])
    file.create_dataset('ar_ts', data=arinfo['ar_ts'])
    file.create_dataset('ar_widths', data=arinfo['ar_widths'])
    file.create_dataset('ar_heights', data=arinfo['ar_heights'])
    file.create_dataset('ar_scales', data=arinfo['ar_scales'])
    file.create_dataset('ar_coordsys', data=arinfo['ar_coordsys'], dtype=dt)
    file.create_dataset('ar_coordunit', data=arinfo['ar_coordunit'], dtype=dt)
    file.create_dataset('ar_num', data=arinfo['ar_num'])
    file.close()


def matchArCme(time_earlier1=24,
               time_earlier2=0, ):
    '''
    match的时候直接比较时间戳（单位是s）
    :param time_earlier1:
    :param time_earlier2:
    :return:
    '''

    def getDists(local_num,local_xs,local_ys,local_ts,CE_coord,CE_tstart,unit=u.deg):
        '''

        :param ARresults:
        :param cache:
        :return:
        '''

        dists = []
        for i in range(local_num):
            AR_coord = arCoord(local_xs[i]*unit,local_ys[i]*unit,datetime.datetime.fromtimestamp(local_ts[i]))

            theRotated_coord = solar_rotate_coordinate(AR_coord.transform_to(frames.Helioprojective), time=CE_tstart)
            theRotated_coord = theRotated_coord.transform_to(AR_coord.frame)
            dists.append(
                np.sqrt((theRotated_coord.lon.deg - CE_coord.lon.deg) ** 2 +
                        (theRotated_coord.lat.deg - CE_coord.lat.deg) ** 2)
            )

        return dists

    def arCmeMatch(dists, local_scales):
        matchflags = np.array(dists) // np.array(local_scales)
        return matchflags

    def findmatchAr(arlist, cmeinfo, matchmax=0):
        '''
        比较cme和ar的时间、位置来判断match
        时间是用的timestamp
        :param arlist:
        :param cmeinfo:
        :return:
        '''
        tstart = datetime.datetime.strptime(cmeinfo["startTime"], "%Y-%m-%dT%H:%MZ")
        t1 = tstart - datetime.timedelta(hours=time_earlier1)
        t2 = tstart - datetime.timedelta(hours=time_earlier2)
        t1 = t1.timestamp()
        t2 = t2.timestamp()
        CeCoordStr = cmeinfo["sourceLocation"]
        CE_coord = getCmeCoord(breakCoordStr(CeCoordStr))
        if CE_coord is None:
            return 0, []

        localAr = (np.array(arlist["ar_tends"]) >= t1) & (np.array(arlist["ar_tends"]) < t2)
        local_num = sum(localAr)
        if local_num==0:
            return 0,[]
        localAridx = np.array([i for i,x in enumerate(localAr) if x])
        dists = getDists(local_num, arlist['ar_xs'][localAr], arlist['ar_ys'][localAr], arlist['ar_ts'][localAr], CE_coord, tstart)
        matchflags = arCmeMatch(dists, arlist['ar_scales'][localAr])
        ismatch = matchflags < (matchmax+1)
        matchnum = sum(ismatch)
        if matchnum==0:
            return 0, []
        matchidx = localAridx[ismatch]
        return matchnum,matchidx

    arlist = h5py.File('data/arlist.h5')
    # ar_num = arlist['ar_num'][0]
    # ar_tstarts = np.array(
    #     [datetime.datetime.fromtimestamp(arlist["ar_tstarts"][i]) for i in range(ar_num)])
    # ar_tends = np.array(
    #     [datetime.datetime.fromtimestamp(arlist["ar_tends"][i]) for i in range(ar_num)])
    # ar_ts = np.array(
    #     [datetime.datetime.fromtimestamp(arlist["ar_ts"][i]) for i in range(ar_num)])

    cmelistpath = 'data/cmelist.json'
    file = open(cmelistpath, 'r', encoding='utf-8')
    cmelist = json.load(file)
    cmeNum = len(cmelist)

    # 每个cme有多少个ar与之match
    matchnums = np.zeros(cmeNum, "int32")
    # 每次有cme match 就生成一个字典，存到下面的列表里，给aridxs，cmeidx，
    matchidxs = []
    for cmeidx in range(cmeNum):
        matchnum,matchidx = findmatchAr(arlist, cmelist[cmeidx])
        matchnums[cmeidx] = matchnum
        matchidxs.append(matchidx)
        print("CMEidx: {}, matchnum: {}".format(cmeidx,matchnum))
        np.savez('data\matchTable.npz', matchnums=matchnums, matchidxs=tuple(matchidxs))
    return matchnums,matchidxs


#matchnums,matchidxs = matchArCme()
# arinfo = creatArDataset()

def positiveSampling(fileName='data/data2/1/testpos.h5',
                     freq='30min',
                     observatorys=("SDO", "SDO", "SDO", "SDO", "SDO", "SDO", "SDO",),
                     instruments=("AIA", "AIA", "AIA", "AIA", "AIA", "HMI", "AIA"),
                     measurements=("94", "171", "193", "211", "304", "magnetogram", '1700'),
                     imgSize=256,
                     i1=0,
                     i2=1054,
                     ):
    # 目前是直接把match的AR全时期的图像都取出来

    def getAArpos(DATA, aridx,arlist,unit=u.deg):
        t1 = datetime.datetime.fromtimestamp(arlist['ar_tstarts'][aridx])
        t2 = datetime.datetime.fromtimestamp(arlist['ar_tends'][aridx])
        arx = arlist['ar_xs'][aridx]
        ary = arlist['ar_ys'][aridx]
        arwidth = arlist['ar_widths'][aridx]
        arheight = arlist['ar_heights'][aridx]
        art  = datetime.datetime.fromtimestamp(arlist['ar_ts'][aridx])
        hv = HelioviewerClient()

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
            assert len(dataShape) == 2, "data dims error"
            bigSize = max(dataShape)
            smallSize = min(dataShape)
            if bigSize == smallSize:
                return submapData
            bigAxis = dataShape.index(bigSize)
            # smallAxis = dataShape.index(smallSize)
            p1 = (bigSize - smallSize) // 2
            p2 = bigSize - smallSize - p1
            bigPad = (0, 0)
            smallPad = (p1, p2)
            if bigAxis == 0:
                thePad = (bigPad, smallPad)
            else:
                thePad = (smallPad, bigPad)
            res = np.pad(submapData, thePad, 'constant', constant_values=(0, 0))
            return res

        def getSubmap(t, arx, ary, arw, arh, art, Nchannels,):
            # get maps

            themaps = []
            for channelIdx in range(Nchannels):
                observatory = observatorys[channelIdx]
                instrument = instruments[channelIdx]
                measurement = measurements[channelIdx]
                themap = getMap(t, observatory, instrument, measurement)
                mapt = datetime.datetime.strptime(themap.date.value, '%Y-%m-%dT%H:%M:%S.%f')
                if (mapt-t) > datetime.timedelta(minutes=10):
                    print("No map loaded, t={}, {}".format(t,measurement))
                    return None
                themaps.append(themap)
            # get submaps
            cmeArCoord = arCoord(arx*unit, ary*unit, art)
            theRotated_arc = solar_rotate_coordinate(cmeArCoord.transform_to(frames.Helioprojective),
                                                     time=t).transform_to(cmeArCoord.frame)
            if np.isnan(theRotated_arc.lon.value):
                theRotated_arc = cmeArCoord
            bottom_left = SkyCoord(theRotated_arc.lon - arw*unit / 2,
                                   theRotated_arc.lat - arh*unit / 2,
                                   frame=cmeArCoord.frame)
            # theRotated_bl = solar_rotate_coordinate(bottom_left, time=t)
            aData = np.zeros((imgSize, imgSize, Nchannels),"single")
            for channelIdx in range(Nchannels):
                thesubmap = themaps[channelIdx].submap(bottom_left,
                                                       width=arw*unit,
                                                       height=arh*unit)
                thedata = thesubmap.data
                thedata = pad2square(thedata)
                dst_size = (imgSize, imgSize)
                thedata = cv2.resize(thedata, dst_size, interpolation=cv2.INTER_AREA)
                aData[:, :, channelIdx] = thedata
            return aData

        ts = list(pd.date_range(t1, t2, freq=freq))
        Nchannels = len(instruments)
        for t in ts:
            try:
                aData = getSubmap(t, arx, ary, arwidth, arheight, art, Nchannels,)
            except ValueError:
                print("AR too close to the edge or nodata ({},{}) ({},{})".format(arx,ary,arwidth,arheight))
                continue
            if aData is None:
                continue
            DATA.append(aData)

    arlist = h5py.File('data/arlist.h5')
    matchTable = np.load('data/matchTable.npz',allow_pickle=True)
    matchnums = matchTable['matchnums']
    matchidxs = matchTable['matchidxs']
    ARidxs = set(np.concatenate(matchidxs[matchnums!=0]))
    ARidxs = list(ARidxs)
    ARidxs.sort()
    DATA = []
    showidx=1
    for aridx in ARidxs[i1:i2]:
        if aridx<1650: # 再次之前是没有匹配的磁图的
            continue
        print('{}/{} ARidx={}'.format(showidx,i2-i1,aridx))
        getAArpos(DATA, aridx, arlist,)
        showidx=showidx+1

    file = h5py.File(fileName,'w')
    file.create_dataset('DATA',data=np.array(DATA))
    file.close()

def negativeSamping(fileName='data/data2/0/testneg.h5',
                    freq='30min',
                    observatorys=("SDO", "SDO", "SDO", "SDO", "SDO", "SDO", "SDO",),
                    instruments=("AIA", "AIA", "AIA", "AIA", "AIA", "HMI", "AIA"),
                    measurements=("94", "171", "193", "211", "304", "magnetogram", '1700'),
                    imgSize=256,
                    mindiff=datetime.timedelta(days=7),
                    dategap=datetime.timedelta(days=2),
                    i1=0, i2=100,
                    new=False):

    def getAArpos(DATA, aridx, arlist, unit=u.deg):
        t1 = datetime.datetime.fromtimestamp(arlist['ar_tstarts'][aridx])
        t2 = datetime.datetime.fromtimestamp(arlist['ar_tends'][aridx])
        arx = arlist['ar_xs'][aridx]
        ary = arlist['ar_ys'][aridx]
        arwidth = arlist['ar_widths'][aridx]
        arheight = arlist['ar_heights'][aridx]
        art = datetime.datetime.fromtimestamp(arlist['ar_ts'][aridx])
        hv = HelioviewerClient()

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
            assert len(dataShape) == 2, "data dims error"
            bigSize = max(dataShape)
            smallSize = min(dataShape)
            if bigSize == smallSize:
                return submapData
            bigAxis = dataShape.index(bigSize)
            # smallAxis = dataShape.index(smallSize)
            p1 = (bigSize - smallSize) // 2
            p2 = bigSize - smallSize - p1
            bigPad = (0, 0)
            smallPad = (p1, p2)
            if bigAxis == 0:
                thePad = (bigPad, smallPad)
            else:
                thePad = (smallPad, bigPad)
            res = np.pad(submapData, thePad, 'constant', constant_values=(0, 0))
            return res

        def getSubmap(t, arx, ary, arw, arh, art, Nchannels,):
            # get maps

            themaps = []
            for channelIdx in range(Nchannels):
                observatory = observatorys[channelIdx]
                instrument = instruments[channelIdx]
                measurement = measurements[channelIdx]
                themap = getMap(t, observatory, instrument, measurement)
                mapt = datetime.datetime.strptime(themap.date.value,'%Y-%m-%dT%H:%M:%S.%f')
                if (mapt-t) > datetime.timedelta(minutes=10):
                    print("No map loaded, t={}, {}".format(t,measurement))
                    return None
                themaps.append(themap)
            # get submaps
            cmeArCoord = arCoord(arx*unit, ary*unit, art)
            theRotated_arc = solar_rotate_coordinate(cmeArCoord.transform_to(frames.Helioprojective),
                                                     time=t).transform_to(cmeArCoord.frame)
            if np.isnan(theRotated_arc.lon.value):
                theRotated_arc = cmeArCoord
            bottom_left = SkyCoord(theRotated_arc.lon - arw*unit / 2,
                                   theRotated_arc.lat - arh*unit / 2,
                                   frame=cmeArCoord.frame)
            # theRotated_bl = solar_rotate_coordinate(bottom_left, time=t)
            aData = np.zeros((imgSize, imgSize, Nchannels),"single")
            for channelIdx in range(Nchannels):
                thesubmap = themaps[channelIdx].submap(bottom_left,
                                                       width=arw*unit,
                                                       height=arh*unit)
                thedata = thesubmap.data
                # if thedata is None:
                #     print("No submapData, channelidx={}".format(channelIdx))
                #     raise ValueError("又是连续谱对不对")
                thedata = pad2square(thedata)
                dst_size = (imgSize, imgSize)
                thedata = cv2.resize(thedata, dst_size, interpolation=cv2.INTER_AREA)
                aData[:, :, channelIdx] = thedata
            return aData

        ts = list(pd.date_range(t1, t2, freq=freq))
        Nchannels = len(instruments)
        for t in ts:
            print('t={}'.format(t))
            try:
                aData = getSubmap(t, arx, ary, arwidth, arheight, art, Nchannels, )
            except ValueError:
                print("AR too close to the edge or nodata ({},{}) ({},{})".format(arx,ary,arwidth,arheight))
                continue

            if aData is None:
                continue
            DATA.append(aData)


    arlist = h5py.File('data/arlist.h5')
    if new:
        cmefilepath = 'data/cmefile.json'
        file = open(cmefilepath, 'r', encoding='utf-8')
        cmefile = json.load(file)
        cmenum = len(cmefile)

        cmeTs = [datetime.datetime.strptime(cmefile[i]["startTime"], "%Y-%m-%dT%H:%MZ") for i in range(cmenum)]
        cmeTdiff = np.diff(cmeTs)
        quiteflag = cmeTdiff > mindiff
        quiteidxs = [i for i, x in enumerate(quiteflag) if x]  # idx to idx+1 标号的cme间隔是大于mindiff的
        quitenum = sum(quiteflag)
        localnums = np.zeros(quitenum, "int32")
        quitearidxs = []
        for j, cmeidx in enumerate(quiteidxs):
            t1 = cmeTs[cmeidx] + dategap
            t2 = cmeTs[cmeidx + 1] - dategap
            t1 = t1.timestamp()
            t2 = t2.timestamp()
            localAr = (np.array(arlist["ar_tstarts"]) >= t1) & (np.array(arlist["ar_tends"]) < t2)
            local_num = sum(localAr)
            localnums[j] = local_num
            localidxs = [i for i, x in enumerate(localAr) if x]
            quitearidxs = quitearidxs + localidxs
        np.savez('data\quiteTable.npz', quiteidxs=quiteidxs, localnums=localnums, quitearidxs=quitearidxs)
    else:
        quiteTable = np.load('data\quiteTable.npz')
        quitearidxs = quiteTable['quitearidxs']
    showidx = 1
    DATA = []
    for aridx in quitearidxs[i1:i2]:
        if aridx<1650:
            continue
        print('{}/{} ARidx={}'.format(showidx, i2 - i1, aridx))
        showidx = showidx + 1
        getAArpos(DATA, aridx, arlist)




    file = h5py.File(fileName, 'w')
    file.create_dataset('DATA', data=np.array(DATA))
    file.close()

# for i in range(0,21):
# # for i in range(13, 14):
#     print('i={}'.format(i))
#     positiveSampling(fileName='data/data2/1/pos{}.h5'.format(i),
#                      #fileName='data/data2/1/pos{}.h5'.format('test'),
#                      freq='30min',
#                      observatorys=("SDO", "SDO", "SDO", "SDO", "SDO", "SDO", "SDO",),
#                      instruments=("HMI", "AIA", "AIA", "AIA", "AIA", "AIA", "AIA"),
#                      measurements=("magnetogram","94", "171", "193", "211", "304", '1700'),
#                      imgSize=256,
#                      i1=50 * i,
#                      i2=50 * (i + 1),
#                      #i2=50*i+1,
#                      )

# negativeSamping(  # fileName='data/data2/0/testneg.h5',
#         fileName='data/data2/0/neg{}.h5'.format('test'),
#         observatorys=("SDO", "SDO", "SDO", "SDO", "SDO", "SDO", "SDO",),
#         instruments=("HMI", "AIA", "AIA", "AIA", "AIA", "AIA", "AIA"),
#         measurements=("magnetogram","94", "171", "193", "211", "304", '1700'),
#         i1=51*25,
#         # i2=100*(i+1),
#         i2=25*51+1)

def keep_connect(url="https://baidu.com"):
    connected = False
    while not connected:
        try:
            r = requests.get(url, timeout=5)
            code = r.status_code
            if code == 200:
                connected = True
                return True
            else:
                print("未连接，等待10s")
                time.sleep(10)
                continue
        except:
            print("未连接，等待10s")
            time.sleep(10)
            continue
        # finally:
        #     "未连接，等待10s"
        #     time.sleep(10)



for i in range(48, 57):
    print('i={}'.format(i))
    done = False
    while not done:
        try:
            negativeSamping(  # fileName='data/data2/0/testneg.h5',
                fileName='data/data2/0/neg{}.h5'.format(i),
                observatorys=("SDO", "SDO", "SDO", "SDO", "SDO", "SDO", "SDO",),
                instruments=("HMI", "AIA", "AIA", "AIA", "AIA", "AIA", "AIA"),
                measurements=("magnetogram", "94", "171", "193", "211", "304", '1700'),
                i1=25 * i,
                # i2=100*(i+1),
                i2=25 * (i + 1))
            done = True
        except (RuntimeError,IOError):
            print("RuntimeError/IOError 检查网络连接是否正常")
            intc = keep_connect()
            print("网络连接正常 检查Helioviewer网站连接是否正常")
            url = 'https://helioviewer.org'
            hvc = keep_connect(url=url)
            print('连接正常，重新运行程序')
            dirname = 'C:/Users/pjy/sunpy/data'
            # 把最近下载的文件删除（因为这个文件很可能是坏的）
            dir_list = os.listdir(dirname)
            if dir_list:
                dir_list = sorted(dir_list,
                                  key=lambda x: os.path.getctime(os.path.join(dirname, x)))
                os.remove(dirname + '/' + dir_list[-1])
            continue





#1054
#1426
print('done')
