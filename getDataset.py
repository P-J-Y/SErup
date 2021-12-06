import numpy as np
from sunpy.net import attrs as a
from sunpy.net import Fido
from sunpy.coordinates import frames
from sunpy.coordinates.utils import GreatArc
import time, datetime
# from sunpy.net import hek2vso
# h2v = hek2vso.H2VClient()
from sunpy.net.helioviewer import HelioviewerClient
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sunpy.map import Map
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
from astropy.units import Quantity
from astropy.time import TimeDelta
import sunpy.data.sample
import sunpy.map
from sunpy.physics.differential_rotation import diff_rot, solar_rotate_coordinate
import pandas as pd
import imageio
import os
import json
# import cv2
import sunpy.coordinates.frames as f
from ffmpy3 import FFmpeg
import shutil
import cv2
from matplotlib.pyplot import imshow

# 1、活动区的时刻现在用的初始时刻，实际上应该考虑从初始到end ok
# 2、加入判断活动区是否在cme附近的判定 ok
# 3、如果事件在边缘，方框应该适当取小一些；不要直接放弃这些活动区，很多大活动区都在边缘
# 4、getFrame 中每个波段（图）都用一个封装的程序来实现，要哪个波段作为一个输入
# 5、从有要爆发的结构出现开始，取。看一下那些结构浮现到爆发的特征时间是多少。
# 6、ARresults、CMEresults这些能不能直接写成一个列表啊？
# 7、需要安装pillow ffmeg

# 给出时间范围内的活动区位置、时间范围等信息

def getCmes(tstart, tend, event_type='CE'):
    '''

    :param tstart: string
    :param tend: string
    :return: CEresult- a list. e.g. use CEresult['hek']['event_starttime'][0] to get the start time of the first CME event
    '''
    # h2v = hek2vso.H2VClient()
    CEresult = Fido.search(a.Time(tstart, tend), a.hek.EventType(event_type))
    return CEresult


'''
tstart = '2015/05/01 07:23:56'
tend = '2015/05/02 08:40:29'
CEresults = getCmes(tstart, tend)
'''


def getArs(CE_tstart, time_earlier1=30, time_earlier2=0, event_type='AR'):
    '''

    :param CEresults: result from getCmes
    :param CEidx: the index of the cme event
    :param time_earlier1: time earlier from cme's start time, to start searching for ARs
    :param time_earlier2: time earlier from cme's start time, to end searching for ARs
    :return: ARresult-a list contain all the ARs during time_earlier1 to time_earlier2 before the CME
            cache-CME info
    '''
    # assert CEidx < len(CEresults['hek']['event_endtime'])
    # fmt = "%Y-%m-%dT%H:%M:%S"
    # CE_x = CEresults['hek']['event_coord1'][CEidx]
    # CE_y = CEresults['hek']['event_coord2'][CEidx]
    # CE_tstart = datetime.datetime.strptime(CEresults['hek']['event_starttime'][CEidx], fmt)
    # CE_tend = datetime.datetime.strptime(CEresults['hek']['event_endtime'][CEidx], fmt)

    AR_tstart = CE_tstart - datetime.timedelta(minutes=time_earlier1)
    AR_tend = CE_tstart - datetime.timedelta(minutes=time_earlier2)
    ARresult = Fido.search(a.Time(AR_tstart, AR_tend), a.hek.EventType(event_type))
    # cache = (CE_x, CE_y, CE_tstart, CE_tend, AR_tstart, AR_tend)
    # cache = (CE_tstart, AR_tstart, AR_tend)
    return ARresult  # , cache


'''
tstart = '2015/05/01 07:23:56'
tend = '2015/05/02 08:40:29'
CEresults = getCmes(tstart, tend)
ARresults,cache = getArs(CEresults)
'''


def getDists(arInfo, CE_coord, CE_tstart):
    '''

    :param ARresults:
    :param cache:
    :return:
    '''
    # hv = HelioviewerClient()
    # CE_tstart, time1, time2 = cache
    dists = []
    for i in range(arInfo['ar_num']):
        AR_coord = arInfo["ar_coords"][i]

        theRotated_coord = solar_rotate_coordinate(AR_coord.transform_to(frames.Helioprojective), time=CE_tstart)
        theRotated_coord = theRotated_coord.transform_to(AR_coord.frame)
        dists.append(
            np.sqrt((theRotated_coord.lon.deg - CE_coord.lon.deg) ** 2 +
                    (theRotated_coord.lat.deg - CE_coord.lat.deg) ** 2)
        )
    # cache = (CE_x, CE_y, CE_tstart, CE_tend, CE_t,
    #         AR_xs, AR_ys, AR_tstarts, AR_tends, AR_ts, AR_coords, cframes,
    #         time1, time2)
    '''
    cache = (CE_x, CE_y, CE_tstart,
             AR_xs, AR_ys, AR_tstarts, AR_tends, AR_ts, AR_coords, cframes,
             time1, time2)
    '''
    return dists


'''
tstart = '2015/05/01 07:23:56'
tend = '2015/05/02 08:40:29'
CEresults = getCmes(tstart, tend)
ARresults,cache = getArs(CEresults)
dists,cache = getDists(ARresults,cache)
'''


# def getFrame(t, c1, c2, width, height, iscme=0, idx=0):
#     '''
#
#     :param t: ar time (python datetime object)
#     :param c1: ar center x in arcsec
#     :param c2: ar center y
#     :param width: ar region's width
#     :param height: ar region's height
#     :param iscme:
#     :return:
#     '''
#     hv = HelioviewerClient()
#     timeStrForFig = t.strftime("%Y%m%d%H%M%S")
#
#     def getMap(t, observatory="SDO", instrument="HMI", measurement="magnetogram"):
#         file = hv.download_jp2(t,
#                                observatory=observatory,
#                                instrument=instrument,
#                                measurement=measurement)
#         themap = Map(file)
#         fileName = "figure/{}/{}/{}/aia_ar_{}_{}.png".format(iscme, measurement, idx, timeStrForFig, idx)
#         return themap, fileName
#
#     hmiMap, hmiName = getMap(t, observatory="SDO", instrument="HMI", measurement="magnetogram")
#     aia171Map, aia171Name = getMap(t, observatory="SDO", instrument="AIA", measurement="171")
#     aia193Map, aia193Name = getMap(t, observatory="SDO", instrument="AIA", measurement="193")
#     aia94Map, aia94Name = getMap(t, observatory="SDO", instrument="AIA", measurement="94")
#     aia211Map, aia211Name = getMap(t, observatory="SDO", instrument="AIA", measurement="211")
#     aia304Map, aia304Name = getMap(t, observatory="SDO", instrument="AIA", measurement="304")
#     bottom_left = SkyCoord((c1 - width / 2) * u.arcsec,
#                            (c2 - height / 2) * u.arcsec,
#                            frame=hmiMap.coordinate_frame)  # 给出区域左下点的坐标（第一个参数是x坐标，第二个是y）
#     width = width * u.arcsec
#     height = height * u.arcsec
#
#     def getSubMap(aMap, bottom_left, width, height, savefileName):
#         thesubmap = aMap.submap
#         figure = plt.figure(frameon=False)
#         ax = plt.subplot(projection=thesubmap)
#         # Disable the axis
#         # ax1.set_axis_off()
#         # Plot the map.
#         # norm = aia_submap.plot_settings['norm']
#         # norm.vmin, norm.vmax = np.percentile(aia_submap.data, [1, 99.9])
#         ax.imshow(thesubmap.data,
#                   #           norm=norm,
#                   cmap=thesubmap.plot_settings['cmap'])
#         plt.savefig(savefileName)
#
#     getSubMap(hmiMap, bottom_left, width, height, hmiName)
#     getSubMap(aia94Map, bottom_left, width, height, aia94Name)
#     getSubMap(aia171Map, bottom_left, width, height, aia171Name)
#     getSubMap(aia193Map, bottom_left, width, height, aia193Name)
#     getSubMap(aia211Map, bottom_left, width, height, aia211Name)
#     getSubMap(aia304Map, bottom_left, width, height, aia304Name)
#
#     '''
#     figure1 = plt.figure(frameon=False)
#     ax1 = plt.subplot(projection=aia_submap)
#     # Disable the axis
#     # ax1.set_axis_off()
#
#     # Plot the map.
#     norm = aia_submap.plot_settings['norm']
#     norm.vmin, norm.vmax = np.percentile(aia_submap.data, [1, 99.9])
#     ax1.imshow(aia_submap.data,
#                norm=norm,
#                cmap=aia_submap.plot_settings['cmap'])
#     plt.savefig("figure/{}/193/{}/aia_ar_{}_{}.png".format(iscme,idx,timeStrForFig,idx))
#
#     figure2 = plt.figure(frameon=False)
#     ax2 = plt.subplot(projection=hmi_submap)
#     # Disable the axis
#     # ax2.set_axis_off()
#
#     # Plot the map. Since are not interested in the exact map coordinates, we can
#     # simply use :meth:`~matplotlib.Axes.imshow`.
#     # norm = hmi_submap.plot_settings['norm']
#     # norm.vmin, norm.vmax = np.percentile(hmi_submap.data, [1, 99.9])
#     ax2.imshow(hmi_submap.data,
#                #           norm=norm,
#                cmap=hmi_submap.plot_settings['cmap'])
#     plt.savefig("figure/{}/mag/{}/hmi_ar_{}_{}.png".format(iscme,idx,timeStrForFig,idx))
#     '''
#
#
# '''
# tstart = '2017/05/01 07:23:56'
# tend = '2017/05/10 08:40:29'
# CEresults = getCmes(tstart, tend)
# ARresults,cache = getArs(CEresults)
# dists,cache = getDists(ARresults,cache)
# CE_x, CE_y, CE_tstart, CE_tend, CE_t, \
#              AR_xs, AR_ys, AR_tstarts, \
#                 AR_tends, AR_ts, AR_coords, cmaps, \
#                  time1,time2 = cache
#
# ar_idx = 2
# AR_area = ARresults['hek']['area_raw'][ar_idx]
# AR_LL_x = ARresults['hek']['boundbox_c1ll'][ar_idx]
# AR_LL_y = ARresults['hek']['boundbox_c2ll'][ar_idx]
# AR_UR_x = ARresults['hek']['boundbox_c1ur'][ar_idx]
# AR_UR_y = ARresults['hek']['boundbox_c2ur'][ar_idx]
# AR_width = AR_UR_x-AR_LL_x
# AR_height = AR_UR_y-AR_LL_y
# getFrame(AR_ts[ar_idx],
#          AR_xs[ar_idx],
#          AR_ys[ar_idx],
#          max(AR_width,300),
#          max(AR_height,300),
#          iscme=0,
#          idx=ar_idx)
# '''
#
#
# def getFrames(AR_coord, tstart, tend, width=100, height=100, iscme=0, freq='min', ar_idx=0):
#     '''
#
#     :param x:
#     :param y:
#     :param t:
#     :param tstart:
#     :param tend:
#     :param width:
#     :param height:
#     :param iscme:
#     :return:
#     '''
#     ts = list(pd.date_range(tstart, tend, freq=freq))
#     for t in ts:
#         rotated_coord = solar_rotate_coordinate(AR_coord, time=t)
#         getFrame(t,
#                  rotated_coord.Tx.arcsec,
#                  rotated_coord.Ty.arcsec,
#                  width,
#                  height,
#                  iscme=iscme,
#                  idx=ar_idx)
#
#
# '''
# tstart = '2017/07/01 07:23:56'
# tend = '2017/07/10 08:40:29'
# CEresults = getCmes(tstart, tend)
# ARresults,cache = getArs(CEresults,CEidx=0)
# dists,cache = getDists(ARresults,cache)
# CE_x, CE_y, CE_tstart, \
#             AR_xs, AR_ys, AR_tstarts, \
#             AR_tends, AR_ts, AR_coords, cmaps, \
#             time1,time2 = cache
#
# #min_idx = dists.index(min(dists))
# #ar_idx = min_idx
# #print(min_idx,min(dists))
#
# AR_areas = ARresults['hek']['area_raw']
# AR_LL_xs = ARresults['hek']['boundbox_c1ll']
# AR_LL_ys = ARresults['hek']['boundbox_c2ll']
# AR_UR_xs = ARresults['hek']['boundbox_c1ur']
# AR_UR_ys = ARresults['hek']['boundbox_c2ur']
# AR_widths = AR_UR_xs-AR_LL_xs
# AR_heights = AR_UR_ys-AR_LL_ys
#
# ar_idx = 2
# getFrames(AR_coords[ar_idx],
#           max(AR_tstarts[ar_idx],time1),
#           min(AR_tends[ar_idx],time2),
#           width=AR_widths[ar_idx],
#           height=AR_heights[ar_idx],
#           iscme=0,
#           freq='1min',
#           ar_idx=ar_idx)
# '''


def deleteSmallARs(ARresults, threshold=(100, 6), fmt="%Y-%m-%dT%H:%M:%S"):
    '''

    :param ARresults:
    :param cache: from gerArs
    :param threshold: (100,6) 100arcsec for projection 6deg for HGS
    :param fmt: 这个是HEK给出的ARresults中时间的格式
    :return:
    '''

    # 把scale足够大的提取出来
    # 如果 |x+-width/2|>900  width = 2*min(|900+-x|)
    AR_coordsyss = ARresults['hek']['event_coordsys']
    AR_coordunitstr = ARresults['hek']['event_coordunit']
    coordunits = {"UTC-HPC-TOPO": u.arcsec, "UTC-HGS-TOPO": u.deg, "UTC-HGC-TOPO": u.deg}
    thresholds = {"UTC-HPC-TOPO": threshold[0], "UTC-HGS-TOPO": threshold[1], "UTC-HGC-TOPO": threshold[1]}
    AR_coordunits = [coordunits[AR_coordsyss[i]] for i in range(len(AR_coordsyss))]
    AR_xs = np.multiply(ARresults['hek']['event_coord1'], AR_coordunits)
    AR_ys = np.multiply(ARresults['hek']['event_coord2'], AR_coordunits)
    AR_LL_xs = np.multiply(ARresults['hek']['boundbox_c1ll'], AR_coordunits)
    AR_LL_ys = np.multiply(ARresults['hek']['boundbox_c2ll'], AR_coordunits)
    AR_UR_xs = np.multiply(ARresults['hek']['boundbox_c1ur'], AR_coordunits)
    AR_UR_ys = np.multiply(ARresults['hek']['boundbox_c2ur'], AR_coordunits)
    AR_widths = AR_UR_xs - AR_LL_xs
    AR_heights = AR_UR_ys - AR_LL_ys
    AR_scales = np.min([AR_widths, AR_heights], axis=0)
    AR_thresholds = [thresholds[AR_coordsyss[i]] for i in range(len(AR_coordsyss))]
    AR_thresholds = np.multiply(AR_thresholds, AR_coordunits)
    bigArIdx = (np.array(AR_scales) >= np.array(AR_thresholds)) & (
                np.array(ARresults['hek']['frm_identifier']) == "HMI Active Region Patch")
    assert np.sum(
        bigArIdx) != 0, "there are only small ARs during this period, choose another event or choose a smaller threshold 没有合适的活动区，换下一个CME，别浪费时间"
    AR_tstarts = np.array(
        [datetime.datetime.strptime(ARresults['hek']['event_starttime'][i], fmt) for i in range(len(AR_LL_xs))])
    AR_tends = np.array(
        [datetime.datetime.strptime(ARresults['hek']['event_endtime'][i], fmt) for i in range(len(AR_LL_xs))])
    AR_ts = AR_tstarts + np.divide(np.subtract(AR_tends, AR_tstarts), 2)
    arInfo = {"ar_xs": AR_xs[bigArIdx],
              "ar_ys": AR_ys[bigArIdx],
              "ar_tstarts": AR_tstarts[bigArIdx],
              "ar_tends": AR_tends[bigArIdx],
              "ar_widths": AR_widths[bigArIdx],
              "ar_heights": AR_heights[bigArIdx],
              "ar_scales": AR_scales[bigArIdx],
              "ar_ts": AR_ts[bigArIdx],
              "ar_num": sum(bigArIdx),
              "ar_coordsyss": AR_coordsyss[bigArIdx]
              }
    # hv = HelioviewerClient()
    # cframes = []
    AR_coords = []
    for i in range(arInfo["ar_num"]):
        # cfile = hv.download_jp2(arInfo["ar_ts"][i], observatory="SDO", instrument="HMI", measurement="magnetogram")
        # cmap = Map(cfile)
        # cframe = cmap.coordinate_frame
        # cframes.append(cframe)
        if arInfo["ar_coordsyss"][i] == "UTC-HPC-TOPO":
            theAR_coord = SkyCoord(arInfo["ar_xs"][i], arInfo["ar_ys"][i],
                                   frame=frames.Helioprojective,
                                   obstime=arInfo["ar_ts"][i],
                                   observer="earth",
                                   )
        elif arInfo["ar_coordsyss"][i] == "UTC-HGS-TOPO":
            theAR_coord = SkyCoord(arInfo["ar_xs"][i], arInfo["ar_ys"][i],
                                   frame="heliographic_stonyhurst",
                                   obstime=arInfo["ar_ts"][i],
                                   observer="earth")
        # theAR_coord = SkyCoord(arInfo["ar_xs"][i] * u.arcsec, arInfo["ar_ys"][i] * u.arcsec, frame=cframe)
        AR_coords.append(theAR_coord)
    # arInfo["ar_cframes"] = cframes
    arInfo["ar_coords"] = AR_coords
    return arInfo


'''tstart = '2017/07/01 07:23:56'
tend = '2017/07/10 08:40:29'
CEresults = getCmes(tstart, tend)
ARresults,cache = getArs(CEresults,CEidx=0)
arInfo = deleteSmallARs(ARresults)
CE_x, CE_y, CE_tstart, CE_tend,AR_tstart,AR_tend = cache'''


# ARs  活动区列表的列表,包含每个CME（按照一定的序号）所对应的大活动区
# 下面的函数先实现一个CME的获取和标记
def getCmeSunWithArIndex(cmeTstart,
                         cmeCoord,
                         arInfo,
                         minidx,
                         time_earlier1=20,
                         time_earlier2=-20,
                         freq='min',
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

    def getMap(t, observatory=observatory, instrument=instrument, measurement=measurement):
        file = hv.download_jp2(t,
                               observatory=observatory,
                               instrument=instrument,
                               measurement=measurement)
        themap = Map(file)
        timeStrForFig = t.strftime("%Y%m%d%H%M%S")
        fileName = "figure/cme/aia_{}_{}.png".format(measurement, timeStrForFig)
        return themap, fileName

    def getCmeSunFrame(t,
                       arInfo,
                       idx,
                       cmeCoord,
                       minidx=minidx,
                       observatory=observatory,
                       instrument=instrument,
                       measurement=measurement,
                       mapFileDir=mapFileDir,
                       submapFileDir=submapFileDir):
        themap, fileName = getMap(t, observatory=observatory, instrument=instrument, measurement=measurement)
        timeStrForFig = t.strftime("%Y-%m-%d %H:%M:%S")
        # fig = plt.figure()
        # Provide the Map as a projection, which creates a WCSAxes object
        ax = plt.subplot(projection=themap)
        # ax = plt.plot()
        im = themap.plot()
        # Prevent the image from being re-scaled while overplotting.
        ax.set_autoscale_on(False)
        for i in range(arInfo["ar_num"]):
            if t > arInfo["ar_tends"][i] or t < arInfo["ar_tstarts"][i]:
                continue
            theAR_coord = arInfo["ar_coords"][i]
            transAR_coord = theAR_coord.transform_to(themap.coordinate_frame)
            # transAR_coord = theAR_coord.transform_to(frames.Helioprojective)
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
            # ax.scatter_coord(theRotated_coord.Tx.arcsec,theRotated_coord.Ty.arcsec)

        # Set title.
        if t < cmeTstart:
            ax.set_title(timeStrForFig)
        else:
            ax.set_title("{} CME".format(timeStrForFig))
        ax.plot_coord(cmeCoord, "r+", label="cme")
        plt.legend()
        # plt.savefig(fileName)
        # plt.show()
        # plt.savefig("figure/cme/{}.png".format(idx),dpi=600)
        plt.savefig("{}{}.png".format(mapFileDir, idx), dpi=600)

        #get submap
        cmeArCoord = arInfo["ar_coords"][minidx]
        theRotated_arc = solar_rotate_coordinate(cmeArCoord.transform_to(frames.Helioprojective),
                                                 time=t).transform_to(cmeArCoord.frame)
        if np.isnan(theRotated_arc.lon.value):
            theRotated_arc = cmeArCoord
        width = arInfo["ar_widths"][minidx]
        height = arInfo["ar_heights"][minidx]
        bottom_left = SkyCoord(theRotated_arc.lon - width / 2,
                               theRotated_arc.lat - height / 2,
                               frame=cmeArCoord.frame)
        #theRotated_bl = solar_rotate_coordinate(bottom_left, time=t)

        thesubmap = themap.submap(bottom_left,
                                  width=width,
                                  height=height)
        figure = plt.figure(frameon=False)
        ax = plt.subplot(projection=thesubmap)
        # Disable the axis
        ax.set_axis_off()
        # Plot the map.
        # norm = aia_submap.plot_settings['norm']
        # norm.vmin, norm.vmax = np.percentile(aia_submap.data, [1, 99.9])
        ax.imshow(thesubmap.data,
                  #           norm=norm,
                  cmap=thesubmap.plot_settings['cmap'])
        ax.set_title(timeStrForFig)
        plt.savefig("{}{}.png".format(submapFileDir, idx), dpi=600)

    # ims = []
    fig = plt.figure()
    tstart = cmeTstart - datetime.timedelta(minutes=time_earlier1)
    tend = cmeTstart - datetime.timedelta(minutes=time_earlier2)
    ts = list(pd.date_range(tstart, tend, freq=freq))
    idx = 0
    for t in ts:
        getCmeSunFrame(t, arInfo, idx, cmeCoord, observatory=observatory, instrument=instrument,
                       measurement=measurement)
        # ims.append(im)
        idx += 1
    # ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
    # plt.show()
    # ani.save("figure/cme/movie.gif", writer='pillow')

    # Writer = animation.writers['ffmpeg']  # 需安装ffmpeg
    # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("figure/cme/movie.mp4", writer=writer)


def breakCoordStr(coordStr):
    # S9E***
    if coordStr[2].isalpha():
        coord2Str = coordStr[0:2]
        coord1Str = coordStr[2:]
    # N81E***
    elif coordStr[3].isalpha():
        coord2Str = coordStr[0:3]
        coord1Str = coordStr[3:]
    else:
        return None
    return (coord2Str, coord1Str)


def getCmeCoord(coordStrs):
    '''
    N正S负，W正E负
    :param coordStrs:
    :return:
    '''
    if coordStrs is None:
        return None
    coord2Str, coord1Str = coordStrs
    if coord2Str[0] == 'S':
        coord2 = -float(coord2Str[1:])
    elif coord2Str[0] == 'N':
        coord2 = float(coord2Str[1:])
    else:
        raise ValueError("coord2 not a coord start with S or N!", coord2Str)

    if coord1Str[0] == 'E':
        coord1 = -float(coord1Str[1:])
    elif coord1Str[0] == 'W':
        coord1 = float(coord1Str[1:])
    else:
        raise ValueError("coord1 not a coord start with E or W!", coord1Str)

    coord = SkyCoord(coord1 * u.deg,
                     coord2 * u.deg,
                     frame="heliographic_stonyhurst",
                     # obstime=arInfo["ar_ts"][i],
                     # observer="earth"
                     )
    return coord


def arCmeMatch(dists, arInfo):
    minDist = min(dists)
    idx = dists.index(minDist)
    # theScale = arInfo['ar_scales'][idx]
    matchFlag = int(np.floor_divide(minDist, arInfo['ar_scales'][idx].value))
    return minDist, idx, matchFlag


def getArInfoWithCmeInfo(cmeinfo,
                         time_earlier1=60,
                         time_earlier2=0,
                         ar_threshold=(100, 6),
                         fmt="%Y-%m-%dT%H:%MZ"):
    tstart = datetime.datetime.strptime(cmeinfo["startTime"], fmt)
    ARresults = getArs(tstart, time_earlier1=time_earlier1, time_earlier2=time_earlier2)
    assert ARresults.file_num>0 , "no ar founded at all"
    arInfo = deleteSmallARs(ARresults, threshold=ar_threshold)
    cache = tstart
    return arInfo, cache


def getCmeFilm(cmeidx,
               cmeInfo,
               arInfo,
               cache,
               time_earlier1=90,
               time_earlier2=-20,
               freq='2min',
               film_path = os.getcwd() + "\\figure\\film\\",
               mustmatch=True,
               ):
    '''

    :param cmeidx:
    :param cmeInfo:
    :param arInfo:
    :param fmt: cme数据的时间格式
    :param time_earlier1: 画图时候提前的时间 min
    :param time_earlier2: 画图结束时间（先对于cme开始），如果要画到cme之后就设置成负的
    :param freq: 画图间隔频率
    :return:
    '''

    def getMeasurementFilm(tstart,
                           arInfo,
                           CE_coord,
                           observatory,
                           instrument,
                           measurement,
                           fileDir,
                           minidx,
                           time_earlier1=time_earlier1,
                           time_earlier2=time_earlier2,
                           freq=freq,
                           cmeidx=cmeidx
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
            os.mkdir(fileDir + "submapcache")
        except FileExistsError:
            shutil.rmtree(fileDir + "mapcache")
            shutil.rmtree(fileDir + "submapcache")
            os.mkdir(fileDir + "mapcache")
            os.mkdir(fileDir + "submapcache")

        getCmeSunWithArIndex(tstart,
                             CE_coord,
                             arInfo,
                             minidx,
                             time_earlier1=time_earlier1,
                             time_earlier2=time_earlier2,
                             freq=freq,
                             observatory=observatory,
                             instrument=instrument,
                             measurement=measurement,
                             mapFileDir=fileDir + "mapcache\\",
                             submapFileDir=fileDir + "submapcache\\")
        ffin1 = FFmpeg(inputs={fileDir + "mapcache\\" + '%d.png': '-y -r 4'},
                      outputs={fileDir + "CME{}Film{}.mp4".format(measurement, cmeidx): None})
        # print(ffin.cmd)
        ffin1.run()
        ffin2 = FFmpeg(inputs={fileDir + "submapcache\\" + '%d.png': '-y -r 4'},
                      outputs={fileDir + "AR{}Film{}.mp4".format(measurement, cmeidx): None})
        # print(ffin.cmd)
        ffin2.run()


        shutil.rmtree(fileDir + "mapcache")
        shutil.rmtree(fileDir + "submapcache")
        # os.mkdir(pic_path)

    tstart = cache
    CeCoordStr = cmeInfo["sourceLocation"]
    CE_coord = getCmeCoord(breakCoordStr(CeCoordStr))
    dists = getDists(arInfo, CE_coord, tstart)
    minDist, minidx, matchFlag = arCmeMatch(dists, arInfo)
    print("cme ar idx = {}".format(minidx))
    print("matchFlag={}".format(matchFlag))
    if matchFlag>=1 and mustmatch:
        raise ValueError("matchFlag>=1, cme's source ar is not found, try next CEidx",matchFlag)

    # film_name = current_name + "\\figure\\film\cme{}film{}.mp4".format(measurement, cmeidx)

    #193
    observatory = "SDO"
    instrument = "AIA"
    # measurement="magnetogram"
    measurement = "193"
    getMeasurementFilm(tstart,
                       arInfo,
                       CE_coord,
                       observatory,
                       instrument,
                       measurement,
                       film_path,
                       minidx,
                       time_earlier1=time_earlier1,
                       time_earlier2=time_earlier2,
                       freq=freq,
                       cmeidx=cmeidx
                       )
    #HMI
    observatory = "SDO"
    instrument = "HMI"
    measurement="magnetogram"
    #measurement = "193"
    getMeasurementFilm(tstart,
                       arInfo,
                       CE_coord,
                       observatory,
                       instrument,
                       measurement,
                       film_path,
                       minidx,
                       time_earlier1=time_earlier1,
                       time_earlier2=time_earlier2,
                       freq=freq,
                       cmeidx=cmeidx
                       )


'''
cmelistpath='data/cmelist.json'
file = open(cmelistpath,'r',encoding='utf-8')
cmelist = json.load(file)
CEidx=5
theCmeInfo = cmelist[CEidx]

#tstart = "2015/10/22 02:00:00"
fmt = "%Y-%m-%dT%H:%MZ" #cme数据的时间格式
#tstart = datetime.datetime.strptime("2021-04-20 00:12",fmt)
#tend = "2015/10/22 04:00:00"
#CeCoordStr = "S24E23"

observatory="SDO"
instrument="AIA"
#measurement="magnetogram"
measurement="193"
ar_threshold=(100,6)
#数据集从https://kauai.ccmc.gsfc.nasa.gov/DONKI/search/ 这个网站获取。选择给出了源区坐标的CME。 然后画出这个CME之前1.5-2小时的图，就能看到源区相应位置的喷流
#选择比较大的？ 用源区坐标和AR匹配
theCmeInfo = cmelist[CEidx]
tstart = datetime.datetime.strptime(theCmeInfo["startTime"],fmt)
CeCoordStr = theCmeInfo["sourceLocation"]




CE_coord = getCmeCoord(breakCoordStr(CeCoordStr))

#CEresults = getCmes(tstart, tend)
ARresults = getArs(tstart,time_earlier1=60,time_earlier2=0)
arInfo = deleteSmallARs(ARresults,threshold=ar_threshold)
#CE_tstart, AR_tstart,AR_tend = cache
dists = getDists(arInfo, CE_coord, tstart)
minDist,minidx,matchFlag = arCmeMatch(dists,arInfo)
getCmeSunWithArIndex(tstart,
                     CE_coord,
                    arInfo,
                    time_earlier1=90,
                    time_earlier2=-20,
                    freq='2min',
                    observatory=observatory,
                    instrument=instrument,
                    measurement=measurement)

current_name = os.getcwd()
#gif_name = "E:\GithubLocal\SErup\\figure\gif\cme{}gif{}.gif".format(measurement,CEidx)
#film_name = "E:\GithubLocal\SErup\\figure\\film\cme{}film{}.mp4".format(measurement,CEidx)
film_name = current_name+"\\figure\\film\cme{}film{}.mp4".format(measurement,CEidx)
#film_name = "E:\GithubLocal\SErup\\figure\\film\\test.mp4"
#pic_path = "E:\GithubLocal\SErup\\figure\cme\\"
pic_path = current_name+"\\figure\cme\\"
'''
'''
可以用cv2来合成视频，好像会快一点，但是会限制必须输入图片大小，不太方便（还需要注意视频格式必须和编码器匹配）
images = os.listdir(pic_path)
images.sort(key=lambda x: int(x.split('.')[0]))
fps = 4          # 视频帧率
size = (3840, 2880) # 需要转为视频的图片的尺寸
videoWriter = cv2.VideoWriter(film_name,
                              #-1,
                              cv2.VideoWriter_fourcc(*'DIVX'),
                              fps,
                              size,isColor=True)

for f in images:
    frame = cv2.imread(pic_path + f)
    videoWriter.write(frame)
videoWriter.release()
cv2.destroyAllWindows()
#imageio.mimwrite(gif_name, frames, 'GIF', duration=0.5)
'''
'''
ffin = FFmpeg(inputs={pic_path+'%d.png': '-y -r 4'},
              outputs={film_name: None})
#print(ffin.cmd)
ffin.run()


import shutil
shutil.rmtree(pic_path)
os.mkdir(pic_path)
'''
#####################get films###################
'''
cmelistpath = 'data/cmelist.json'
file = open(cmelistpath, 'r', encoding='utf-8')
cmelist = json.load(file)
ar_search_t1 = 60
ar_search_t2 = 20
film_t1 = 300
film_t2 = 0
freq = '2min'
ar_threshold = (100,6)
film_path = os.getcwd() + "\\figure\\longfilm\\"
#for CEidx in range(521,len(cmelist)):
for CEidx in [2,15]:
    theCmeInfo = cmelist[CEidx]

    try:
        theArInfo, cache = getArInfoWithCmeInfo(theCmeInfo,
                                                time_earlier1=ar_search_t1,
                                                time_earlier2=ar_search_t2,
                                                ar_threshold=ar_threshold,
                                                fmt="%Y-%m-%dT%H:%MZ")
        getCmeFilm(CEidx,
                   theCmeInfo,
                   theArInfo,
                   cache,
                   time_earlier1=film_t1,
                   time_earlier2=film_t2,
                   freq=freq,
                   film_path=film_path
                   )
    except ValueError:
        print("CMEidx: {} cme coordstr error, or no AR found. goto next CEidx".format(CEidx))
        continue
    except AssertionError:
        print("CMEidx: {} 可能没找到合适的AR，换到下一个CME".format(CEidx))
        continue
'''

#####################get images########################
# ARs  活动区列表的列表,包含每个CME（按照一定的序号）所对应的大活动区
# 下面的函数先实现一个AR图片的获取和标记
def getArArray(posres,
               negres,
               cmeTstart,
                arInfo,
                minidx,
                time_earlier1=60,
                time_earlier2=30,
                freq='5min',
                observatorys=("SDO", "SDO", "SDO", "SDO", "SDO", "SDO",),
                instruments = ("AIA", "AIA", "AIA", "AIA", "AIA", "HMI"),
                measurements = ("94", "171", "193", "211", "304", "magnetogram"),
                imgSize=256,
                ):  # cme时间：根据CME catalog；ar信息：根据HEK搜索出比较大的,最好是直接输入AR的完整信息；输出：gif+活动区信息列表
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
                observatorys,
                instruments,
                measurements,
                Nchannels,
                posres,
                negres,
                minidx,
                imgSize = imgSize):
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
            # get submaps
            for aridx in aridxs:

                cmeArCoord = arInfo["ar_coords"][aridx]
                theRotated_arc = solar_rotate_coordinate(cmeArCoord.transform_to(frames.Helioprojective),
                                                         time=t).transform_to(cmeArCoord.frame)
                if np.isnan(theRotated_arc.lon.value):
                    theRotated_arc = cmeArCoord
                width = arInfo["ar_widths"][aridx]
                height = arInfo["ar_heights"][aridx]
                bottom_left = SkyCoord(theRotated_arc.lon - width / 2,
                                       theRotated_arc.lat - height / 2,
                                       frame=cmeArCoord.frame)
                # theRotated_bl = solar_rotate_coordinate(bottom_left, time=t)
                aData = np.zeros((imgSize, imgSize, Nchannels))
                for channelIdx in range(Nchannels):
                    thesubmap = themaps[channelIdx].submap(bottom_left,
                                                           width=width,
                                                           height=height)
                    thedata = thesubmap.data
                    thedata = pad2square(thedata)
                    dst_size = (imgSize, imgSize)
                    thedata = cv2.resize(thedata, dst_size, interpolation=cv2.INTER_AREA)
                    aData[:, :, channelIdx] = thedata
                if aridx == minidx:
                    posres.append(aData)
                else:
                    negres.append(aData)
        else:
            print("no ar at this time")



    tstart = cmeTstart - datetime.timedelta(minutes=time_earlier1)
    tend = cmeTstart - datetime.timedelta(minutes=time_earlier2)
    ts = list(pd.date_range(tstart, tend, freq=freq))
    Nchannels = len(measurements)
    for t in ts:
        getSubmap(t,arInfo,observatorys,instruments,measurements,Nchannels,posres,negres,minidx,
                          imgSize=imgSize)


if __name__ == '__main__':
    cmelistpath = 'data/cmelist.json'
    file = open(cmelistpath, 'r', encoding='utf-8')
    cmelist = json.load(file)
    ar_search_t1 = 80
    ar_search_t2 = 40
    data_t1 = 60 * 24
    data_t2 = 60
    freq = '60min'
    ar_threshold = (100, 6)

    POS = []
    NEG = []

    for CEidx in range(100, 200):
        theCmeInfo = cmelist[CEidx]

        try:
            theArInfo, cache = getArInfoWithCmeInfo(theCmeInfo,
                                                    time_earlier1=ar_search_t1,
                                                    time_earlier2=ar_search_t2,
                                                    ar_threshold=ar_threshold,
                                                    fmt="%Y-%m-%dT%H:%MZ")
            CEtstart = cache
            CeCoordStr = theCmeInfo["sourceLocation"]
            CE_coord = getCmeCoord(breakCoordStr(CeCoordStr))
            dists = getDists(theArInfo, CE_coord, CEtstart)
            minDist, minidx, matchFlag = arCmeMatch(dists, theArInfo)
            print("cme ar idx = {}".format(minidx))
            print("matchFlag={}".format(matchFlag))
            if matchFlag >= 1:
                raise ValueError("matchFlag>=1, cme's source ar is not found, try next CEidx", matchFlag)
            getArArray(POS,
                       NEG,
                       CEtstart,
                       theArInfo,
                       minidx,
                       time_earlier1=data_t1,
                       time_earlier2=data_t2,
                       freq=freq,
                       observatorys=("SDO", "SDO", "SDO", "SDO", "SDO", "SDO", "SDO",),
                       instruments=("AIA", "AIA", "AIA", "AIA", "AIA", "HMI", "HMI"),
                       measurements=("94", "171", "193", "211", "304", "magnetogram", 'continuum'),
                       imgSize=256,
                       )

        except ValueError:
            print("CMEidx: {} cme coordstr error, or no AR found. goto next CEidx".format(CEidx))
            continue
        except AssertionError:
            print("CMEidx: {} 可能没找到合适的AR，换到下一个CME".format(CEidx))
            continue
        except RuntimeError:
            print("CMEidx: {} Don't know what happened, maybe it's the internet?".format(CEidx))
            np.savez("data/data24hr_1hr/base1.npz", pos=POS, neg=NEG)
            continue

    filename = 'data/data24hr_1hr/dataset1.npz'
    np.savez(filename, pos=POS, neg=NEG)
    print('dd')

# CEidx = 3
# theCmeInfo = cmelist[CEidx]
# theArInfo, cache = getArInfoWithCmeInfo(theCmeInfo,
#                                         time_earlier1=ar_search_t1,
#                                         time_earlier2=ar_search_t2,
#                                         ar_threshold=ar_threshold,
#                                         fmt="%Y-%m-%dT%H:%MZ")
# tstart = cache
# CeCoordStr = theCmeInfo["sourceLocation"]
# CE_coord = getCmeCoord(breakCoordStr(CeCoordStr))
# dists = getDists(theArInfo, CE_coord, tstart)
# minDist, minidx, matchFlag = arCmeMatch(dists, theArInfo)
# print("cme ar idx = {}".format(minidx))
# print("matchFlag={}".format(matchFlag))
# if matchFlag >= 1:
#     raise ValueError("matchFlag>=1, cme's source ar is not found, try next CEidx", matchFlag)

# 加入其他不等于minidx的情况（但是输出的就是neg）
# observatorys = ("SDO","SDO","SDO","SDO","SDO","SDO",)
# instruments = ("AIA","AIA","AIA","AIA","AIA","HMI")
# measurements = ("94","171","193","211","304","magnetogram")



# getArArray(POS,
#            tstart,
#            CE_coord,
#            theArInfo,
#            minidx,
#            time_earlier1=data_t1,
#            time_earlier2=data_t2,
#            freq=freq,
#            # observatory="SDO",
#            # instrument="AIA",
#            # measurement="193",
#            )

'''
#CEidx = 59
theCmeInfo = cmelist[CEidx]
theArInfo, cache = getArInfoWithCmeInfo(theCmeInfo,
                                        time_earlier1=60,
                                        time_earlier2=0,
                                        ar_threshold=(100, 6),
                                        fmt="%Y-%m-%dT%H:%MZ")
getCmeFilm(CEidx,
           theCmeInfo,
           theArInfo,
           cache,
           time_earlier1=90,
           time_earlier2=-20,
           freq='5min',
           film_path=os.getcwd()+"\\figure\\film\\"
           )
'''
# 人工判断，哪个活动区
# 需要的活动区信息：时间、中心位置、边界的4个坐标
# 需要的CME信息：时间（完全可以根据CME catalog获取）
# 也可以先用HEK的CME数据，根据时间来区分不同的CME
# 之后for 循环取所有的CMEs，并根据标记的cmeARidx，把cmeAR的图片放进1，其他AR的图片放进0

# list：CE_starttimes  CE_starttimes[i]对应于下面列表中的AR_infos[i]
# list: AR_infos   AR_infos[i][j] = (AR_st,AR_et,AR_x,AR_y,AR_t,AR_width,AR_height,AR_scale)

'''
问题是这个gif分辨率很低，最好能弄高清一点，方便识别CME爆发源
或者直接存成视频是不是高清一点？
如果还是判断不了可能还要结合lasco去观测。
或者直接上CME catalog找比较大的CME，因为比较小的cme可能很难判断源区。
'''
