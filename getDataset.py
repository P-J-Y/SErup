import numpy as np
from sunpy.net import attrs as a
from sunpy.net import Fido
from sunpy.coordinates import frames
from sunpy.coordinates.utils import GreatArc
import time,datetime
#from sunpy.net import hek2vso
#h2v = hek2vso.H2VClient()
from sunpy.net.helioviewer import HelioviewerClient
import matplotlib.pyplot as plt
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


# 1、活动区的时刻现在用的初始时刻，实际上应该考虑从初始到end ok
# 2、加入判断活动区是否在cme附近的判定 ok
# 3、如果事件在边缘，那么画一个方框就会报错，所以我限制了事件的位置不能太偏边缘
# 4、getFrame 中每个波段（图）都用一个封装的程序来实现，要哪个波段作为一个输入

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

def getArs(CEresults,CEidx = 0,time_earlier1=30,time_earlier2=0,event_type='AR',fmt = "%Y-%m-%dT%H:%M:%S"):
    '''
    
    :param CEresults: result from getCmes 
    :param CEidx: the index of the cme event
    :param time_earlier1: time earlier from cme's start time, to start searching for ARs
    :param time_earlier2: time earlier from cme's start time, to end searching for ARs
    :return: ARresult-a list contain all the ARs during time_earlier1 to time_earlier2 before the CME
            cache-CME info
    '''
    assert CEidx < len(CEresults['hek']['event_endtime'])

    CE_x = CEresults['hek']['event_coord1'][CEidx]
    CE_y = CEresults['hek']['event_coord2'][CEidx]
    CE_tstart = datetime.datetime.strptime(CEresults['hek']['event_starttime'][CEidx], fmt)
    CE_tend = datetime.datetime.strptime(CEresults['hek']['event_endtime'][CEidx],fmt)

    AR_tstart = CE_tstart - datetime.timedelta(minutes=time_earlier1)
    AR_tend = CE_tstart - datetime.timedelta(minutes=time_earlier2)
    ARresult = Fido.search(a.Time(AR_tstart, AR_tend), a.hek.EventType(event_type))
    cache = (CE_x, CE_y, CE_tstart, CE_tend,AR_tstart,AR_tend)
    ARresults = []
    return ARresult,cache
'''
tstart = '2015/05/01 07:23:56'
tend = '2015/05/02 08:40:29'
CEresults = getCmes(tstart, tend)
ARresults,cache = getArs(CEresults)
'''

def getDists(ARresults,cache,fmt = "%Y-%m-%dT%H:%M:%S"):
    '''
    
    :param ARresults: 
    :param cache: 
    :return: 
    '''
    hv = HelioviewerClient()
    CE_x, CE_y, CE_tstart, CE_tend,time1,time2 = cache
    CE_t = CE_tstart+(CE_tend-CE_tstart)/2
    AR_tstarts = [datetime.datetime.strptime(i,fmt) for i in ARresults['hek']['event_starttime']]
    AR_tends = [datetime.datetime.strptime(i,fmt) for i in ARresults['hek']['event_endtime']]
    AR_ts = AR_tstarts+np.divide(np.subtract(AR_tends,AR_tstarts),2)
    AR_xs = ARresults['hek']['event_coord1']
    AR_ys = ARresults['hek']['event_coord2']
    AR_coords = []
    dists = []
    cframes = []
    for i in range(len(AR_xs)):
        cfile = hv.download_jp2(AR_ts[i], observatory="SDO", instrument="HMI", measurement="magnetogram")
        cmap = Map(cfile)
        cframe = cmap.coordinate_frame
        cframes.append(cframe)
        theAR_coord = SkyCoord(AR_xs[i] * u.arcsec, AR_ys[i] * u.arcsec, frame=cframe)
        theRotated_coord = solar_rotate_coordinate(theAR_coord , time=CE_t)
        AR_coords.append(theAR_coord)
        dists.append(
            np.sqrt((theRotated_coord.Tx.arcsec-CE_x)**2+
                     (theRotated_coord.Ty.arcsec-CE_y)**2)
        )
    cache = (CE_x, CE_y, CE_tstart, CE_tend, CE_t,
             AR_xs, AR_ys, AR_tstarts, AR_tends, AR_ts,AR_coords,cframes,
             time1,time2)
    return dists,cache
'''
tstart = '2015/05/01 07:23:56'
tend = '2015/05/02 08:40:29'
CEresults = getCmes(tstart, tend)
ARresults,cache = getArs(CEresults)
dists,cache = getDists(ARresults,cache)
'''

def getFrame(t,c1,c2,width,height,iscme=0,idx=0):
    '''
    
    :param t: ar time (python datetime object)
    :param c1: ar center x in arcsec
    :param c2: ar center y
    :param width: ar region's width
    :param height: ar region's height
    :param iscme: 
    :return: 
    '''
    hv = HelioviewerClient()
    timeStrForFig = t.strftime("%Y%m%d%H%M%S")

    def getMap(t,observatory="SDO", instrument="HMI", measurement="magnetogram"):
        file = hv.download_jp2(t,
                               observatory=observatory,
                               instrument=instrument,
                               measurement=measurement)
        themap = Map(file)
        fileName = "figure/{}/{}/{}/aia_ar_{}_{}.png".format(iscme, measurement, idx, timeStrForFig, idx)
        return themap,fileName

    hmiMap,hmiName = getMap(t,observatory="SDO", instrument="HMI", measurement="magnetogram")
    aia171Map, aia171Name = getMap(t, observatory="SDO", instrument="AIA", measurement="171")
    aia193Map,aia193Name = getMap(t,observatory="SDO", instrument="AIA", measurement="193")
    aia94Map,aia94Name = getMap(t, observatory="SDO", instrument="AIA", measurement="94")
    aia211Map,aia211Name = getMap(t, observatory="SDO", instrument="AIA", measurement="211")
    aia304Map,aia304Name = getMap(t, observatory="SDO", instrument="AIA", measurement="304")
    bottom_left = SkyCoord((c1 - width / 2) * u.arcsec,
                           (c2 - height / 2) * u.arcsec,
                           frame=hmiMap.coordinate_frame)  # 给出区域左下点的坐标（第一个参数是x坐标，第二个是y）
    width = width * u.arcsec
    height = height * u.arcsec

    def getSubMap(aMap,bottom_left,width,height,savefileName):
        thesubmap = aMap.submap(bottom_left, width=width, height=height)
        figure = plt.figure(frameon=False)
        ax = plt.subplot(projection=thesubmap)
        # Disable the axis
        # ax1.set_axis_off()
        # Plot the map.
        #norm = aia_submap.plot_settings['norm']
        #norm.vmin, norm.vmax = np.percentile(aia_submap.data, [1, 99.9])
        ax.imshow(thesubmap.data,
        #           norm=norm,
                   cmap=thesubmap.plot_settings['cmap'])
        plt.savefig(savefileName)
    getSubMap(hmiMap,bottom_left,width,height,hmiName)
    getSubMap(aia94Map, bottom_left, width, height, aia94Name)
    getSubMap(aia171Map, bottom_left, width, height, aia171Name)
    getSubMap(aia193Map, bottom_left, width, height, aia193Name)
    getSubMap(aia211Map, bottom_left, width, height, aia211Name)
    getSubMap(aia304Map, bottom_left, width, height, aia304Name)

    '''
    figure1 = plt.figure(frameon=False)
    ax1 = plt.subplot(projection=aia_submap)
    # Disable the axis
    # ax1.set_axis_off()

    # Plot the map.
    norm = aia_submap.plot_settings['norm']
    norm.vmin, norm.vmax = np.percentile(aia_submap.data, [1, 99.9])
    ax1.imshow(aia_submap.data,
               norm=norm,
               cmap=aia_submap.plot_settings['cmap'])
    plt.savefig("figure/{}/193/{}/aia_ar_{}_{}.png".format(iscme,idx,timeStrForFig,idx))

    figure2 = plt.figure(frameon=False)
    ax2 = plt.subplot(projection=hmi_submap)
    # Disable the axis
    # ax2.set_axis_off()

    # Plot the map. Since are not interested in the exact map coordinates, we can
    # simply use :meth:`~matplotlib.Axes.imshow`.
    # norm = hmi_submap.plot_settings['norm']
    # norm.vmin, norm.vmax = np.percentile(hmi_submap.data, [1, 99.9])
    ax2.imshow(hmi_submap.data,
               #           norm=norm,
               cmap=hmi_submap.plot_settings['cmap'])
    plt.savefig("figure/{}/mag/{}/hmi_ar_{}_{}.png".format(iscme,idx,timeStrForFig,idx))
    '''

'''
tstart = '2017/05/01 07:23:56'
tend = '2017/05/10 08:40:29'
CEresults = getCmes(tstart, tend)
ARresults,cache = getArs(CEresults)
dists,cache = getDists(ARresults,cache)
CE_x, CE_y, CE_tstart, CE_tend, CE_t, \
             AR_xs, AR_ys, AR_tstarts, \
                AR_tends, AR_ts, AR_coords, cmaps, \
                 time1,time2 = cache

ar_idx = 2
AR_area = ARresults['hek']['area_raw'][ar_idx]
AR_LL_x = ARresults['hek']['boundbox_c1ll'][ar_idx]
AR_LL_y = ARresults['hek']['boundbox_c2ll'][ar_idx]
AR_UR_x = ARresults['hek']['boundbox_c1ur'][ar_idx]
AR_UR_y = ARresults['hek']['boundbox_c2ur'][ar_idx]
AR_width = AR_UR_x-AR_LL_x
AR_height = AR_UR_y-AR_LL_y
getFrame(AR_ts[ar_idx],
         AR_xs[ar_idx],
         AR_ys[ar_idx],
         max(AR_width,300),
         max(AR_height,300),
         iscme=0,
         idx=ar_idx)
'''

def getFrames(AR_coord,tstart,tend,width=100,height=100,iscme=0,freq='min',ar_idx=0):
    '''
    
    :param x: 
    :param y: 
    :param t: 
    :param tstart: 
    :param tend: 
    :param width: 
    :param height: 
    :param iscme: 
    :return: 
    '''
    ts = list(pd.date_range(tstart, tend, freq=freq))
    for t in ts:
        rotated_coord = solar_rotate_coordinate(AR_coord, time=t)
        getFrame(t,
                 rotated_coord.Tx.arcsec,
                 rotated_coord.Ty.arcsec,
                 width,
                 height,
                 iscme=iscme,
                 idx=ar_idx)



tstart = '2017/07/01 07:23:56'
tend = '2017/07/10 08:40:29'
CEresults = getCmes(tstart, tend)
ARresults,cache = getArs(CEresults)
dists,cache = getDists(ARresults,cache)
CE_x, CE_y, CE_tstart, CE_tend, CE_t, \
            AR_xs, AR_ys, AR_tstarts, \
            AR_tends, AR_ts, AR_coords, cmaps, \
            time1,time2 = cache

#min_idx = dists.index(min(dists))
#ar_idx = min_idx
#print(min_idx,min(dists))

AR_areas = ARresults['hek']['area_raw']
AR_LL_xs = ARresults['hek']['boundbox_c1ll']
AR_LL_ys = ARresults['hek']['boundbox_c2ll']
AR_UR_xs = ARresults['hek']['boundbox_c1ur']
AR_UR_ys = ARresults['hek']['boundbox_c2ur']
AR_widths = AR_UR_xs-AR_LL_xs
AR_heights = AR_UR_ys-AR_LL_ys

ar_idx = 3
getFrames(AR_coords[ar_idx],
          max(AR_tstarts[ar_idx],time1),
          min(AR_tends[ar_idx],time2),
          width=AR_widths[ar_idx],
          height=AR_heights[ar_idx],
          iscme=0,
          freq='1min',
          ar_idx=ar_idx)