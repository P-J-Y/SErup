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


# 1、活动区的时刻现在用的初始时刻，实际上应该考虑从初始到end
# 2、加入判断活动区是否在cme附近的判定
# 3、如果事件在边缘，那么画一个方框就会报错，所以我限制了事件的位置不能太偏边缘

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
    cache = (CE_x, CE_y, CE_tstart, CE_tend)

    AR_tstart = CE_tstart - datetime.timedelta(minutes=time_earlier1)
    AR_tend = CE_tstart - datetime.timedelta(minutes=time_earlier2)
    ARresult = Fido.search(a.Time(AR_tstart, AR_tend), a.hek.EventType(event_type))
    return ARresult,cache
'''
tstart = '2015/05/01 07:23:56'
tend = '2015/05/02 08:40:29'
CEresults = getCmes(tstart, tend)
ARresults,cache = getArs(CEresults)
'''

def getDists(ARresults,cache,fmt = "%Y-%m-%dT%H:%M:%S"):  ##!!!还有bug
    '''
    
    :param ARresults: 
    :param cache: 
    :return: 
    '''
    CE_x, CE_y, CE_tstart, CE_tend = cache
    CE_t = CE_tstart+(CE_tend-CE_tstart)/2
    AR_tstarts = [datetime.datetime.strptime(i,fmt) for i in ARresults['hek']['event_starttime']]
    AR_tends = [datetime.datetime.strptime(i,fmt) for i in ARresults['hek']['event_endtime']]
    AR_ts = AR_tstarts+np.divide(np.subtract(AR_tends,AR_tstarts),2)
    AR_xs = ARresults['hek']['event_coord1']
    AR_ys = ARresults['hek']['event_coord2']
    #AR_coords = []
    dists = []
    for i in range(len(AR_xs)):
        theAR_coord = SkyCoord(AR_xs[i] * u.arcsec, AR_ys[i] * u.arcsec, observer='earth',
                     frame=frames.Helioprojective,
                     obstime=AR_ts[i])
        theRotated_coord = solar_rotate_coordinate(theAR_coord , time=CE_t)
        #AR_coords.append(theAR_coord)
        dists.append(
            np.sqrt((theRotated_coord.Tx.arcsec-CE_x)**2+
                     (theRotated_coord.Ty.arcsec-CE_y)**2)
        )
    cache = (CE_x, CE_y, CE_tstart, CE_tend, CE_t,
             AR_xs, AR_ys, AR_tstarts, AR_tends, AR_ts)
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
    timeStrForFig = t.strftime("%Y%m%d%H%M%S")
    hv = HelioviewerClient()
    # plot AR images
    file = hv.download_jp2(t, observatory="SDO", instrument="HMI", measurement="magnetogram")
    hmi = Map(file)
    bottom_left = SkyCoord((c1 - width/2) * u.arcsec,
                           (c2 - height/2) * u.arcsec,
                           frame=hmi.coordinate_frame)  # 给出区域左下点的坐标（第一个参数是x坐标，第二个是y）
    width = width * u.arcsec
    height = height * u.arcsec

    file = hv.download_jp2(t, observatory="SDO", instrument="AIA", measurement="193")
    aia = Map(file)
    aia_submap = aia.submap(bottom_left, width=width, height=height)
    hmi_submap = hmi.submap(bottom_left, width=width, height=height)

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
    plt.savefig("figure/{}/193/aia_ar_{}_{}.png".format(iscme,timeStrForFig,idx))

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
    plt.savefig("figure/{}/mag/hmi_ar_{}_{}.png".format(iscme,timeStrForFig,idx))


tstart = '2017/05/01 07:23:56'
tend = '2017/05/10 08:40:29'
CEresults = getCmes(tstart, tend)
ARresults,cache = getArs(CEresults)
dists,cache = getDists(ARresults,cache)
CE_x, CE_y, CE_tstart, CE_tend, CE_t, \
             AR_xs, AR_ys, AR_tstarts, AR_tends, AR_ts = cache

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