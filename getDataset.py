import numpy as np
from sunpy.net import attrs as a
from sunpy.net import Fido
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

    rotated_coords = []
    for i in range(len(AR_xs)):
        rotated_coords.append(solar_rotate_coordinate(
            SkyCoord(AR_xs[i] * u.arcsec, AR_ys[i] * u.arcsec, 1 * u.AU,
                     frame="heliographic_stonyhurst",
                     obstime=AR_ts[i]),
            time=CE_t))
    return rotated_coords

tstart = '2015/05/01 07:23:56'
tend = '2015/05/02 08:40:29'
CEresults = getCmes(tstart, tend)
ARresults,cache = getArs(CEresults)
dists = getDists(ARresults,cache)


