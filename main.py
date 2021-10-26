from sunpy.net import attrs as a
from sunpy.net import Fido

#1、活动区的时刻现在用的初始时刻，实际上应该考虑从初始到end
#2、加入判断活动区是否在cme附近的判定
#3、如果事件在边缘，那么画一个方框就会报错，所以我限制了事件的位置不能太偏边缘

#给出时间范围内的活动区位置、时间范围等信息
tstart = '2015/05/01 07:23:56'
tend = '2015/05/02 08:40:29'

event_type = 'CE' #CME 可以设置成其他的 如 AR、FL
from sunpy.net import hek2vso
h2v = hek2vso.H2VClient()
CEresult = Fido.search(a.Time(tstart, tend), a.hek.EventType(event_type))
vso_records = h2v.translate_and_query(CEresult[0][0])
len(vso_records[0])

tend = CEresult['hek']['event_starttime'][0]
tstart = "2015/05/01 07:00:05"
#q = h2v.full_query((a.Time('2011/08/09 07:23:56', '2011/08/09 12:40:29'), a.hek.EventType('FL')))
#print(CEresult['hek']['event_starttime'])
#print(CEresult['hek']['event_endtime'])
#print(CEresult['hek']['event_coord1'])
#print(CEresult['hek']['event_coord2']) # 角秒 event_coord1/2/3 第一二三个坐标，还有其他参数见https://www.lmsal.com/hek/VOEvent_Spec.html

event_type = 'AR'
ARresult = Fido.search(a.Time(tstart, tend), a.hek.EventType(event_type))
vso_records = h2v.translate_and_query(ARresult[0][0])
len(vso_records[0])
#q = h2v.full_query((a.Time('2011/08/09 07:23:56', '2011/08/09 12:40:29'), a.hek.EventType('FL')))
#print(ARresult['hek']['event_starttime'])
#print(ARresult['hek']['event_endtime'])
#print(ARresult['hek']['event_coord1'])
#print(ARresult['hek']['event_coord2']) # 角秒 event_coord1/2/3 第一二三个坐标，还有其他参数见https://www.lmsal.com/hek/VOEvent_Spec.html

from sunpy.net.helioviewer import HelioviewerClient
import matplotlib.pyplot as plt
from sunpy.map import Map
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
from astropy.units import Quantity

for i in range(20):
    thetime = ARresult['hek']['event_starttime'][i]
    thex = ARresult['hek']['event_coord1'][i]
    they = ARresult['hek']['event_coord2'][i]
    if np.abs(thex)>800 or np.abs(they)>800:
        continue
    hv = HelioviewerClient()
    file = hv.download_jp2(thetime, observatory="SDO", instrument="HMI", measurement="continuum")
    hmi = Map(file)
    bottom_left = SkyCoord((thex - 100) * u.arcsec,
                           (they - 100) * u.arcsec,
                           frame=hmi.coordinate_frame)  # 给出区域左下点的坐标（第一个参数是x坐标，第二个是y）
    width = 200 * u.arcsec
    height = 200 * u.arcsec

    file = hv.download_jp2(thetime, observatory="SDO", instrument="AIA", measurement="171")
    aia = Map(file)
    aia_submap = aia.submap(bottom_left, width=width, height=height)
    hmi_submap = hmi.submap(bottom_left, width=width, height=height)
    # aia.submap(bottom_left, width=width, height=height).peek()
    # plt.savefig('aia.png')
    # hmi.submap(bottom_left, width=width, height=height).peek()
    # plt.savefig('hmi.png')

    figure1 = plt.figure(frameon=False)
    ax1 = plt.axes([0, 0, 1, 1])
    # Disable the axis
    ax1.set_axis_off()

    # Plot the map. Since are not interested in the exact map coordinates, we can
    # simply use :meth:`~matplotlib.Axes.imshow`.
    norm = aia_submap.plot_settings['norm']
    norm.vmin, norm.vmax = np.percentile(aia_submap.data, [1, 99.9])
    ax1.imshow(aia_submap.data,
               norm=norm,
               cmap=aia_submap.plot_settings['cmap'])
    plt.savefig('figure/aia%d.png'%i)

    figure2 = plt.figure(frameon=False)
    ax2 = plt.axes([0, 0, 1, 1])
    # Disable the axis
    ax2.set_axis_off()

    # Plot the map. Since are not interested in the exact map coordinates, we can
    # simply use :meth:`~matplotlib.Axes.imshow`.
    # norm = hmi_submap.plot_settings['norm']
    # norm.vmin, norm.vmax = np.percentile(hmi_submap.data, [1, 99.9])
    ax2.imshow(hmi_submap.data,
               #           norm=norm,
               cmap=hmi_submap.plot_settings['cmap'])
    plt.savefig('figure/hmi%d.png'%i)

'''
#给出helioviewer相应位置的太阳图像
thetime = ARresult['hek']['event_starttime'][0]
hv = HelioviewerClient()
file = hv.download_jp2(thetime, observatory="SDO", instrument="HMI",measurement="continuum")
hmi = Map(file)
bottom_left = SkyCoord((ARresult['hek']['event_coord1'][0]-100) * u.arcsec, (ARresult['hek']['event_coord2'][0]-100) * u.arcsec, frame=hmi.coordinate_frame) #给出区域左下点的坐标（第一个参数是x坐标，第二个是y）
width = 200 * u.arcsec
height = 200 * u.arcsec

file = hv.download_jp2(thetime, observatory="SDO", instrument="AIA",measurement="171")
aia = Map(file)
aia_submap = aia.submap(bottom_left, width=width, height=height)
hmi_submap = hmi.submap(bottom_left, width=width, height=height)
#aia.submap(bottom_left, width=width, height=height).peek()
#plt.savefig('aia.png')
#hmi.submap(bottom_left, width=width, height=height).peek()
#plt.savefig('hmi.png')

figure1 = plt.figure(frameon=False)
ax1 = plt.axes([0, 0, 1, 1])
# Disable the axis
ax1.set_axis_off()

# Plot the map. Since are not interested in the exact map coordinates, we can
# simply use :meth:`~matplotlib.Axes.imshow`.
norm = aia_submap.plot_settings['norm']
norm.vmin, norm.vmax = np.percentile(aia_submap.data, [1, 99.9])
ax1.imshow(aia_submap.data,
           norm=norm,
           cmap=aia_submap.plot_settings['cmap'])
plt.savefig('aia.png')

figure2 = plt.figure(frameon=False)
ax2 = plt.axes([0, 0, 1, 1])
# Disable the axis
ax2.set_axis_off()

# Plot the map. Since are not interested in the exact map coordinates, we can
# simply use :meth:`~matplotlib.Axes.imshow`.
#norm = hmi_submap.plot_settings['norm']
#norm.vmin, norm.vmax = np.percentile(hmi_submap.data, [1, 99.9])
ax2.imshow(hmi_submap.data,
#           norm=norm,
           cmap=hmi_submap.plot_settings['cmap'])
plt.savefig('hmi.png')
'''