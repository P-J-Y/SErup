import json
#import requests
#import h5py
path='data/cmefile.json'
file = open(path,'r',encoding='utf-8')
cmeSourceCatalog = json.load(file)
def getCmeList():
    cmelist = []
    for thecme in cmeSourceCatalog:
        if not thecme['sourceLocation']:
            continue
        else:
            cmelist.append(thecme)

    filename = 'data/cmelist.json'
    with open(filename, 'w') as file_obj:
        json.dump(cmelist, file_obj)

import numpy as np
import datetime
cmelistpath = 'data/cmelist.json'
file = open(cmelistpath, 'r', encoding='utf-8')
cmelist = json.load(file)

sts = np.zeros(len(cmelist),datetime.datetime)
fmt = '%Y-%m-%dT%H:%MZ'
for i in range(len(sts)):
    sts[i] = datetime.datetime.strptime(cmelist[i]['startTime'],fmt)
stdiff = np.diff(sts)
# 总cme数据的时间
sts2 = np.zeros(len(cmeSourceCatalog),datetime.datetime)
for i in range(len(sts2)):
    sts2[i] = datetime.datetime.strptime(cmeSourceCatalog[i]['startTime'],fmt)
st2diff = np.diff(sts2)
#note
#1. cmelist每个元素是字典，字典的sourceLocation就是源区位置，如果没有，可以直接用判断 if not celist[i]['sourceLocation']: continum 这样
#2. 如何得到cmelist.json: 直接从网站下载json文件就行 例如输入网址 https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CME?startDate=2015-01-23&endDate=2015-12-01（我搞不来，并不能直接下一个json）
#3. 建议直接在 https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CME?startDate=2015-01-23&endDate=2015-12-01 网站下数据（找到你感兴趣的时间）
'''
startDate = "2015-01-01"
endDate = "2016-01-01"
cmeFileUrl = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CME?startDate={}&endDate={}".format(startDate,endDate)
cmeFile = requests.get(cmeFileUrl,allow_redirects=True)
cmeFile = cmeFile.text
cmeFile = eval(cmeFile)
'''
print('down')