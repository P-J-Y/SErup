# from sunpy.net import jsoc
# from sunpy.net import attrs as a
# client = jsoc.JSOCClient()
# response = client.search(a.Time('2014-01-01T00:00:00', '2014-01-01T00:10:00'),
#                          a.jsoc.Series('hmi.sharp_720s'), a.jsoc.Notify("2101110617@pku.edu.cn"))
# res = client.fetch(response)
# #requests = client.request_data(response)
#
# print(response)

import getDataset
import json
if __name__ == '__main__':
    CEidx = 690
    # testcme = {'startTime':"2021-06-23T07:24Z",'sourceLocation':"S00W00"}
    # theArInfo, cache = getDataset.getArInfoWithCmeInfo(testcme,
    #                                         time_earlier1=80,
    #                                         time_earlier2=20,
    #                                         ar_threshold=(100,6),
    #                                         fmt="%Y-%m-%dT%H:%MZ")
    # CEtstart = cache
    # getDataset.getCmeFilm(2,
    #            testcme,
    #            theArInfo,
    #            cache,
    #            time_earlier1=60,
    #            time_earlier2=10,
    #            freq='1min',
    #            film_path='figure/test/',
    #            mustmatch=False,
    #            )

    cmelistpath = 'data/cmelist.json'
    file = open(cmelistpath, 'r', encoding='utf-8')
    cmelist = json.load(file)
    ar_search_t1 = 60
    ar_search_t2 = 20
    film_t1 = 60
    film_t2 = 0
    freq = '2min'
    ar_threshold = (100, 6)
    film_path = "figure\\test\\"

    theCmeInfo = cmelist[CEidx]

    theArInfo, cache = getDataset.getArInfoWithCmeInfo(theCmeInfo,
                                            time_earlier1=ar_search_t1,
                                            time_earlier2=ar_search_t2,
                                            ar_threshold=ar_threshold,
                                            fmt="%Y-%m-%dT%H:%MZ")
    getDataset.getCmeFilm(CEidx,
               theCmeInfo,
               theArInfo,
               cache,
               time_earlier1=film_t1,
               time_earlier2=film_t2,
               freq=freq,
               film_path=film_path,
               mustmatch=False,
               )

    print("hhh")

