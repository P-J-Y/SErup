from sunpy.net import jsoc
from sunpy.net import attrs as a
client = jsoc.JSOCClient()
response = client.search(a.Time('2014-01-01T00:00:00', '2014-01-01T00:10:00'),
                         a.jsoc.Series('hmi.sharp_720s'), a.jsoc.Notify("2101110617@pku.edu.cn"))
res = client.fetch(response)
#requests = client.request_data(response)

print(response)