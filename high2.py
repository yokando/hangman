import requests
import json

url = 'https://zipcloud.ibsnet.co.jp/api/search'
param = {'zipcode': '860-0044'}
res = requests.get(url, params=param)
print(res.content)

res_dict = json.loads(res.text)
addrdata = res_dict['results'][0]
print(addrdata['address1'] + addrdata['address2'] + addrdata['address3'])
