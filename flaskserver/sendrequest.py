import requests

dictToSend= {'RESIDENTIAL_UNITS':2, 'COMMERCIAL_UNITS':1, 'GROSS_SQUARE_FEET':4824, 'AGE':107,'11207':0, '11239':0,'11236':0,'11208':0,'11234':0,'11203':0,'11212':0,'11224':0,'11210':0,'11229':0,'11233':0,'11228':0,'11204':0,'11214':0,'11221':0,'11209':0,'11235':0,'11213':0,'11223':0,'11220':0,
             '11219':0,'11218':0,'11230':0,'11226':0,'11232':0,'11237':0,'11216':0,'11225':0,'11231':0,'11215':0,'11222':0,'11217':0,'11238':0,'11206':0,'11205':0,'11211':0,'11249':0,'11201':1,'TAX:1':1,'TAX:2':0,'TAX:4':0}
res = requests.post('http://127.0.0.1:80/predict', json=dictToSend)
print ('response from server:',res.text)
dictFromServer = res.json()
