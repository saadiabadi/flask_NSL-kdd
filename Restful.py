import requests
import json
import numpy as np
import pandas as pd
import random
from ttictoc import tic,toc

xtest = np.array(pd.read_csv('results/prepared_kdd/xtest.csv',  header=None))

# url= 'http://130.238.29.198:5000/ml'  #drone
url ='http://130.238.28.26:5000/ml'  #agent

tic()
for i in range(100):
    index=random.randint(0,xtest.shape[0])
    xtest1=xtest[index]
    # print(index)
    xtest1=xtest1.reshape(42,1).transpose()
    x=xtest1.tolist()
    r = requests.post(url, json=json.dumps(x))
    # print(r.json())

elapsed = toc()
print("Timing for one sample",elapsed)


#################################### Blocks

xtest_block = pd.read_csv('results/prepared_kdd/xtest.csv',  header=None)
tic()
for i in range(100):
    xtest_blo = np.array(xtest_block.sample(n=10,replace=True))
    # print(xtest_blo.shape)
    x1 = xtest_blo.tolist()
    r1 = requests.post(url, json=json.dumps(x1))
    # print(i)

elapsed_block = toc()
print("Timing for block of 10 sample", elapsed_block)


tic()
for i in range(100):
    xtest_blo = np.array(xtest_block.sample(n=100,replace=True))
    # print(xtest_blo.shape)
    x1 = xtest_blo.tolist()
    r1 = requests.post(url, json=json.dumps(x1))
    # print(i)

elapsed_block = toc()
print("Timing for block of 100 sample", elapsed_block)
