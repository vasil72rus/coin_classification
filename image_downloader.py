import pandas as pd
import re
import numpy as np
from requests import request
from PIL import Image

data = pd.read_csv('catalognew (3).csv', sep=',', encoding='utf-8')
data = data[pd.notnull(data['image_marker'])].sort_values('catalog')
for line in data.values:
    try:
        img = request('GET', f'https://www.numizmatik.ru/images/shopcoins/2000/{line[22]}_a.jpg').content
        with open(f'images/{line[0]}_a.jpg', 'wb') as f:
            f.write(img)
        img = request('GET', f'https://www.numizmatik.ru/images/shopcoins/2000/{line[22]}_r.jpg').content
        with open(f'images/{line[0]}_r.jpg', 'wb') as f:
            f.write(img)
    except Exception as ex:
        print(line[0], line[22])