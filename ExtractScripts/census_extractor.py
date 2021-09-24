from os.path import basename

import pandas as pd
import requests
from bs4 import BeautifulSoup

# headers = {
#     'Access-Control-Allow-Origin': '*',
#     'Access-Control-Allow-Methods': 'GET',
#     'Access-Control-Allow-Headers': 'Content-Type',
#     'Access-Control-Max-Age': '3600',
#     'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
#     }

url = 'https://www.metmuseum.org/art/collection/search/247533'
page = requests.get(url)

soup = BeautifulSoup(page.content, 'html.parser')
print(soup.prettify())

# <img id="artwork__image" class="artwork__image gtm__artwork__image"
# src="https://collectionapi.metmuseum.org/api/collection/v1/iiif/248166/538347/main-image" alt="Terracotta statuette
# of a standing woman, Terracotta, Greek, probably Boeotian " itemprop="contentUrl" style="max-height: 581px">

image = soup.find('img', {"id": "artwork__image"})

print('HEREEE')
print(image)
src = image.find('src')
print(src)
elements = soup.findAll('span', {"class": "artwork-tombstone--value"})
print(elements[1].getText)
columns = ['Title', 'Period', 'Date', 'Region', 'Medium', 'Dimensions', 'Classifications']
dictionary = dict(zip(columns, elements))
print(dictionary)
# for index, element in enumerate(elements):
#     element = element.text
#     print(element)

    # r = requests.get(element).content

# period = soup.find('span', {"class": "artwork-tombstone--label"}).getText()
# with open('./images/Tanagra' + str() + '.jpg', 'wb') as handler:
#     handler.write(src)
#     handler.close()
# with open(url) as fp:
#     soup = BeautifulSoup(fp, "html.parser")
#     print(soup.prettify())
