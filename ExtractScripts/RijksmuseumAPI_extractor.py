import pandas as pd
import requests
from bs4 import BeautifulSoup


urls = 'https://www.rijksmuseum.nl/api/en/collection?key=GNFymmyP&&q=&title=Pieta'
page = requests.get(urls)
print(page.json())
items = page.json()['artObjects']

print(page)
r = 'https://www.rijksmuseum.nl/api/en/collection/'
key = '?key=GNFymmyP&&q='
df = pd.DataFrame(columns=['AUTHOR',	'TITLE',	'DATE',	'TECHNIQUE',	'LOCATION',	'ID',	'TYPE', 'image'])
for objectID in items:
    if objectID['showImage']:
        print(objectID)
        url = requests.get(r + str(objectID['objectNumber'] + key))
        url = url.json()
        url = url['artObject']
        print(url)
        ID = url['objectNumber']
        print(ID)
        image = url['webImage']['url']
        print(image)
        TITLE = url['title']
        print(TITLE)
        TYPE = url['classification']
        print(TYPE)
        LOCATION = url['productionPlaces']
        DATE = url['dating']['presentingDate']
        AUTHOR = url['principalMaker']
        TECHNIQUE = str(url['techniques']) + ', ' + str(url['subTitle'])
        if 'objectTypes' in url:
            print(url['objectTypes'])
            TYPE = url['objectTypes']
        print(image)
        image1 = requests.get(image, stream=True)
        with open('./Databases/Pieta/' + str(ID) + '.jpg', 'wb') as f:
            f.write(image1.content)
        df2 = pd.DataFrame([[AUTHOR, TITLE,	DATE, TECHNIQUE, LOCATION, ID, TYPE, image]], columns=['AUTHOR',	'TITLE',	'DATE',	'TECHNIQUE',	'LOCATION',	'ID',	'TYPE', 'image'])
        df = pd.concat([df2, df])

print(df.head(50))

with open('./Databases/Pieta.csv', 'a') as f:
    df.to_csv(f, header=f.tell() == 0)


# df.to_excel('Tanagra.xlsx', index = False)
# df.to_excel("./Databases/Tanagra.xlsx")