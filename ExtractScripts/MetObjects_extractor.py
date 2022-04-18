import pandas as pd
import requests
from bs4 import BeautifulSoup


urls = 'https://collectionapi.metmuseum.org/public/collection/v1/search?q=Antinous'
page = requests.get(urls)
print(page.json())
page = page.json()['objectIDs']

print(page)
r = 'https://collectionapi.metmuseum.org/public/collection/v1/objects/'
df = pd.DataFrame(columns=['AUTHOR',	'TITLE',	'DATE',	'TECHNIQUE',	'LOCATION',	'ID',	'TYPE', 'image'])
for objectID in page:
    url = requests.get(r + str(objectID))
    url = url.json()
    print(url)
    ID = url['objectID']
    print(ID)
    image = url['primaryImage']
    print(image)
    TITLE = url['title']
    print(TITLE)
    TYPE = url['classification']
    print(TYPE)
    LOCATION = url['culture']
    DATE = url['objectDate']
    AUTHOR = url['artistDisplayName']
    TECHNIQUE = url['medium'] + url['dimensions']
    TYPE = url['classification']
    image1 = url['primaryImage']
    print(image)
    image = requests.get(image, stream=True)
    with open('./Databases/Pieta/' + str(objectID) + '.jpg', 'wb') as f:
        f.write(image.content)
    df2 = pd.DataFrame([[AUTHOR, TITLE,	DATE, TECHNIQUE, LOCATION, ID, TYPE, image1]], columns=['AUTHOR',	'TITLE',	'DATE',	'TECHNIQUE',	'LOCATION',	'ID',	'TYPE', 'image'])
    df = pd.concat([df2, df])

print(df.head(100))
with open('./Databases/Pieta.csv', 'a') as f:
    df.to_csv(f, header=f.tell() == 0)
