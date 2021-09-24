import pandas as pd
import requests
from bs4 import BeautifulSoup


urls = 'https://api.smk.dk/api/v1/art/search?keys=Pieta'
page = requests.get(urls)
page = page.json()
print(page['found'])
items = page['items']
#
# print(page)
# r = 'https://collectionapi.metmuseum.org/public/collection/v1/objects/'
df = pd.DataFrame(columns=['AUTHOR',	'TITLE',	'DATE',	'TECHNIQUE',	'LOCATION',	'ID',	'TYPE', 'image'])
cnt = 0
for object in items:

    print(object)
    if(object['has_image']) :
        cnt = 1+ cnt
        AUTHOR = TITLE = DATE = TECHNIQUE = LOCATION = ID = TYPE = image1 = ''
        ID = object['object_number']
        print(ID)
        TITLE = object['titles'][0]['title']
        print(TITLE)
        if 'object_names' in object:
            TYPE = object['object_names'][0]['name']
        else: TYPE = ''
        print(TYPE)
        if 'production' in object:
            if 'creator_nationality' in object['production'][0]:
                LOCATION = object['production'][0]['creator_nationality']
                print('insdif')
                print(LOCATION)
        else:
            LOCATION = ''
        if 'production_date' in object:
            DATE = object['production_date'][0]['period']
        else: DATE = ''
        if 'production' in object:
            if 'creator' in object['production'][0]:
                AUTHOR = object['production'][0]['creator']
            else:
                AUTHOR = 'anonymous'
        print(object['techniques'][0])
        if 'techniques' in object:
            TECHNIQUE = object['techniques'][0]
            if 'dimensions' and 'notes' in object:
                TECHNIQUE = TECHNIQUE + ', ' + object['dimensions'][0]['notes']
        else: TECHNIQUE = ''
        print(TECHNIQUE)
        image1 = object['image_thumbnail']
        print(image1)
        image = requests.get(image1, stream=True)
        with open('./Databases/Sebastian/' + str(ID) + '.jpg', 'wb') as f:
            f.write(image.content)
        # df2 = pd.DataFrame([[AUTHOR, TITLE,	DATE, TECHNIQUE, LOCATION, ID, TYPE, image1]], columns=['AUTHOR',	'TITLE',	'DATE',	'TECHNIQUE',	'LOCATION',	'ID',	'TYPE', 'image'])
        # df = pd.concat([df2, df])

print(df.head(100))
print(cnt)
with open('./Databases/Sebastian.csv', 'a') as f:
    df.to_csv(f, header=f.tell() == 0)


# df.to_excel('Tanagra.xlsx', index = False)
# df.to_excel("./Databases/Reclining.xlsx")