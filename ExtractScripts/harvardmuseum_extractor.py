import pandas as pd
import requests
from bs4 import BeautifulSoup


urls = 'https://api.harvardartmuseums.org/object?apikey=729d6989-7ea2-46fd-a0f1-9626d1e3278b&keyword=three+graces'
page = requests.get(urls)
page = page.json()
print(page['info']['totalrecords'])
items = page['records']

# Find all of the objects with the word "cat" in the title and return only a few fields per record
# r = http.request('GET', 'https://api.harvardartmuseums.org/object?apikey=729d6989-7ea2-46fd-a0f1-9626d1e3278b&keyword=Tanagra')
    # fields = {
    #     'apikey': '729d6989-7ea2-46fd-a0f1-9626d1e3278b',
    #     'title': 'cat',
    #     'fields': 'objectnumber,title,dated'
    # })


print(items)
#
# print(page)
# r = 'https://collectionapi.metmuseum.org/public/collection/v1/objects/'
df = pd.DataFrame(columns=['AUTHOR',	'TITLE',	'DATE',	'TECHNIQUE',	'LOCATION',	'ID',	'TYPE', 'image', 'SCHOOL'])
cnt = 0
for object in items:
    print(object)

    if('primaryimageurl' in object and object['primaryimageurl'] != 'None') :
        ID = object['objectid']
        print(ID)
        TITLE = object['title']
        print(TITLE)
        TYPE = object['classification']
        print(TYPE)
        if 'culture' in object:
            LOCATION = object['culture']
        else: LOCATION = ''
        if 'dated' in object:
            DATE = object['dated']
        else: DATE = ''
        if object['peoplecount'] != 0:
            print(object['people'])
            authors = object['people']
            AUTHOR = ''
            for author in authors:
                AUTHOR = AUTHOR + ', ' + author['name']
                if (author['role'] == "Artist after"):
                    print(author['role'][0])
                    SCHOOL = "After " + author['name']
                else:
                    SCHOOL = ''
        else:
            AUTHOR = 'anonymous'
            SCHOOL = ' '
        if 'medium' in object:
            TECHNIQUE = object['medium']
            # if 'dimensions' in object:
            #     TECHNIQUE = TECHNIQUE + ' ' + object['dimensions']
        else: TECHNIQUE = ''
        if object['primaryimageurl'] != None:
            image1 = object['primaryimageurl']
            print(image1)
            print(image1)
            image = requests.get(image1, stream=True)
            print()
            if image.content != None:
                with open('./Databases/Graces/' + str(ID) + '.jpg', 'wb') as f:
                    f.write(image.content)
                df2 = pd.DataFrame([[AUTHOR, TITLE,	DATE, TECHNIQUE, LOCATION, ID, TYPE, image1, SCHOOL]], columns=['AUTHOR',	'TITLE',	'DATE',	'TECHNIQUE',	'LOCATION',	'ID',	'TYPE', 'image', 'SCHOOL'])
                df = pd.concat([df2, df])
                cnt = 1+ cnt

print('number of entries: ' + str(cnt))
with open('./Databases/Graces.csv', 'a') as f:
    df.to_csv(f, header=f.tell() == 0)

print(df.head(50))
print(cnt)
# df.to_excel('Tanagra.xlsx', index = False)
# df.to_excel("./Databases/TanagraHarvardDataset.xlsx")