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


# df.to_excel('Tanagra.xlsx', index = False)
# df.to_excel("./Databases/Tanagra.xlsx")

#search?tags=tanagra&hasImages=true

# Unfinished code for transcribing the rijks csv into the normalized excel file with all data
# dataframe = pd.DataFrame(df, columns=['URL', '', 'TITLE', 'FORM', 'AUTHOR', 'DATE', 'IMAGE'])
    # with open('./images' + str(index) + '.jpg', 'wb') as handler:
    #     handler.write(r)
    #     handler.close()

# soup = BeautifulSoup(page.content, 'html.parser')
# print(soup.prettify())

# <img id="artwork__image" class="artwork__image gtm__artwork__image"
# src="https://collectionapi.metmuseum.org/api/collection/v1/iiif/248166/538347/main-image" alt="Terracotta statuette
# of a standing woman, Terracotta, Greek, probably Boeotian " itemprop="contentUrl" style="max-height: 581px">

# image = soup.find('img', {"id": "artwork__image"})

# print('HEREEE')
# print(image)
# src = image.find('src')
# print(src)
# elements = soup.findAll('span', {"class": "artwork-tombstone--value"})
# print(elements[1].getText)
# columns = ['Title', 'Period', 'Date', 'Region', 'Medium', 'Dimensions', 'Classifications']
# dictionary = dict(zip(columns, elements))
# print(dictionary)



# const axios = require('axios')
# const url = 'https://collectionapi.metmuseum.org/public/collection/v1/objects'
#
#
# async function getObject(image) {
#     const response = await axios.get(url + `/${image}`)
#     console.log(response)
#     console.log(response.data.medium)
#     console.log(response.data.primaryImage)
#     return response;
# }
#
# async function getIDs() {
#     const response = await axios.get(url)
#     console.log(response.data)
#     for (let i = 0; i < response.data.objectIDs.length; i++) {
#         let image = await axios.get(url + `/${response.data.objectIDs[i]}`)
#         console.log(image.data.primaryImage)
#         console.log(image.data.medium)
#         if (image.data.primaryimage !== undefined) {
#             // save the image
#         }
#     }
# }
#
# # getIDs()
# # // getObjects(1)
# // getObjects(4000)