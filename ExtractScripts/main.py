import pandas as pd
import requests
from bs4 import BeautifulSoup


if __name__ == '__main__':
    df = pd.read_excel('./Databases/WGA2.xlsx')
    if (df[df['TITLE'].str.contains("Gladiators", na=False)].shape[0]) != 0:
        url = df[df['TITLE'].str.contains("Gladiators", na=False)]
    else:
        print('EMPTY')
        exit()
    print(url)

    # print(url)
    # print(df)

    df2 = pd.DataFrame(columns=['AUTHOR', 'TITLE', 'DATE', 'TECHNIQUE', 'LOCATION', 'ID', 'TYPE', 'URL'])
    base_url = 'https://www.wga.hu'
    images = []
    cnt=0
    # print(urls)
    # print(urls[0])
    # print(urls[1])
    websites = url['URL']

    for image1 in websites:
        cnt = cnt + 1
        print(image1)
        page = requests.get(image1)
        soup = BeautifulSoup(page.content, 'html.parser')
        images_in_page = soup.findAll('a')
        for image in images_in_page:
            if 'art' in image.get('href'):
                full_url = base_url + image.get('href')
                r = requests.get(full_url).content  # Get request on full_url
                # with open('./Databases/Reclining/' + str(cnt) + '.jpg', 'wb') as handler:
                #     handler.write(r)
                #     handler.close()

        df3 = pd.concat([url, df2])
        print(url)

print(df3)
# with open('./Databases/Reclining.csv', 'a') as f:
#     df3.to_csv(f, header=f.tell() == 0)