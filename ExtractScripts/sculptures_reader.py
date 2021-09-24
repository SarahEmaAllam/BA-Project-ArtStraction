import pandas as pd
import requests
from bs4 import BeautifulSoup

if __name__ == '__main__':
    df = pd.read_csv('./Databases/sculpture.csv')
    urls = df['URL']

    base_url = 'https://www.wga.hu'
    images = []
    for index, url in enumerate(urls):
        print(url)
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        images_in_page = soup.findAll('a')
        for image in images_in_page:
            if 'art' in image.get('href'):
                full_url = base_url + image.get('href')
                r = requests.get(full_url).content  # Get request on full_url
                with open('./images' + str(index) + '.jpg', 'wb') as handler:
                    handler.write(r)
                    handler.close()