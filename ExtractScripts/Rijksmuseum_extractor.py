import pandas as pd
import requests
from bs4 import BeautifulSoup

if __name__ == '__main__':
    df = pd.read_csv('./Databases/202001-rma-csv-collection.csv')
    nan_value = float("NaN")
    df.replace("", nan_value, inplace=True)
    df.dropna(subset=['objectImage'], inplace=True)

    # Unfinished code for transcribing the rijks csv into the normalized excel file with all data
    # dataframe = pd.DataFrame(df, columns=['URL', '', 'TITLE', 'FORM', 'AUTHOR', 'DATE', 'IMAGE'])

    urls = df['objectImage']
    print(urls.head(5))

    images = []

    for index, url in enumerate(urls):
        print(url)
        r = requests.get(url).content
        with open('./images' + str(index) + '.jpg', 'wb') as handler:
            handler.write(r)
            handler.close()

