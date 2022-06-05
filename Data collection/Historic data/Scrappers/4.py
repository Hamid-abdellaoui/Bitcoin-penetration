import requests
from bs4 import BeautifulSoup
import pandas as pd



def scrap_page_3(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    
    box = soup.find('div', class_='region-content')
    articles = box.find_all(class_='block-1') 
    for article in articles:
        date = article.find('div', class_='date-actualites')
        if date:
            row = [article.find('h3').get_text(), date.get_text(), article.find('div', class_='content-lead-ville').get_text() ] 
            data.append(row)
        else : continue

def crawler(urls, site, scraping_fun):
    
    i = 1
    for url in urls:
        scraping_fun(url)
        print(i, url)
        i = i+1
        
    df = pd.DataFrame(data, columns =['titre', 'date', 'extrait'])
    df.to_csv(str('../raw data/')+str(site)+'.csv', index=False, sep=',')       

#list the websites to scrape
n = 931
urls = ['https://www.mapnews.ma/fr/actualites/economie?page='+str(k) for k in range(0,n)]
data = []
crawler(urls, 'mapnews', scrap_page_3)