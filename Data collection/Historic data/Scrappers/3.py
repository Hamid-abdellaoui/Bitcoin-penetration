import requests
from bs4 import BeautifulSoup
import pandas as pd


def scrap_page_2(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    
    box = soup.find('div', class_='content-block')
    articles = box.find_all('div', class_='views-row') 
    for article in articles:
        row = [article.find('h3').get_text(), article.find('span', class_='date').find('span').get_text(), article.find('p').get_text() ] 
        data.append(row)

def crawler(urls, site, scraping_fun):
    
    i = 1
    for url in urls:
        scraping_fun(url)
        print(i, url)
        i = i+1
        
    df = pd.DataFrame(data, columns =['titre', 'date', 'extrait'])
    df.to_csv(str('../raw data/')+str(site)+'.csv', index=False, sep=',')         

#list of websites to scrape
urls = ['https://fr.le360.ma/economie?page='+str(k) for k in range(1,3047)]
data = []
crawler(urls, 'le360', scrap_page_2)
