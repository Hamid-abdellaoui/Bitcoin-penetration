from datetime import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd
import dateparser



def scrap_page_6(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    
    box = soup.find('div', class_='cat-body')
    articles = box.find_all('li','post') 
    for article in articles:
        date = article.find('time')['datetime']
        date = datetime.strptime(date, '%Y-%m-%dT%H:%M:%S%z')
        row = [ article.find('h2', class_='cat-list-title').get_text() , date,  ' '.join(article.find('p').get_text().split())] 
        data.append(row)
        

def crawler(urls, site, scraping_fun):
    
    i = 1
    for url in urls:
        scraping_fun(url)
        print(i, url)
        i = i+1
        
    df = pd.DataFrame(data, columns =['titre', 'date', 'extrait'])
    df.to_csv(str('../raw data/')+str(site)+'.csv', index=False, sep=',') 
    
#list the websites to scrape
n = 1270
urls = ['https://aujourdhui.ma/category/economie/page/'+str(k) for k in range(1,n)]
data = []
crawler(urls, 'aujourdui', scrap_page_6)
