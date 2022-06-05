import requests
from bs4 import BeautifulSoup
import pandas as pd
import dateparser


def scrap_page(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    
    box = soup.find(id='vw-page-content')
    articles = box.find_all(class_='vw-block-grid-item') 
    for article in articles:
        date = article.find('time').get_text()
        date = dateparser.parse(str(date)).date()
        h3 = article.find('h3', class_='vw-post-box-title')
        titre = h3.get_text()
        link = h3.find('a', href=True)['href']      
        extrait = article.find(class_='vw-post-box-excerpt').get_text()
        if date and titre and extrait:
            row = [' '.join(titre.split()), date, ' '.join(extrait.split()), link]
            data.append(row)
        else : continue

def crawler(urls, site, scraping_fun):
    
    i = 1
    for url in urls:
        scraping_fun(url)
        print(i, url)
        i = i+1
        
    df = pd.DataFrame(data, columns =['titre', 'date', 'extrait', 'link'])
    df.to_csv("../raw data/"+str(site)+'.csv', index=False, sep=',') 
    
#list of urls to scrape
n = 44
urls = ['https://www.challenge.ma/category/economie/page/'+str(k) for k in range(0,n)]
data = []

  
crawler(urls, 'challenge',scrap_page)