import requests
from bs4 import BeautifulSoup
import pandas as pd
import dateparser



def scrap_page_7(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    
    box = soup.find('div', class_='block-content')
    articles = box.find_all('div','news-post') 
    for article in articles:
        date = article.find_all('li')[0].get_text()
        date = dateparser.parse(str(date)).date()
        row = [ article.find('h2').get_text() , date,  ' '.join(article.find('p').get_text().split())] 
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
n = 313
urls = ['https://www.maroc-hebdo.press.ma/categorie/economie?page='+str(k) for k in range(1,n)]
data = []
crawler(urls, 'hebdomaroc', scrap_page_7)