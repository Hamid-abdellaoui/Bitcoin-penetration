import requests
from bs4 import BeautifulSoup
import pandas as pd
import dateparser



def scrap_page_4(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    
    box = soup.find('div', class_='row_list_article')
    sub_box = box.find('div', class_='col-lg-8')
    articles = sub_box.find_all('div', class_='row') 
    for article in articles:
        date = article.find('a').find('span')
        unwanted = date.find('b', class_='par')
        unwanted.extract()
        date = date.get_text()
        date = dateparser.parse(str(date.split(' ')[1:])).date()
        row = [ ' '.join(article.find('a').get_text().split()) , date,  ' '.join(article.find('p').get_text().split())  ] 
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
n = 920
urls = ['https://laquotidienne.ma/articles/economie/'+str(k) for k in range(1,n)]
data = []

crawler(urls, 'laquotidienne', scrap_page_4)