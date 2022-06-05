import requests
from bs4 import BeautifulSoup
import pandas as pd
import dateparser


def scrap_page_8(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    
    box = soup.find('div', class_='grid3')
    articles = box.find_all('a') 
    for article in articles:
        Titre = article.find('h2').get_text()
        date = article.find('span', class_='accueildate').get_text()
        date = dateparser.parse(str(date)).date()
        extrait = article.find('div',class_='mbs')
        unwanted = extrait.find('div', class_='accueil')
        unwanted = extrait.find('h2')
        unwanted = extrait.find('span', class_='accueildate')
        unwanted.extract()
        row = [ Titre, date,  ' '.join(extrait.get_text().split())] 
        data.append(row)
        

def crawler(urls, site, scraping_fun):
    
    i = 1
    for url in urls:
        scraping_fun(url)
        print(i, url)
        i = i+1
        
    df = pd.DataFrame(data, columns =['titre', 'date', 'link'])
    df.to_csv(str('../raw data/')+str(site)+'.csv', index=False, sep=',') 

#list the websites to scrape
n = 7100
urls = ['https://www.bladi.net/economie.html?debut_suite_rubrique='+str(k)+'#pagination_suite_rubrique' for k in range(1,n)]
data = []
crawler(urls, 'bladi', scrap_page_8)
