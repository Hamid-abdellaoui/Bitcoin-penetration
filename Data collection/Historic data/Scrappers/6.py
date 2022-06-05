import requests
from bs4 import BeautifulSoup
import pandas as pd
import dateparser



def scrap_page_5(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    
    box = soup.find('div', class_='col-gold-lg')
    articles = box.find_all('article') 
    for article in articles:
        date = article.find('i',class_='publishDate').get_text()
        date = dateparser.parse(str(date)).date()
        row = [ article.find('h2', class_='titre').get_text() , date,  article.find('p', class_='extrait').get_text()  ] 
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
n = 1118
urls = ['https://medias24.com/categorie/economie/page/'+str(k) for k in range(1,n)]
data = []
crawler(urls, 'media24', scrap_page_5)
