
import requests
from bs4 import BeautifulSoup
import pandas as pd
import dateparser


def scrap_extrait(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    return soup.find_all('p')[0].get_text()

def scrap_page_1(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    
    box = soup.find('div', class_='wide-post-box')
    articles = box.find_all(class_='post-item') 
    for article in articles:
        date = article.find('span', class_='date').get_text()
        date = dateparser.parse(str(date.split(' ')[1:4])).date()
        titre = article.find('h2').get_text()
        link = article.find('a', href=True)['href'].replace("../../","")
        link = "https://www.lesiteinfo.com/economie/"+str(link)        
        extrait = scrap_extrait(link)
        if date and titre and extrait:
            row = [titre, date, extrait] 
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
    
#list of urls to scrape
n = 391
urls = ['https://www.lesiteinfo.com/economie/page/'+str(k) for k in range(0,n)]
data = []

  
crawler(urls, 'lesiteinfooooo',scrap_page_1)