import requests
from bs4 import BeautifulSoup
import pandas as pd
import dateparser

def scrap_links(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    
    container = soup.find(id="article-container")
    
    if container != None:
        paragraphs = container.find_all('p')
    else:
        paragraphs = soup.find('div', class_='single-pre-content').find_all('p')
    
    if len(paragraphs)!=0:
        extrait = paragraphs[0].get_text()
    else: extrait = ' '
    return extrait

def scrap_page(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    
    box = soup.find('ul', class_=["articles-list", "infinite-archive"])
    articles= box.find_all('li',class_='article')
    
    for article in articles:
        date = article.find('time', class_='article-publish')
        date = date.get_text()
        date = dateparser.parse(str(date)).date()
        anchor = article.find('h3', class_='article-heading')
        titre = ' '.join(anchor.get_text().split()) 
        link = anchor.find('a')['href']
        extrait = scrap_links(link)
        row = [ titre , date, ' '.join(extrait.split())] 
        data.append(row)

def crawler(urls, site, scraping_fun):
    
    i = 1
    for url in urls:
        scraping_fun(url)
        print(i, url)
        i = i+1
        
    df = pd.DataFrame(data, columns =['titre', 'date', 'extrait'])
    df.to_csv("../raw data/"+str(site)+'.csv', index=False, sep=',')         

#list the websites to scrape
n = 165
urls = ['https://telquel.ma/categorie/economie/page/'+str(k) for k in range(23,24)]
data = []

crawler(urls, 'telquel', scrap_page)