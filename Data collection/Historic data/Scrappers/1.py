from bs4 import BeautifulSoup
import requests
from datetime import datetime
import pandas as pd


# ######## Article19 #########


def scrap_links(webpage):
    result = requests.get(webpage)
    content = result.text
    soup = BeautifulSoup(content, 'lxml')

    box = soup.find('div', class_='td-ss-main-content')
    
    links = [link['href'] for link in box.find_all('a', href=True, class_='td-image-wrap')]
    AllLinks.extend(links)
    
def scrap_pages(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    
    box = soup.find('div', class_='td-post-header')
    content = soup.find(class_='td-post-content')
    if len(content.find_all('p'))!=0:
        paragraph = content.find_all('p')[0].get_text()
    else: paragraph = ''
    title = box.find('h1', class_='entry-title').get_text()
    date = box.find('time', class_='entry-date')['datetime']
    date = datetime.strptime(date, '%Y-%m-%dT%H:%M:%S%z')
    date = date.strftime('%d/%m/%Y')
    
    if date and title and paragraph:
        row = [title, date, ' '.join(paragraph.split()) ] 
        Data.append(row)
        
def crawler(urls, site, scraping_fun): 
    i = 1
    for url in urls:
        scraping_fun(url)
        print(i, url)
        i = i+1
        
    df = pd.DataFrame(Data, columns =['titre', 'date', 'extrait'])
    df.to_csv(str('../raw data/')+str(site)+'.csv', index=False, sep=',') 
    
AllLinks = []

n = 212
urls = ['https://article19.ma/accueil/archives/category/economie/page/'+str(k) for k in range(0,n)]

for url in urls :
    scrap_links(url)
    print(url)

Data = []
crawler(AllLinks,'Article19',scrap_pages)