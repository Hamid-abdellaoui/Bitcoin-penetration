import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import dateparser


def scrap_article(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    box = soup.find(class_='main-content')
    date = box.find('span', class_='date').get_text()
    date = dateparser.parse(str(date)).date()
    extrait = box.find_all('p')[0].get_text()
    return extrait , date

def scrap_page(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    
    box = soup.find('div', class_='block-content')
    articles = box.find_all('article') 
    for article in articles:
        h2 =  article.find('h2', class_='post-title')
        titre =h2.get_text()
        link = h2.find('a', href=True)['href']        
        extrait, date = scrap_article(link)
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
n = 41
ajaxlink = "https://www.h24info.ma/wp-admin/admin-ajax.php/?action=bunyad_block&block%5Bid%5D=grid&block%5Bprops%5D%5Bcat_labels%5D=1&block%5Bprops%5D%5Bcat_labels_pos%5D=bot-left&block%5Bprops%5D%5Breviews%5D=stars&block%5Bprops%5D%5Bpost_formats_pos%5D=center&block%5Bprops%5D%5Bload_more_style%5D=a&block%5Bprops%5D%5Bshow_post_formats%5D=1&block%5Bprops%5D%5Bmedia_ratio%5D&block%5Bprops%5D%5Bmedia_ratio_custom%5D&block%5Bprops%5D%5Bread_more%5D=none&block%5Bprops%5D%5Bcontent_center%5D=0&block%5Bprops%5D%5Bexcerpts%5D=1&block%5Bprops%5D%5Bexcerpt_length%5D=20&block%5Bprops%5D%5Bpagination%5D=true&block%5Bprops%5D%5Bpagination_type%5D=infinite&block%5Bprops%5D%5Bspace_below%5D=none&block%5Bprops%5D%5Bsticky_posts%5D=false&block%5Bprops%5D%5Bcolumns%5D=2&block%5Bprops%5D%5Bmeta_items_default%5D=true&block%5Bprops%5D%5Bpost_type%5D&block%5Bprops%5D%5Bposts%5D=100&block%5Bprops%5D%5Btaxonomy%5D=category&block%5Bprops%5D%5Bterms%5D=7&paged="
urls = [str(ajaxlink)+str(k) for k in range(0,n)]
data = []

  
crawler(urls, 'H24info',scrap_page)