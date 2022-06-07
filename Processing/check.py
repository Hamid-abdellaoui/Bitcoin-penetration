
import pandas as pd

dict1 = ["bitcoin", "btc", "btcoins", "crypto", "cryptomonnaie", "cryptomonnaies", "cryptocurency", "cryptocurencies","ether", "etherium", "tokens","PoS", "Altcoin", "Blockchain"]
dict2 = ["monnaies numériques", "Smart Contract"]


df=pd.read_csv("Outpout/Datasets.csv", parse_dates=['date'])
df['period']=df['date'].dt.to_period('M')
corpuses = []
l_periods=list(set(df.period.values))
l_periods.sort()
for i in range(len(l_periods)-1) :
    indice1=df.period.values.tolist().index(l_periods[i])
    indice2=df.period.values.tolist().index(l_periods[i+1])
    liste = []
    for j in range(indice1,indice2,1):
        liste.append(df.extrait.values[j]) 
    corpuses.append(liste)
    
    

def check_word_in_article(article, dict):
    test = False
    for word in dict:
        if word in article:
            test = True
            break
    return test

print(len(corpuses))
# d=0
# for article in articles[0:175721]:
    
#     if check_word_in_article(article, dict1):
#         d+=1
# print(d)
