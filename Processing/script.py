#importation des bibiotheque necessaire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
np.set_printoptions(precision=2, linewidth=80)
# from nltk import FreqDist
# Gensim
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel

import spacy
# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
#from bs4 import BeautifulSoup
import unicodedata

from spacy.lang.fr.stop_words import STOP_WORDS

from sklearn.feature_extraction.text import TfidfVectorizer

spacy.load('fr_core_news_md')
import fr_core_news_md #import spacy french stemmer
from sklearn.decomposition import NMF,LatentDirichletAllocation

import pyLDAvis #Nous utilisons pyLDAvis pour créer des visualisations interactives de modèles de sujet.
import pyLDAvis.sklearn

#dealing with date
import dateparser

import datetime


data_set1=pd.read_csv("../Data collection/Historic data/raw data/Article19.csv")
data_set2=pd.read_csv("../Data collection/Historic data/raw data/aujourdhui.csv")
data_set3=pd.read_csv("../Data collection/Historic data/raw data/bladi.csv")
data_set4=pd.read_csv("../Data collection/Historic data/raw data/challenge.csv")
data_set4.drop("link",axis=1,inplace=True)
data_set5=pd.read_csv("../Data collection/Historic data/raw data/H24info.csv")
data_set5.drop("link",axis=1,inplace=True)
data_set6=pd.read_csv("../Data collection/Historic data/raw data/lesiteinfo.csv")
data_set7=pd.read_csv("../Data collection/Historic data/raw data/mapnews.csv")
data_set8=pd.read_csv("../Data collection/Historic data/raw data/marochebdo.csv")
data_set9=pd.read_csv("../Data collection/Historic data/raw data/media24.csv")
data_set10=pd.read_csv("../Data collection/Historic data/raw data/telquel.csv")
data_set11=pd.read_csv("../Data collection/Historic data/raw data/le360.csv")
data_set12=pd.read_csv("../Data collection/Historic data/raw data/laquotidien.csv")
datasets = [data_set1,data_set2,data_set3,data_set4,data_set5,data_set6,
    data_set7,data_set8,data_set9,data_set10,data_set11,data_set12]


for dataset in datasets :
    print(dataset.columns , np.shape(dataset))


whole_data_set= pd.concat(datasets)


whole_data_set.head()


np.shape(whole_data_set)


lexique=["bitcoin"]


#output French accents correctly
def convert_accents(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore')


#convertisse les documents en minuscule
def lower_text(corpus):
    LowerCorpus=[]
    for doc in corpus:
        lowerDoc=str(doc).lower() #convertissent le texte en minuscules
        lowerDoc=convert_accents(lowerDoc).decode("utf-8") #supprimes les accents
        LowerCorpus.append(lowerDoc)
    return LowerCorpus


def remove_characters(corpus,keep_apostrophes=True):
    filtered_corpus=[]
    for doc in corpus:
        doc = doc.strip()
        if keep_apostrophes:
            doc =re.sub('(https|http)\S*\s?', '',doc) #supprimes les urls
            PATTERN = r'[?|$|&|*|%|@|(|)|~|\d]'
            filtered_doc = re.sub(PATTERN, r'', doc)
            filtered_corpus.append(filtered_doc)
        else:
            PATTERN = r'[^a-zA-Z ]'
            #supprimes les urls
            doc =re.sub('(https|http)\S*\s?', '',doc) #supprimes les urls
            filtered_doc = re.sub(PATTERN, r'', doc)
        
            filtered_corpus.append(filtered_doc)
    return filtered_corpus


#Tokenization
def tokenize_text(corpus):
    tokensCorpus=[]
    for doc in corpus:
        doc_tokens = word_tokenize(doc)
        tokensCorpus.append(doc_tokens)
    return tokensCorpus


#recuperer les mots qui apparaissent dans plusieurs documents
def get_mostCommonWords(corpus,max_freq=100):
    vocabulaire=dict() #dictionnaire qui va contenir le nombre d'occurence des mots dans les documents
    for doc in corpus:
        for word in set(doc.split()): #recupere les mots unique de chaque documents
            if word in vocabulaire:
                vocabulaire[word]+=1
            else:
                vocabulaire[word]=1
    
    #recupere les dont le nombre d'occurences dans les documents > max_freq
    mostCommonsWord=[word for word,value in vocabulaire.items() if value>max_freq ]
        
    return mostCommonsWord


# removing stopwords
def remove_stopwords(corpus,mostCommonsWord):
    filtered_corpus=[]
    for tokens in corpus:
        others_sw=["maroc","morocco","marocain","marocaine","marocains","marocaines","maghreb","météorologique","journée",
                   "méteo","retweet","newspic","twitter","com","pic","newspic","illustration"]
        
        #french_sw = stopwords.words('french') 
        french_sw=list(STOP_WORDS) #get french stopwords
        french_sw.extend(others_sw)
        french_sw.extend(mostCommonsWord)
        
        filtered_tokens = [token for token in tokens.split() if token not in french_sw and len(token)>2]
        filtred_text=' '.join(filtered_tokens) #reforme le text du documents separé par espace
        filtered_corpus.append(filtred_text)
    return filtered_corpus


#lemmatisation
def lemm_tokens(corpus):
    
    nlp = fr_core_news_md.load() #initialisation du model "fr_core_news_md" de spacy
    allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']
    corpus_lemms=[]
    
    idx_doc=[] #liste qui va stocker les indices documents qui seront dans le corpus
    for idx,document in enumerate(corpus):
        doc = nlp(document)
        lemms=[token.lemma_ for token in doc if token.pos_ in allowed_postags] #recupere les lemms des tokens
        
        if len(lemms)>5: #supprime les document qui ne contient pas plus de 2 mots
            text=' '.join(lemms) #reforme le text du documents separé par espace
            corpus_lemms.append(text)
            idx_doc.append(idx) #ajoute l'indice du documents
            
    return corpus_lemms,idx_doc


#fonction qui supprimes les documents vides ou tres courte
def remove_shortDocument(corpus,min_length=3):
    filtred_corpus=[]
    idx_doc=[]
    for idx,doc in enumerate(corpus):
        
        if len(doc.split())>min_length:
            filtred_corpus.append(doc)
            idx_doc.append(idx)
        
    
    return filtred_corpus,idx_doc


def make_bigrams(texts):
    bigram = gensim.models.Phrases(texts, min_count=2, threshold=10) # higher threshold fewer phrases
    # Un moyen plus rapide d'obtenir une phrase matraquée comme un trigramme / bigramme
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    corpus_bigram=[" ".join(bigram_mod[doc]) for doc in texts]
    return corpus_bigram


def preprocessing(corpus):
    
    corpus=lower_text(corpus)
    corpus=remove_characters(corpus)
    corpus=tokenize_text(corpus)
    #corpus=remove_mostCommonWords(corpus,max_freq=20)
    corpus=remove_stopwords(corpus)
    corpus,idx_docs=lemm_tokens(corpus)
    
    
    return corpus,idx_docs


corpus = whole_data_set.extrait.values.tolist()
print("Taille du corpus = "+str(len(corpus))+" Documents")


corpus=lower_text(corpus)


most_commonWords=get_mostCommonWords(corpus,500) #recupere les mots qui apparaisent dans plusieurs document
print("Nombre de mots tres frequents = "+str(len(most_commonWords))+" Mots")


corpus=remove_stopwords(corpus,most_commonWords)#supprimes les mots les plus frequents,les stop words et qlq mots inutiles
corpus,idx_doc=remove_shortDocument(corpus,min_length=3) #supprimes les documents vides
print("Nouvelle Taille du corpus = "+str(len(corpus))+" Documents")


# # TF-IDF


import nltk
nltk.download('punkt')


#build TFIDF features on train reviews with a specifique vocabulary
corpus_lemmatized=tokenize_text(corpus) 
id2word = corpora.Dictionary(corpus_lemmatized)
vocabulaire=id2word.token2id #get vocabulary dict where keys are terms and values are indices in the feature matrix

tfidf = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0,sublinear_tf=True,lowercase=True,ngram_range=(1,2),vocabulary=vocabulaire)
tfidf_train_features = tfidf.fit_transform(corpus)

 
# # Construction du modèle de sujet:
# Le modèle est construit avec 10 sujets initiaux où chaque sujet est une combinaison de mots-clés et chaque mot-clé contribue à un certain poids au sujet.


total_topics = 10 #nombre de topics
pos_nmf=NMF(n_components=total_topics,random_state=42,l1_ratio=0.2,max_iter=200)
pos_nmf.fit(tfidf_train_features) 

 
# # Affichage des sujets


# extractions des features et des poids
pos_feature_names = tfidf.get_feature_names()
pos_weights = pos_nmf.components_


# extracts topics with their terms and weights
# format is Topic N: [(term1, weight1), ..., (termn, weightn)]        
def get_topics_terms_weights(weights, feature_names):
    feature_names = np.array(feature_names)
    
    #trie les indices des mots de chaque topics selon la poids du mots dans le topics
    sorted_indices = np.array([list(row[::-1]) for row in np.argsort(np.abs(weights))])
    
    #trie les poids des mots de chaques topics,en recuperant les poids des indices deja triée
    sorted_weights = np.array([list(wt[index]) for wt, index in zip(weights,sorted_indices)])
    
    #recupres les mots selon leurs indices deja triée
    sorted_terms = np.array([list(feature_names[row]) for row in sorted_indices])
    
    #concatene chaque mots et sa poids sous formes de tuple (mot,poids)
    topics = [np.vstack((terms.T, term_weights.T)).T for terms, term_weights in zip(sorted_terms, sorted_weights)]     
    
    return topics


# prints components of all the topics 
# obtained from topic modeling
def print_topics_udf(topics, total_topics=1,weight_threshold=0.0001,display_weights=False,num_terms=None):
    
    for index in range(total_topics):
        topic = topics[index]
        topic = [(term, float(wt)) for term, wt in topic] #recupere les mots et les poids du topic
        
        #seuillage des mots selon le seuil de poids definie
        topic = [(word, round(wt,2)) for word, wt in topic if abs(wt) >= weight_threshold]
        
        #affiches les "num_terms" de chaque topics
        if display_weights:
            print('Topic #'+str(index+1)+' with weights')
            print(topic[:num_terms]) if num_terms else topic
        else:
            print('Topic #'+str(index+1)+' without weights')
            tw = [term for term, wt in topic]
            print(tw[:num_terms]) if num_terms else tw
        print()


# extract and display topics and their components
pos_topics = get_topics_terms_weights(pos_weights, pos_feature_names)
print_topics_udf(topics=pos_topics,total_topics=total_topics,num_terms=15,display_weights=True)

 
# # Calcul de la cohérence du modèle :
# La cohérence des sujets constituent une mesure pratique pour juger de la qualité d'un modèle de sujet,ici on utilise La Coherence UMass qu'on a implementé nous meme.


def getTopicTerms(pos_topics):
    """
    Fonction qui retourne l'ensemble des mots qui compose chaque topics
    ----Input----
    pos_topics: ensemble des topics qui contients les mots et leurs poids
    ---output---
    topic_terms : ensemble des mots des topics
    
    """
    topic_terms=[]
    for topic in pos_topics:
        #topic=topic[:max_term] #recupere les "max_term" premiere mots et leurs poids
        terms=[]
        for doc in topic:
            terms.append(doc[0]) #recupere justes les mots sans les poids
        
        topic_terms.append(terms) #ajoute l'ensemble des mots
    
    return topic_terms


topic_terms=getTopicTerms(pos_topics) #recupere les mot des de chaque topics

topic_terms=[topics[:20] for topics in topic_terms] #recupere les termes avec les plus grandes poids
# Term Document Frequency
common_corpus = [id2word.doc2bow(text) for text in corpus_lemmatized]

cm = CoherenceModel(topics=topic_terms,corpus=common_corpus, dictionary=id2word, coherence='u_mass')
coherence = cm.get_coherence()
print("Coherence = ", coherence)

 
# # Trouver le nombre de sujets optimal :
# Pour trouver le nombre de sujets optimal est de construire de nombreux modèles avec différentes valeurs de nombre de sujets k et de choisir celui qui donne la valeur de cohérence UMass la plus élevée


def compute_coherence_values(tfidf_train_features,feature_names,corpus,data_lemmatized,id2word,max_term=20,limit=50, start=5, step=5):
    """
    Calcul la coherence UMass pour different nombre de topic
    
    Parameters:
    ----------
    tfidf_train_features : features tf-idf qu'on va utiliser pour entrainer chaque model
    feature_names : ensemble des mots contenue dans la matrice tf-idf
    corpus: corpus de base qui contients les documents sous forme de texte
    max_term: nombre maximal de mots qu'on va prendre pour calculé la coherence de chaque topic
    data_lemmatized: corpus sous forme de tokens
    id2word:vocabulaire du corpus au format de gensim
    max_term:le nombre de termes qu'on va prendre dans chaque topic pour calculer la Coherence
    limit : Nombre maximal de topics qu'on va tester

    Returns:
    -------
    best_model : le model qui contient le plus grande coherence
    coherence_values : Valeurs des Cohérences correspondant au modèle avec le nombre respectif de sujets
    """
    
    
    model_list = [] #listes qui va contenir les modeles tester
    coherence_values = [] #liste qui contenir les coherences de chaque models
    # Term Document Frequency
    
    common_corpus = [id2word.doc2bow(text) for text in data_lemmatized] #recupere la matrice bog of word du corpus sous le format de gensim
   
    #print(coherence)
    for num_topics in range(start, limit, step):
        
        model=NMF(n_components=num_topics,random_state=42) #model MNF
        model.fit(tfidf_train_features)
        weights = model.components_ #recupere les poids
        
        model_list.append(model) #ajoute le model la liste des models utilisé
        
        
        topics=get_topics_terms_weights(weights,feature_names)
        
        topic_terms=getTopicTerms(topics)#recupere les mot des de chaque topics
        
        topic_terms=[topics[:max_term] for topics in topic_terms] #recupere les  "max_term" termes avec les plus grandes poids
        
        #calcule du Coherence UMass
        cm = CoherenceModel(topics=topic_terms,corpus=common_corpus, dictionary=id2word, coherence='u_mass')
        coherence = cm.get_coherence()
        coherence_values.append(coherence)
    
    idx_max=np.array(coherence_values).argmax() #recupere l'indice du model qui possede le plus grands coherence
    best_model=model_list[idx_max] #recupere le meilleur models
    

    return best_model,coherence_values


best_model,coherence_values=compute_coherence_values(tfidf_train_features,pos_feature_names,
                                                     corpus,corpus_lemmatized,id2word,max_term=20,limit=50)


# Show graph
import seaborn as sns
sns.set(style="darkgrid")
plt.figure(figsize=(10,6))
limit=50; start=5; step=5;
x = range(start, limit, step)
plt.plot(x,np.array(coherence_values),marker="o")
plt.xlabel("Nombre Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


total_topics=best_model.n_components #recupere le nombre de topics du meilleurs modeles
weights = best_model.components_ #recuperes les poids du meilleurs modeles
# extract features and component weights
feature_names = tfidf.get_feature_names()

print("Le nombre de topic optimal est : ",total_topics)
print("*"*100)
topics = get_topics_terms_weights(weights,feature_names)
print_topics_udf(topics,total_topics,num_terms=15,display_weights=False)


# Visualize the topics
pyLDAvis.sklearn.prepare(best_model,tfidf_train_features, tfidf, R=15)

 
# # Trouver le sujet dominant dans chaque document :
# Pour trouver cela,il suffit de trouver le numéro de topic qui a le pourcentage de contribution le plus élevé dans ce document


def topic_dominant(model,tdidf_features,corpus,topics):
    
    #document topic distribution :la probabilité des topics pour chaque document
    doc_topic_dist = model.transform(tdidf_features) 
    
    topic_num=[] #liste qui contenir le numero du topic dominant dans chaque documents
    probs_topics=[] #liste qui va contenir les probabilités du topic dominant dans chaque documents
    topic_keywords=[] #liste qui contenir les 5 termes les plus representative du sujet
    text_doc=[] #liste qui va contenir le texte de chaque documents
    
    topic_terms=getTopicTerms(topics) #recupere les mot de chaque topics
    
    num_doc=[]
    
    for i,doc in enumerate(doc_topic_dist):
        text_doc.append(corpus[i]) #recupere le texte du documents
        num_doc.append(i+1) #recupere le numero du documents
        
        idx_max=doc.argmax() #recupere l'indice du topic qui a de la probabilité maximal
        topic_num.append(idx_max) 
        probs_topics.append(round(doc.max(),4)) #recupere la probabilité maximal arrondis
        
        kw=",".join(topic_terms[idx_max][:5]) #recupere les mots clé du topic
        topic_keywords.append(kw)
        
    
    sent_topics_df = pd.DataFrame([num_doc,topic_num,probs_topics,topic_keywords,text_doc]).T
    sent_topics_df.columns=["Num Document","Topic Dominant","Contrib Topic","Key Word","Text"]
    
    return sent_topics_df


sent_topics_df=topic_dominant(best_model,tfidf_train_features,corpus,topics)
sent_topics_df.sample(20)

 
# # Trouvez le document le plus représentatif pour chaque sujet :
# Parfois, seuls les mots-clés du sujet peuvent ne pas être suffisants pour donner un sens au sujet d'un sujet. Donc, pour vous aider à comprendre le sujet, vous pouvez trouver les documents auxquels un sujet donné a le plus contribué et en déduire le sujet en lisant ce document


# Group top 5 sentences under each topic
sent_topics_sorted = pd.DataFrame()

sent_topics_outdf_grpd =sent_topics_df.groupby('Topic Dominant')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorted= pd.concat([sent_topics_sorted, grp.sort_values(['Contrib Topic'], ascending=[0]).head(1)], 
                                    axis=0)

# Reset Index    
sent_topics_sorted.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorted=sent_topics_sorted.drop(["Num Document"],axis=1)
# Show
sent_topics_sorted.head(20)


def format_date(date):
    return  dateparser.parse(str(date)).date()

whole_data_set["New_Date"] = whole_data_set["date"].apply(format_date)


whole_data_set.sample(7)


#recupere les dates sous formes de datime,on cherche juste les documents dont les indices sont "idx_doc"
dates=pd.to_datetime((whole_data_set.iloc[idx_doc]["New_Date"].values)[:-2])

#dates=pd.to_datetime(dataset["date"].values)#recupere les dates sous formes de datime

#document topic distribution :la probabilité des topics pour chaque document
doc_topic_dist = best_model.transform(tfidf_train_features) 

labels=getTopicTerms(topics) #recupere les mots des topics
labels=[",".join(topic_term[:5]) for topic_term in labels] #recupere juste les 3 premieres mots

#formation 'un dataframe qui contient la date,le numero et le text de chaque documents
df=pd.DataFrame({"text":corpus[:-2],"Date":dates,"doc_num":np.arange(len(corpus)-2)})

stories=df.groupby("doc_num")["text","Date"].min().reset_index() #trie les articles selon les dates

#formation d'un dataframe qui contient juste le numero de chaque document du corpus
story_topics=pd.DataFrame(dict(doc_num=np.arange(doc_topic_dist.shape[0])))


#recuperation des poids de chaque sujets dans chaque documents,puis on cree un une colonne dans le dataframe 
#qui va contenir l'ensemble de ces poids pour chaque sujets
for idx in range(len(labels)):
    story_topics[labels[idx]] = doc_topic_dist[:, idx]


#concatenations des dataframes par rapport au numero de documents
trends = stories.merge(story_topics, on='doc_num')

mass = lambda x: ((x) * 1.0).sum() / x.shape[0] #fonction qui calcule la moyenne 
window = 10
aggs = {labels[17]: mass,labels[4]:mass,labels[23]:mass,labels[16]:mass,labels[11]:mass}

#regroupe les poids par date,puis calcule la moyenne de chaque groupe,puis ouvre une fenetre glissant et enfin calcul la moyenne
data=trends.groupby(trends['Date'].dt.date).agg(aggs).rolling(window).mean()


plt.figure(figsize=(15,8))
sns.lineplot(data=data, palette="tab10", linewidth=2.5)

 
# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=749251a5-b032-43b6-9ff6-da57d492ca8d' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>





