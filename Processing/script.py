from text_mining import *
import seaborn as sns
import matplotlib.pyplot as plt 
import json

################ define process function ########################## 

def process_corpus(corpus,indice):
    """retourne une liste output"""
    ############# traitement général ################################
    most_commonWords=get_mostCommonWords(corpus,500)
    corpus=remove_stopwords(corpus,most_commonWords)
    corpus,idx_doc=remove_shortDocument(corpus,min_length=3)
    corpus_lemmatized=tokenize_text(corpus) 
    id2word = corpora.Dictionary(corpus_lemmatized)
    vocabulaire=id2word.token2id 
    ########### former la matrice tf-idf #############################
    tfidf = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0,sublinear_tf=True,
                        lowercase=True,ngram_range=(1,2),vocabulary=vocabulaire)
    tfidf_train_features = tfidf.fit_transform(corpus)
    feature_names = tfidf.get_feature_names()
    ################# Topic Modelling : NMF ############################################
    total_topics = 10
    pos_nmf=NMF(n_components=total_topics,random_state=42,l1_ratio=0.2,max_iter=200)
    pos_nmf.fit(tfidf_train_features) 
    pos_weights = pos_nmf.components_
    pos_topics = get_topics_terms_weights(pos_weights, feature_names)
    topic_terms=getTopicTerms(pos_topics)
    topic_terms=[topics[:20] for topics in topic_terms] 
    common_corpus = [id2word.doc2bow(text) for text in corpus_lemmatized] 
    ####### choix du meilleur model ###################################
    cm = CoherenceModel(topics=topic_terms,corpus=common_corpus, dictionary=id2word, 
                coherence='u_mass')
    coherence = cm.get_coherence()
    best_model,coherence_values=compute_coherence_values(tfidf_train_features,
    feature_names,corpus,corpus_lemmatized,id2word,max_term=20,limit=50)
    ######## résultat du modèle optimal ##############################
    total_topics=best_model.n_components 
    weights = best_model.components_ 
    topics = get_topics_terms_weights(weights,feature_names)
    ################### sujet dominant pour chaque document #############
    sent_topics_df=topic_dominant(best_model,tfidf_train_features,corpus,topics)
    
    
    #sent_topics_sorted = pd.DataFrame()
    #sent_topics_outdf_grpd =sent_topics_df.groupby('Topic Dominant')
    #for i, grp in sent_topics_outdf_grpd:
    #    sent_topics_sorted= pd.concat([sent_topics_sorted,
    #    grp.sort_values(['Contrib Topic'], ascending=[0]).head(1)], axis=0) 
    #sent_topics_sorted.reset_index(drop=True, inplace=True)
    #sent_topics_sorted=sent_topics_sorted.drop(["Num Document"],axis=1)

    ##### Data_For_Plot ########################################################
    period = l_periods[indice]
    clean_modified = clean.loc[clean.period == period , :]
    dates=pd.to_datetime(clean_modified.iloc[idx_doc]["date"].values)
    doc_topic_dist = best_model.transform(tfidf_train_features) 
    labels=getTopicTerms(topics)
    labels=[",".join(topic_term[:5]) for topic_term in labels] 
    df=pd.DataFrame({"text":corpus,"Date":dates,"doc_num":np.arange(len(corpus))})
    stories=df.groupby("doc_num")["text","Date"].min().reset_index() 
    story_topics_for_plot=pd.DataFrame(dict(doc_num=np.arange(doc_topic_dist.shape[0])))
    for idx in range(len(labels)):
        story_topics_for_plot[labels[idx]] = doc_topic_dist[:, idx]
    trends = stories.merge(story_topics_for_plot, on='doc_num')
    mass = lambda x: ((x) * 1.0).sum() / x.shape[0]  
    window = 10
    if len(labels)> 18 :
        aggs = {labels[17]: mass,labels[4]:mass,labels[17]:mass,labels[16]:mass,labels[11]:mass}
    else :
        aggs = {labels[0]: mass,labels[1]:mass,labels[2]:mass,labels[3]:mass,labels[4]:mass}
    data_for_plot=trends.groupby(trends['Date'].dt.date).agg(aggs).rolling(window).mean()

    ######### output de chaque corpus ###############################
    output=[sent_topics_df,story_topics_for_plot,data_for_plot]

    return output


#Now read the file back into a Python list object
with open('test.txt', 'r') as f:
    corpuses = json.loads(f.read())

test = process_corpus(list(corpuses[1]),1) 

test[1].drop(["doc_num"],axis=1).mean(axis=0).plot(kind='bar', title ="Importance des sujets",
                                                        figsize=(15,10), fontsize=15)
plt.xlabel("Sujet", fontsize=20)
plt.ylabel("Frequence d'apparution", fontsize=20)
plt.xticks(rotation=70)
plt.show()