from text_mining import *
import json

######################### load the cleaned data
clean = pd.read_csv("./Outpout/Datasets.csv", parse_dates=['date'])


##############transformation date to period########################
clean['period']=clean['date'].dt.to_period('M')
clean.sort_values(by="date" ,inplace=True)

l_periods = list(set(clean['period']))

################ Diviser le corpus suivant les periods ##########
corpuses = []
l_periods=list(set(clean.period.values))
l_periods.sort()
for i in range(len(l_periods)-1) :
    indice1=clean.period.values.tolist().index(l_periods[i])
    indice2=clean.period.values.tolist().index(l_periods[i+1])
    liste = []
    for j in range(indice1,indice2,1):
        liste.append(clean.extrait.values[j]) 
    corpuses.append(liste)

dernier_indice=clean.period.values.tolist().index(l_periods[121])
derniere_corpus = [] 
for j in range(dernier_indice,len(clean),1):
    derniere_corpus.append(clean.extrait.values[j]) 
corpuses.append(derniere_corpus)


################ lower corpus #######################################
corpuses = [ lower_text(corpus) for corpus in corpuses ]

############# joining corpuses #####################################
corpuses = [ "sep_ara_tor".join(corpus) for corpus in corpuses ]

############# exporting as csv file #################################

pd.DataFrame(corpuses,columns=["corpuses"]).to_csv("./Outpout/corpuses.csv",index=False)





