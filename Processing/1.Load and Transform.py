import os
import pandas as pd
import glob
import dateparser

############## Loading Data ##############
path = "../Data collection/Historic data/raw data/"
files = glob.glob(path + "/*.csv") 
Datasets = pd.DataFrame()
content = [pd.read_csv(filename, index_col=None) for filename in files]
Datasets = pd.concat(content)

############## Transforming Data ##############
def format_date(date):
    return  dateparser.parse(str(date)).date()
Datasets["date"] = Datasets["date"].apply(format_date)
Datasets.to_csv(r"Outpout\Datasets.csv", index=False)
formated_data=pd.read_csv(r"Outpout\Datasets.csv")
formated_data['date'] = pd.to_datetime(formated_data['date'], errors = 'coerce')

############# Cleaning Data ##############
formated_data.dropna(inplace=True)

############# Saving Data ##############
formated_data.to_csv(r"Outpout\Datasets.csv", index=False)
