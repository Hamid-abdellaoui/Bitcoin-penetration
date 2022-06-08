import pandas as pd
import os
os.getcwd()
collection = "C:/Users/hp/Desktop/PE/App/static/images/trends"

min_month = "2012-11"
max_month = "2022-5"
months = pd.period_range(min_month, max_month, freq='M')
months = months.to_timestamp(how='end').strftime('%Y-%m')


for i, filename in enumerate(os.listdir(collection)):
    os.rename("C:/Users/hp/Desktop/PE/App/static/images/trends/"+str(i)+".png", "C:/Users/hp/Desktop/PE/App/static/images/trends/"+str(months[i])+".png")