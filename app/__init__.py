import pandas as pd
import numpy as np

df = pd.read_csv('D:/Users/ASUS-X509J/Desktop/ExcaliburAiProject/Tweets.csv', sep=',')

df = df.drop(['negativereason_gold', 'tweet_coord', 'retweet_count'], axis=1)
print(df.head())


