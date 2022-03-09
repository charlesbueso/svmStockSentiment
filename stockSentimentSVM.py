import pandas as pd
import numpy as np

#Import data and split into 70%, 15%, 15%
df= pd.read_csv('Combined_News_DJIA.csv',encoding='ISO-8859-1')
train = df.iloc[:1392,:]
development = df.iloc[1393:1690]
test=  df.iloc[1690:,:]

#converts the training data into a numpy array of shape 
#(num_of_training_examples x num_of_features)
def ownVectorizer(dataframe, stopwords, maxFreq, minFreq, normalization=True):
    
    return