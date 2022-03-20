import pandas as pd
import numpy as np

#Import data and split into 70%, 15%, 15%
#NOTE: Maybe randomize?
df= pd.read_csv('Combined_News_DJIA.csv',encoding='ISO-8859-1')
train = df.iloc[:1392,:]
development = df.iloc[1393:1690]
test=  df.iloc[1690:,:]

# REMOVING PUNCTUATIONS
data= train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

# RENAMING COLUMN NAME FOR EASE OF ACCESS
list1=[i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns= new_Index

#LOWERCASE
for index in new_Index:
    data[index]=data[index].str.lower()

#Bag of words for each headline to convert into vectors
headlines=[]
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))

print(headlines)
#print(data.head())

#converts the training data into a numpy array of shape 
#(num_of_training_examples x num_of_features), uses TF-IDF with default
#stopword removal, text normalization, and sublinearTF (replace tf with 1 + log(tf))
def ownTFIDFVectorizer(dataframe, maxFreq, minFreq, stopwords=True, normalization=True, sublinearTf=True):
    
    return