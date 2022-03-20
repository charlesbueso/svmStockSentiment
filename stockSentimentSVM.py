import pandas as pd
import numpy as np

#Import data and split into 70%, 15%, 15%
#NOTE: Maybe randomize?
df= pd.read_csv('Combined_News_DJIA.csv',encoding='ISO-8859-1')
train = df.iloc[:1392,:]
development = df.iloc[1393:1690]
test=  df.iloc[1690:,:]

#Stop Words hashmap
with open("stopWords.txt", "r") as f:
    stopWords = {k:v for k, *v in map(str.split, f)}  

#Remove punctuation
data= train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

#Rename column name for ease of access, and lowercase words
list1=[i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns= new_Index
for index in new_Index:
    data[index]=data[index].str.lower()

#Bag of words for each headline to convert into vectors
headlines=[]
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))

#converts the training data into a numpy array of shape 
#(num_of_training_examples x num_of_features), uses TF-IDF with default  (HEADLINES)1392*28196(FEATURES)
#stopword removal and sublinearTF (replace tf with 1 + log(tf))
def ownTFIDFVectorizer(dataframe, stopwords=True, sublinearTf=True):
    
    vocabulary = set([]) #len 28196 without stopwords
    for i in dataframe:
        for word in i.split():
            if stopwords == True and word not in stopWords:
                    vocabulary.add(word)
            elif stopwords == False:
                vocabulary.add(word)
    vocabularyList = list(vocabulary)            
    trainVectorized = np.zeros((len(dataframe),len(vocabulary)))        
    
    #Numpy array headers
    #index = 1
    #for x in vocabulary:
     #   trainVectorized[0,index] = x
      #  index +=1
    
    #index = 1
    #for x in dataframe:
     #   trainVectorized[index,0] = x
      #  index += 1
    
    
    rowIndex = 0
    for headline in dataframe:
        #Dictionary with frequency of words for each headline
        freqDict = {}
        for word in headline.split():
            if word in freqDict:
                freqDict[word] = freqDict[word] + 1
            else: freqDict[word] = 1
            
        #Populate with term frequencies TF 
        for colIndex in range(len(vocabularyList)):
            if vocabularyList[colIndex] in freqDict:
                trainVectorized[rowIndex,colIndex] = freqDict[vocabularyList[colIndex]]
                    
        rowIndex += 1

    return trainVectorized, not np.any(trainVectorized), trainVectorized.shape

print(ownTFIDFVectorizer(headlines))