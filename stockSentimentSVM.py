import pandas as pd
import numpy as np
import math
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

#Import data and split into 70%, 15%, 15%
df= pd.read_csv('Combined_News_DJIA.csv',encoding='ISO-8859-1')
train = df.iloc[:1392,:]
development = df.iloc[1393:1690]
test=  df.iloc[1690:,:]

#Stop Words hashmap
with open("stopWords.txt", "r") as f:
    stopWords = {k:v for k, *v in map(str.split, f)}  

##Data normalization for all 3 datasets
#Remove punctuation TRAINING
data= train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
#Rename column name for ease of access, and lowercase words TRAINING
list1=[i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns= new_Index
for index in new_Index:
    data[index]=data[index].str.lower()
#Bag of words for each headline to convert into vectors TRAINING
headlines=[]
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))

#Remove punctuation DEVELOPMENT
data = development.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
#Rename column name for ease of access, and lowercase words DEVELOPMENT
list1=[i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns= new_Index
for index in new_Index:
    data[index]=data[index].str.lower()
#Bag of words for each headline to convert into vectors DEVELOPMENT
developmentHeadlines=[]
for row in range(0,len(data.index)):
    developmentHeadlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))    
    
#Remove punctuation TESTING
data = test.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
#Rename column name for ease of access, and lowercase words TESTING
list1=[i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns= new_Index
for index in new_Index:
    data[index]=data[index].str.lower()
#Bag of words for each headline to convert into vectors DEVELOPMENT
testHeadlines=[]
for row in range(0,len(data.index)):
    testHeadlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))


#Vocabulary with training data, remove stopwords
vocabulary = set([])
for i in headlines:
    for word in i.split():
        if word not in stopWords:
            vocabulary.add(word)
vocabularyList = list(vocabulary)            
      

#converts the data into a numpy array of shape 
#(num_of_training_examples x num_of_features) // training = (HEADLINES)1392*28196(FEATURES)
#sublinearTF (replace tf with 1 + log(tf)), takes in vocabularyList from training set as features 
#always takes dataframe*vocabularyList as dimension of final vector
def ownCountVectorizer(dataframe, vocabularyList, sublinearTf=False):
    
    trainVectorized = np.zeros((len(dataframe),len(vocabularyList)))  

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
                if sublinearTf == True:
                    trainVectorized[rowIndex,colIndex] = 1 + math.log(freqDict[vocabularyList[colIndex]])
                else: trainVectorized[rowIndex,colIndex] = freqDict[vocabularyList[colIndex]]
                
        rowIndex += 1

    return trainVectorized

#Building SVC
trainedVectorized = ownCountVectorizer(headlines, vocabularyList)
supportVectorClassifier = LinearSVC()
supportVectorClassifier.fit(trainedVectorized, train['Label'])

#Development predictions
developmentVectorized = ownCountVectorizer(developmentHeadlines, vocabularyList)
predictions = supportVectorClassifier.predict(developmentVectorized)

#Results
score= accuracy_score(development["Label"],predictions)
print("Development set with own CountVectorizer results: \n",score)
report= classification_report(development['Label'],predictions)
print(report)

#Test predictions
testVectorized = ownCountVectorizer(testHeadlines, vocabularyList)
predictions = supportVectorClassifier.predict(testVectorized)

#Results
score= accuracy_score(test["Label"],predictions)
print("Test set with own CountVectorizer results: \n",score)
report= classification_report(test['Label'],predictions)
print(report)

##Implementing sklearn's TFIDF Vectorizer
tfidf = TfidfVectorizer(ngram_range=(2,2))
traindataset= tfidf.fit_transform(headlines)
tfidfSupportVectorClassifier = LinearSVC()
supportVectorClassifier.fit(traindataset, train['Label'])

developmentTFIDFVectorized = tfidf.transform(developmentHeadlines)
predictionsTfidf = supportVectorClassifier.predict(developmentTFIDFVectorized)

score= accuracy_score(development["Label"],predictionsTfidf)
print("Test set with sklearn's TfidfVectorizer results: \n",score)
report= classification_report(development['Label'],predictionsTfidf)
print(report)
