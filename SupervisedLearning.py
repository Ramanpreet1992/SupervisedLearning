# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 10:29:03 2019

@author: Owner
"""
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score  
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import decomposition, ensemble
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
from sklearn.pipeline import Pipeline  
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2 
from xml.etree import ElementTree
from sklearn.preprocessing import LabelEncoder
import nltk
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
from string import digits
from nltk .stem import  SnowballStemmer

stop_words=set(stopwords.words('english'))
snowball = SnowballStemmer(language = 'english')
#Setting the home path
path = 'C:\\Users\\Owner\\Documents\\Machine Learning\\Data2'
#Declaring the variables
head=""
text=""
code="" 
pdate=""
item=""
data=[]
bio=[]
data1=[]
arrayb=[]
arraystop=[]
textnum=""
new_stop=['though','said','was','would','well','per','respectively']

#Feature
stop_words.update(new_stop)




def PDataframe(file):
            text=""
            #Declaring array to store the row values to be inserted in the pandas dataframe
            array1=[]
            #Declaring the array to store the Bio:Topics code for the given file/news
            arrcode=[]
            bio1=[]
            d=ElementTree.parse(file)
            
            code2=''
            #Root Name
            root=d.getroot()  
            #Fetching the value of the Item attribute
            item=root.get('itemid')
            for c in root:
                count1='0'
            #Fetch Headline
                if(c.tag=='headline'):
                    head=c.text
            #Fetch Text
                if(c.tag=='text'):
                    for b in c:
                        if(text is None):
                            text=b.text
                        else:
                            text=text+" "+b.text
            #Fetching the bio:Topis code
                if(c.tag=='metadata'):
                    for e in c:
                        metacode=e
                        code=e.get('class')
                        date=e.get('element')
                        for d in e:
                            code1=d.get('code')
                            if(code=='bip:topics:1.0'):
                                g=e.findall('code')
                                if(count1=='0'):
                                    code2=d.get('code')

                                    count1=1
                                
                                for f in d:
                                    topics=[]
                                    date=f.get('date')
                                    action=f.get('action')
                                    attribution=f.get('attribution')
                                    topics.append(code1)
                                    bio.append(topics)
                                arrcode.append(topics)
                                 #Fetch Date
                        if(date=='dc.date.published'):
                            pdate=e.get('value')


            arrayb.append(code2)
            
            array1.append(head)
            array1.append(text)          
            array1.append(arrcode)
            array1.append(pdate)
            array1.append(item)
            array1.append(file)
            data.append(array1)
            
            #Stoptext array
            text=StopText(text)
            arraystop.append(text)
            dfstop=pd.DataFrame(arraystop,columns=['Text'])
            df2=Bip_Topics(arrayb)
            df2=pd.DataFrame(arrayb,columns=['Code'])
            df2=Bip_Topics(arrayb)

            
            df=pd.DataFrame(data,columns=['Headline','Text','Bip_Topics','Dc.date.published','itemid','XMLfilename'])
            print(df)
            print(df2)
            print(dfstop)
            

            return dfstop,df2


#Extracting the labels
##def Labels(df):
    
def Bip_Topics(arrayb):
    
     df2=pd.DataFrame(arrayb,columns=['Code'])
     return df2
#Cleaning the data
def StopText(text):
    text=text.lower()
    word_tokens = word_tokenize(text)
    text = [w for w in word_tokens if not w in stop_words]
    text=TreebankWordDetokenizer().detokenize(text)
    return text
#Data Preprocessing  
def FeatureExtraction(df,df2):
    data1=[]
    arrayt=df['Text']
    
    i=0
    len(arrayt)
    while i<len(arrayt):
      value1=''
      value=re.sub('-','',arrayt[i])
      value=re.sub(',','',value)
      value=re.sub('()','',value)
      value = re.sub(r'\d+','',value)
      value=re.sub('()','',value)
      #Data Cleansing
      value = re.sub(r'[?|$|.|:|*|!|#|%|&|!|+|=|-|\|}|{|<|>|@|/|;|".|`]',r'',value)
      value=re.sub(' \w{1,1} ', '', value)
      value=re.sub(r' \w{1} |^\w{1} | \w{1}$', '', value)
      value=value.replace('(','').replace(')','').replace(',','').replace('-','').replace(',','').replace('.','').replace(':','').replace('*','').replace("'",'')
      array=[]
      array.append(i)
      array.append(value)
      word_tokens = word_tokenize(value)
      #Stemming
      for w in word_tokens:
          value=snowball.stem(w)
          value1+=value+" "
     
      array.append(value1)
      data1.append(array)
      i=i+1
      
      df3=pd.DataFrame(data1,columns=['Key','Value','Stem'])
      print(df3)
     
    FeatureandLabels(df3,df2)

    return df3
#Feature Extraction
def FeatureandLabels(df3,dfc):
    cv=CountVectorizer()
    word_count_vector=cv.fit_transform(df3['Stem'])  
    vectorizer = CountVectorizer()
    #X = vectorizer.fit_transform(df3['Stem'])
    #list(cv.vocabulary_.keys())[:1000]
    bag_of_words = vectorizer.fit_transform(df3['Stem'])        
    feature_names = vectorizer.get_feature_names()
    df=pd.DataFrame(bag_of_words.toarray(),columns=feature_names)
    vectorizertf = TfidfVectorizer(analyzer='char',use_idf=True)
    tf_idf = vectorizertf.fit_transform(df3['Stem'])
    features = vectorizertf.get_feature_names()
    df4=pd.DataFrame(tf_idf.toarray(),columns=features)
    print(df4)
    dfbio=dfc
 
    #Extracting the top most best featues from bag of words and tfidf
    chi2_features = SelectKBest(chi2, k = 1500) 
    df = chi2_features.fit_transform(df, dfbio)
    
    TrainingandTestdata(df4,df,dfbio)
    
    #return df
def TrainingandTestdata(df4,df,dfbio):
    
    # split the dataset into training and validation datasets 
    trainy=dfbio.values.ravel()
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df, trainy)
    TrainClassifier(df4,df,trainy)
    
def TrainClassifier(df4,df,trainy):
    # label encode the target variable 
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df, trainy,random_state=1)
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)
     # Train the data on a classifier #Naive Bayes
    classifier = Pipeline([
    ('clf', OneVsRestClassifier(MultinomialNB(
                    fit_prior=True, class_prior=None)))])
    feature_vector_train=train_x
    is_neural_net=False
    label=train_y
    feature_vector_valid=valid_x
    classifier.fit(feature_vector_train, label)
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    MeasureScores(predictions,valid_y)
    MultipleClassifier(df,trainy)

    

def MeasureScores(predictions,valid_y):
    accuracy = metrics.accuracy_score(predictions, valid_y)
    f1_score =metrics.f1_score(predictions, valid_y,average="macro")
    precision_score =metrics.precision_score(predictions, valid_y,average="macro")
    recall_score=metrics.recall_score(predictions, valid_y,average = "macro")
    print("Training on the classifier: ", accuracy)
    print("Training on the classifier: ", f1_score)
    print("Training on the classifier: ", precision_score)
    print("Training on the classifier: ", recall_score)
  #  print("Training on the classifier: ", roc_auc_score)   
def MultipleClassifier(df,trainy):
    
    
    
    
    
    models = [
            RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),         
            DecisionTreeClassifier(criterion = "entropy",random_state=0,min_samples_split=4),    
            svm.SVC(random_state=0),
            LogisticRegression(random_state=0),
            MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
            ]
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, df, trainy, scoring='accuracy', cv=CV)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'folds', 'accuracy'])
    print(cv_df)
    import seaborn as s
    s.boxplot(x='model_name', y='accuracy', data=cv_df)
    s.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
    plt.show()
    TuneParameters(df,trainy)  
    
#GridSearch to tune the parameters
    # Creating the hyperparameter grid 
def TuneParameters(df,trainy):

    entries = []
        #SVM Tuning
    parameter_candidates = [
            {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
            {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]
    # Create a classifier object with the classifier and parameter candidates
    clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, cv=10)
    model_name='SVM'
    
    # Train the classifier on data1's feature and target data
    clf.fit(df, trainy)  
    print('Best Parameters:',clf.best_estimator_) 
    print(clf.best_score_)
    accuracy=clf.best_score_
    entries.append((model_name, accuracy))
    
    # Decision Tree Tuning
    sample_split_range = list(range(10,500,2000))
    parameter_candidates = dict(min_samples_split=sample_split_range)
    model_name='DecisionTreeClassifier'
    # instantiate the grid
    clf = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=parameter_candidates, cv=10, scoring='accuracy')

    # fit the grid with data
    clf.fit(df, trainy)
    print('Best Parameters:',clf.best_estimator_) 
    print(clf.best_score_)
    accuracy=clf.best_score_
    entries.append((model_name, accuracy))
    #Random Forest
    model_name='RandomForestClassifier'
    parameter_candidates={ 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'criterion' :['gini', 'entropy']
        }
    clf = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=parameter_candidates, cv=10, scoring='accuracy')
    clf.fit(df, trainy)  
    print('Best Parameters:',clf.best_estimator_) 
    print(clf.best_score_)
    accuracy=clf.best_score_
    entries.append((model_name, accuracy))

    #Logistic Regression
    model_name='LogisticRegression'
    parameter_candidates = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
    clf = GridSearchCV(estimator=LogisticRegression(C=1.0, intercept_scaling=1,   
               dual=False, fit_intercept=True, penalty='l2', tol=0.0001), param_grid=parameter_candidates, cv=10, scoring='accuracy')
    clf.fit(df, trainy)  
    print('Best Parameters:',clf.best_estimator_) 
    print(clf.best_score_)
    accuracy=clf.best_score_
    entries.append((model_name, accuracy))
    #Neural Networks
    model_name='Neural Network'
    parameter_candidates = {'solver': ['lbfgs'], 'max_iter': [1000 ], 'alpha': 10.0 ** -np.arange(1, 10), 'hidden_layer_sizes':np.arange(10, 15), 'random_state':[0]}
    clf = GridSearchCV(estimator=MLPClassifier(), param_grid=parameter_candidates, cv=10, scoring='accuracy')
    clf.fit(df, trainy)  
    print('Best Parameters:',clf.best_estimator_) 
    print(clf.best_score_)
    accuracy=clf.best_score_
    entries.append((model_name, accuracy))
    
    cv_df = pd.DataFrame(entries, columns=['model_name', 'accuracy'])
    print(cv_df)
    import seaborn as s1
    s1.boxplot(x='model_name', y='accuracy', data=cv_df)
    s1.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
    plt.show() 

#Traversing through the Files in the mentioned path
count = len(glob.glob1(os.getcwd(),"*.xml"))
for r, d, f in os.walk(path):
 for file in f:
        if '.xml' in file:
            #files.append(os.path.join(r, file))
            df,df2=PDataframe(file)
           
            
            #print(df.Bip_Topics.unique())  
            #Display uniques Bio Topics
            #df2.drop_duplicates().to_excel('test7.xlsx',sheet_name='sheet2',index=False)
            #print(file) 
            #print(df)
           # print(df2)

#Transferring the data to excel for validation
   
df3=FeatureExtraction(df,df2)
#df3.to_excel('test40.xlsx', sheet_name='sheet1', index=False)
#df2.to_excel('test41.xlsx', sheet_name='sheet1', index=False)
     