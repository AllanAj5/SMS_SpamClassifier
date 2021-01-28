# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 19:52:40 2021

@author: Adrien
"""

# Dataset Obtained From : http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/

import pandas as pd
messages = pd.read_csv('C:/Users/Adrien/.spyder-py3/Allan Python Files/NLP/Spam Classification/SMSSpamCollection',
                       sep='\t',
                       names = ['label','message'])

import nltk
#nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re

#### DATA CLEANING
p_stemmer = PorterStemmer()
corpus =[]

for i in range(0,len(messages)):
    review = re.sub('[^a-zA-Z]',' ',messages['message'][i]) #Keeping only alphabets
    review = review.lower() # coverting to LowerCase
    review = review.split() # Breaking Sentences into Words
    # Checking if the words are present in stopWords List and only if they are
    # NOT present , sending that word to be Stemmed / lemmatized
    review = [p_stemmer.stem(word) for word in review if word not in set(stopwords.words('english'))]
    # Joining back the lemmatized or Stemmed words back into a sentence
    review = " ".join(review)
    # Appending Sentences into a new List
    corpus.append(review)
    
    
#### Converting Text Data to Model Understandable VECTORS
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)  # Add Min DF and check 
X = cv.fit_transform(corpus).toarray()

# Converting Dependant Yi to Numerical value
y = pd.get_dummies(messages['label'])  # Fill The Row that Contains the Categorical True Value to 1
y =y.iloc[:, 1] # Get the Values for SPAM alone
                # iloc gets us the value for a particular row and Column


##### Divide  Data to Train and Test

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#### Training Our Model
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB()
spam_detect_model.fit(X_train,y_train)

#### Testing our Model for predicting Unseen Data
y_pred = spam_detect_model.predict(X_test)

#### Veryfying with Confusion Matrix
from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y_test,y_pred)

#### Double Verifying with Accuracy Score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)

print("Your Model is {0}% Accurate".format(accuracy))


