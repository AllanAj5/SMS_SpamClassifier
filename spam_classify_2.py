# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 23:42:22 2021

@author: Adrien
"""

import pandas as pd

messages = pd.read_csv('C:/Users/Adrien/.spyder-py3/Allan Python Files/NLP/Spam Classification/SMSSpamCollection',
            sep='\t',
            names = ['class','msg'],
            header=None)

# Data Cleaning

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re



wn_Lemmatizer = WordNetLemmatizer()
corpus = []

for i in range (0,len(messages)):
    review = re.sub('[^a-zA-Z]', 
                    ' ',
                    messages['msg'][i])
    review = review.lower()
    review = review.split()
    review = [wn_Lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Text to Vectors

from sklearn.feature_extraction.text import TfidfVectorizer

tf_vec =  TfidfVectorizer(max_features=4500,min_df=2)

X = tf_vec.fit_transform(corpus).toarray()  


y = pd.get_dummies(messages['class'])
y = y.iloc[:,1]

# Train test Split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2, random_state=5)


# Make Model from Vector Data

from sklearn.naive_bayes import BernoulliNB
classifier_model = BernoulliNB()

classifier_model.fit(X_train,y_train)


y_pred = classifier_model.predict(X_test)

# Verifying Model Outputs

from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test,y_pred)

print("You Have an Accuracy of {0} %".format(accuracy*100))
