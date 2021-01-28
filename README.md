# SMS_SpamClassifier

Steps followed

1 - Read CSV using Pandas
2 - Clean Data 
    1 - RE - remove numbers special char
    2 - lowercase conversion
    3 - split sentences to words
    4 - cross check words agains english StopWords
    5 - stem the words
    
3 - Make Text Ready for Model by making them numeric
    1 - Pass the corpus to CountVectorizer to fit it and return vectors (Bag Of Words) . Now Independant Variables are obtained
    2 - To Obtain Yi or Dependant Variables , use pd.get_dummies and then take the column of Spam using iloc
    
4 - Train the Model 
   1 - Now that Xi and Yi are obtained , Use Train_Test_split and divide date as Train and Test
   2 - Use MultinominalNB as it is a good classifier and Fit the train data
   
5 - test the Model
  1 - Pass the X_test to the model's predict function and get the y_pred
  2 - use confusion matrix and check if y_test and y_pred are similar
  3 - use accuracy score to get a % of accuracy ie % of correctness of Model
