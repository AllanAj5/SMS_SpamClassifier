# SMS_SpamClassifier

Steps followed  

1 - Read CSV using Pandas  
2 - Clean Data\
 &nbsp;&nbsp;->RE - remove numbers special char  
 &nbsp;&nbsp;-> - lowercase conversion  
 &nbsp;&nbsp;-> - split sentences to words  
 &nbsp;&nbsp;-> - cross check words agains english StopWords  
 &nbsp;&nbsp;-> - stem the words  
    
3 - Make Text Ready for Model by making them numeric\
    &nbsp;&nbsp;1 - Pass the corpus to CountVectorizer to fit it and return vectors (Bag Of Words) . Now Independant Variables are obtained\
    &nbsp;&nbsp;2 - To Obtain Yi or Dependant Variables , use pd.get_dummies and then take the column of Spam using iloc  
    
4 - Train the Model\
   &nbsp;&nbsp;1 - Now that Xi and Yi are obtained , Use Train_Test_split and divide date as Train and Test\
   &nbsp;&nbsp;2 - Use MultinominalNB as it is a good classifier and Fit the train data  
   
5 - test the Model\
  &nbsp;&nbsp;1 - Pass the X_test to the model's predict function and get the y_pred\
  &nbsp;&nbsp;2 - use confusion matrix and check if y_test and y_pred are similar\
  &nbsp;&nbsp;3 - use accuracy score to get a % of accuracy ie % of correctness of Model  
