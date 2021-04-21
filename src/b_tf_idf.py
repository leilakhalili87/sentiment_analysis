# Therefore, the performance is similar to the "bag of words" model.

# Importing libraries
import numpy as np
import pandas as pd
# BeautifulSoup is used to remove html tags from the text
from bs4 import BeautifulSoup 
import re # For regular expressions
from nltk.corpus import stopwords
import logging
import util_func as uf
import nltk.data
from gensim.models import word2vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import timeit
start = timeit.default_timer()

remove_stopwords = False
clean_up = False

path2train = "./P1_training.xlsx"
path2test = "./P1_testing.xlsx"

train = uf.import_xlsx(path2train, "Sheet1")
test = uf.import_xlsx(path2test, "P1_testing")
# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# sentences = []
# print("Parsing sentences from training set")
# for review in train["sentence"]:
#     sentences += uf.review_sentences(review, tokenizer, remove_stopwords, clean_up)


X_train_tfidf, tfidf_vectorizer = uf.tfidf(train['sentence'])
trainDataVecs = X_train_tfidf.toarray()

X_test_tfidf = tfidf_vectorizer.transform(test['sentence'])
testDataVecs = X_test_tfidf.toarray()


forest = RandomForestClassifier(n_estimators = 100, random_state=1366)
    
print("Fitting random forest to training data....")    
forest = forest.fit(trainDataVecs, train["label"])

# Predicting the sentiment values for test data and saving the results in a csv file 
result = forest.predict(testDataVecs)
test['predicted_label'] = result
test.rename(columns={'label':'gold_label'}, inplace=True)
test.to_csv( "./output/testing_output_tf_idf.csv", index=False )


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
cm = confusion_matrix(test['gold_label'], test['predicted_label'])
print(cm)
print("The accuracy of the model in the tested data is: ",accuracy_score(test['gold_label'], test['predicted_label']))
print("The f1 score of the model in the tested data is: ",f1_score(test['gold_label'], test['predicted_label'], average='weighted'))
stop = timeit.default_timer()
execution_time = stop - start
print("Program Executed in "+str(execution_time)) # It returns time in seconds

