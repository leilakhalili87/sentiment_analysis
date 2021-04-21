# Importing libraries
import numpy as np
import pandas as pd
# BeautifulSoup is used to remove html tags from the text

import re # For regular expressions

# Stopwords can be useful to undersand the semantics of the sentence.
# Therefore stopwords are not removed while creating the word2vec model.
# But they will be removed  while averaging feature vectors.
from nltk.corpus import stopwords
import logging
import util_func as uf
import nltk.data
from gensim.models import word2vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


import timeit
start = timeit.default_timer()

remove_stopwords = False
clean_up = False

num_features = 300
model_name = "300features_40minwords_10context"

path2train = "./P1_training.xlsx"
path2test = "./P1_testing.xlsx"

train = uf.import_xlsx(path2train, "Sheet1")
test = uf.import_xlsx(path2test, "P1_testing")


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

sentences = []
for review in train["sentence"]:
    sentences += uf.review_sentences(review, tokenizer, remove_stopwords, clean_up)


print("Training model")
model = word2vec.Word2Vec(sentences,\
                          min_count=5,
                         window=5,
                         size=num_features,
                         sample=6e-4, 
                         alpha=0.03, 
                         workers=2)

model.init_sims(replace=True)
# Saving the model for later use. Can be loaded using Word2Vec.load()
model.save(model_name)

# Calculating average feature vector for training set
clean_train_reviews = []
for review in train['sentence']:
    clean_train_reviews.append(uf.review_wordlist(review, remove_stopwords, clean_up))


trainDataVecs = uf.getAvgFeatureVecs(clean_train_reviews, model, num_features)

# Calculating average feature vactors for test set     
clean_test_reviews = []
for review in test["sentence"]:
    clean_test_reviews.append(uf.review_wordlist(review,remove_stopwords, clean_up))
    
testDataVecs = uf.getAvgFeatureVecs(clean_test_reviews, model, num_features)

# Fitting a random forest classifier to the training data
forest = RandomForestClassifier(n_estimators = 100, random_state=1366)
forest = forest.fit(trainDataVecs, train["label"])
result = forest.predict(testDataVecs)

test['predicted_label'] = result
test.rename(columns={'label':'gold_label'}, inplace=True)
#saving the output file
test.to_csv( "./output/testing_output_word2vec.csv", index=False )


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
cm = confusion_matrix(test['gold_label'], test['predicted_label'])
print(cm)
print("The accuracy of the model in the tested data is: ",accuracy_score(test['gold_label'], test['predicted_label']))
print("The f1 score of the model in the tested data is: ",f1_score(test['gold_label'], test['predicted_label'], average='weighted'))
stop = timeit.default_timer()
execution_time = stop - start
print("Program Executed in "+str(execution_time)) # It returns time in seconds