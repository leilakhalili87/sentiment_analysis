from nltk.corpus import stopwords
import logging
import util_func as uf
import pandas as pd
# import nltk.data
# from gensim.models import word2vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import timeit
start = timeit.default_timer()

remove_stopwords = False
clean_up = False
sess = tf.Session()

path2train = "./P1_training.xlsx"
path2test = "./P1_testing.xlsx"

train = uf.import_xlsx(path2train, "Sheet1")
test = uf.import_xlsx(path2test, "P1_testing")


embed_data = hub.Module('./input/module/module_useT')

x_training = np.reshape(train.sentence.values,(len(train.sentence.values)))

x_test = np.reshape(test.sentence.values,(len(test.sentence.values)))


embed_fn = uf.embed_useT('./input/module/module_useT')

x_training = embed_fn(x_training)

x_test = embed_fn(x_test)


y_train = train.label.values

y_test = test.label.values


# Fitting a random forest classifier to the training data
forest = RandomForestClassifier(n_estimators = 100, random_state=1366)
forest = forest.fit(x_training, y_train)
result = forest.predict(x_test)

test['predicted_label'] = result
test.rename(columns={'label':'gold_label'}, inplace=True)
#saving the output file
test.to_csv( "./output/testing_output_proposed.csv", index=False )


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
cm = confusion_matrix(test['gold_label'], test['predicted_label'])
print(cm)
print("The accuracy of the model in the tested data is: ",accuracy_score(test['gold_label'], test['predicted_label']))
print("The f1 score of the model in the tested data is: ",f1_score(test['gold_label'], test['predicted_label'], average='weighted'))

stop = timeit.default_timer()
execution_time = stop - start
print("Program Executed in "+str(execution_time)) # It returns time in seconds