
__author__ = 'Debjit Paul'
"""
This class uses Logistic Regression 
"""
#!/usr/bin/env python3
# Import data and modules
import numpy as np
from sklearn import linear_model
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import argparse
from sklearn.ensemble import AdaBoostClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--train_path',default='/home/debjit/Documents/offline_challenge/offline_challenge/data_80_20/xtrain_obfuscated.txt',help="training data file containing descriptions")
parser.add_argument('--train_label_path',default='/home/debjit/Documents/offline_challenge/offline_challenge/data_80_20/ytrain.txt',help="training labels")
parser.add_argument('--valid_path',default='/home/debjit/Documents/offline_challenge/offline_challenge/data_80_20/xvalid_obfuscated.txt',help="validation data file containing descriptions")
parser.add_argument('--valid_label_path',default='/home/debjit/Documents/offline_challenge/offline_challenge/data_80_20/yvalid.txt',help="validation labels")
parser.add_argument('--output_dir',default='/home/debjit/Documents/offline_challenge/offline_challenge/data_80_20/',help="output directory")
parser.add_argument('--ngram',default=5,help="ngram level")

args = parser.parse_args()

train_file = args.train_path
train_label_file = args.train_label_path
valid_file = args.valid_path
valid_label_file = args.valid_label_path
ngram = int(args.ngram)
output_filename = args.output_dir + 'Logistic Regression_' + str(ngram) +'_output.txt'

def _read_lines(filename):
   data = []
   # read line by line
   with open(filename, 'rb') as f:
      for line in f:
         line = line.rstrip().decode('ascii')
         data.append(line)
   return data


train_data = _read_lines(train_file)
train_label = _read_lines(train_label_file)
valid_data = _read_lines(valid_file)
valid_label = _read_lines(valid_label_file)


# Setting up Bag of Words Model
count_vect = CountVectorizer(lowercase =False, decode_error='ignore', analyzer='char', ngram_range= (1,ngram))
X_train_counts = count_vect.fit_transform(train_data)
X_test_counts = count_vect.transform(valid_data)

# Fitting tdidf vectorization without IDF but with normalization
tfidf_transformer = TfidfTransformer(use_idf=False)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)


# feature selection using Univariate Feature Selection
selector = SelectPercentile(f_classif, percentile=30)
selector.fit(X_train_tfidf,train_label)
X_train_tfidf = selector.transform(X_train_tfidf)
X_test_tfidf = selector.transform(X_test_tfidf)

# applying Multinominal Classifier on feature vectors obtained
ada = AdaBoostClassifier(n_estimators=10)
ada.fit(X_train_tfidf, train_label) 

#clf = MultinomialNB().fit(X_train_tfidf, train_label)
predicted = ada.predict(X_test_tfidf)
print(predicted)

# Generating Model Performance Report
res = np.mean(predicted == valid_label)
print("Validation Accuracy = {}".format(res))

with open(output_filename, 'w') as f:
   f.write("Number of training data = %d \n" % (len(train_data)))
   f.write("Number of validation data = %d \n" % (len(valid_data)))
   f.write("Validation Accuracy = %g" % (res))
   f.close()

