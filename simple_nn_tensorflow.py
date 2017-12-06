# Implementation of a simple MLP network with one hidden layer.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier

import itertools 
RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)
import argparse


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat

def get_data():
    """ Read the data set and split them into training and validation sets """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',default='/home/debjit/Documents/offline_challenge/offline_challenge/xtrain_obfuscated.txt',help="training data file containing descriptions")
    parser.add_argument('--train_label_path',default='/home/debjit/Documents/offline_challenge/offline_challenge/ytrain.txt',help="training labels")
    parser.add_argument('--valid_path',default='/home/debjit/Documents/offline_challenge/offline_challenge/xvalid_obfuscated.txt',help="validation data file containing descriptions")
    parser.add_argument('--valid_label_path',default='/home/debjit/Documents/offline_challenge/offline_challenge/yvalid.txt',help="validation labels")
    parser.add_argument('--output_dir',default='/home/debjit/Documents/offline_challenge/offline_challenge/',help="output directory")
    parser.add_argument('--ngram',default=4,help="ngram level")
    args = parser.parse_args()

    train_file = args.train_path
    train_label_file = args.train_label_path
    valid_file = args.valid_path
    valid_label_file = args.valid_label_path
    ngram = int(args.ngram)
    #output_filename = args.output_dir + 'Naive_Bayes_BagOfCharacters_ngram_' + str(ngram) +'_output.txt'
    train_data = _read_lines(train_file)
    train_label = _read_lines(train_label_file)
    valid_data = _read_lines(valid_file)
    valid_label = _read_lines(valid_label_file)   
    #train_X=_extract_char(train_data)
    #test_X=_extract_char(valid_data)
    train_y=_labels_vector(train_label)
    test_y=_labels_vector(valid_label)
    
    # Setting up Bag of Words Model
    count_vect = CountVectorizer(lowercase =False, decode_error='ignore', analyzer='char', ngram_range= (1,ngram))
    X_train_counts = count_vect.fit_transform(train_data)
    X_test_counts = count_vect.transform(valid_data)

#count_vect_word = CountVectorizer(lowercase =False, decode_error='ignore', analyzer='word', ngram_range= (3,ngram))

# Fitting tdidf vectorization without IDF but with normalization
    tfidf_transformer = TfidfTransformer(use_idf=False)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)


# feature selection using Univariate Feature Selection
    selector = SelectPercentile(f_classif, percentile=30)
    selector.fit(X_train_tfidf,train_label)
    X_train_tfidf = selector.transform(X_train_tfidf)
    X_test_tfidf = selector.transform(X_test_tfidf)
    return(X_train_tfidf, X_test_tfidf, train_y, test_y)


def _labels_vector(labels):
  vector=[]
  max_value=max(int(l) for l in labels)
  print(len(labels),max_value)
  for i in range(len(labels)):
          vector.append([0]*(max_value+1))
  for i in range(len(labels)):
       for j in range(len(vector[i])):
           if int(labels[i])==j:
               vector[i][j]=1
  return(vector)

##reading the data
def _read_lines(filename):
   data = []
   # read line by line
   with open(filename, 'rb') as f:
      for line in f:
         line = line.rstrip().decode('ascii')
         data.append(line)
   return data


char_id={}
char_embedd={}
                         ###extracting character from the documents
def _extract_char(data):
   max_length=max(len(l) for l in data)
   print(max_length)
   for doc in range(len(data)):
       for char in range(len(data[doc])):            
             if data[doc][char] not in char_id:
                    char_id[data[doc][char]]=len(char_id)

   char_embedd=_char_one_hot_vector(char_id)  
   doc_embedd=[]
   sentence_embedd=[]
   
   for doc in range(len(data)):
       sentence_embedd=[]
       for char in range(len(data[doc])):
          sentence_embedd=list(itertools.chain(sentence_embedd,char_embedd[data[doc][char]]))
       if len(data[doc])==max_length:  
           print('aww',doc,len(sentence_embedd))      
           doc_embedd.append(sentence_embedd)
                                                   ########padding######
       else:
             s=max_length-len(data[doc])            
             for i in range(s):
                sentence_embedd=list(itertools.chain(sentence_embedd,[0]*len(char_id)))
             print('wow',doc,len(sentence_embedd)) 
             doc_embedd.append(sentence_embedd)               
        
   return(doc_embedd)


###one_hot vector                               
def _char_one_hot_vector(char_id):
      char_embedd={}
      for key in char_id.keys():
         char_embedd[key]=[0]*len(char_id)
         for j in range (len(char_embedd[key])):
              if j==char_id[key]:
                char_embedd[key][j]=1
      return(char_embedd) 
   

def main():
    train_X, test_X, train_y, test_y = get_data()
    
    # Layer's sizes
    x_size = len(train_X[0])   
    h_size = 1            
    y_size = len(train_y[0])  

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(100):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_X, y: train_y}))
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: test_X, y: test_y}))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    sess.close()

if __name__ == '__main__':
    main()
