import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt   #for data visualization and graphical plotting

import joblib
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize      #to divide strings into lists of substrings
from nltk.corpus import stopwords            #to filterout useless data
stopword = set(stopwords.words('english'))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import re
import string

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub(r"\@w+|\#",'',text)
    text = re.sub(r"[^\w\s]",'',text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    tweet_tokens = word_tokenize(text)
    filtered_tweets=[w for w in tweet_tokens if not w in stopword] #removing stopwords
    return " ".join(filtered_tweets)

def main(comment):
    data = pd.read_csv('labeled_data.csv', encoding='latin-1')

    data["labels"] = data["class"].map({0: "Hate Speech", 
                                        1: "Offensive Language", 
                                        2: "No Hate or Offensive"})

    data = data[["tweet", "labels"]]
    data.tweet=data['tweet'].apply(clean)
    tweetData = data.drop_duplicates("tweet")   # removing duplicate data

    #creating a trigram language model
    vect=TfidfVectorizer(ngram_range=(1,3)).fit(tweetData['tweet'])

    #separating the data into x and y to build the model
    X = tweetData['tweet']
    Y = tweetData['labels']
    X = vect.transform(X) #transforming the x data

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model_name = "comment_scanner_model"
    
    dtree = DecisionTreeClassifier()    #for training the data on decision tree classifier model
    dtree.fit(X_train, Y_train)         #loading x_train and y_train data on model
    joblib.dump(dtree, model_name)
    dtree_predict = dtree.predict(X_test) #predicting the value for test data
    dtree_acc = accuracy_score(dtree_predict, Y_test)
    
    input_string=comment
    input_vector = vect.transform([input_string])
    prediction=dtree.predict(input_vector)

    print("Test accuracy: {:.2f}%".format(dtree_acc*100)) #printing accuracy of the model

    return prediction
    