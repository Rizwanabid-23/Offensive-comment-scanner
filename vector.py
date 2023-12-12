
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize      #to divide strings into lists of substrings
from nltk.corpus import stopwords            #to filterout useless data
stopword = set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer
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

def vectorizer(comment):
    data = pd.read_csv('labeled_data.csv', encoding='latin-1')
    data["labels"] = data["class"].map({0: "Hate Speech", 
                                        1: "Offensive Language", 
                                        2: "No Hate or Offensive"})
    data = data[["tweet", "labels"]]
    data.tweet=data['tweet'].apply(clean)

    tweetData = data.drop_duplicates("tweet")       # removing duplicate data

    vect=TfidfVectorizer(ngram_range=(1,3)).fit(tweetData['tweet'])
    input_string=comment
    text = vect.transform([input_string])

    return text
    