# pickle
import pickle
# utilities
import re
import numpy as np
import pandas as pd
#string
from string import punctuation


# plotting
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# nltk

import nltk
from nltk.tag import PerceptronTagger
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
# sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

data = 'sad sad sad sad sad'

def get_text(data):

    data=data.lower()
    


#5.2 REMOVE STOP WORDS
def cleaning_URLs(data):
    return re.sub('((www.[^s]+)|(http?://[^s]+)|(https?://[^s]+))','',data)
data=cleaning_URLs(data)



#5.3 REMOVE STOP WORDS
stopwords1=stopwords.words('english')

def cleaning_stopwords(data):
    return " ".join([word for word in str(data).split() if word not in stopwords1])
data=cleaning_stopwords(data)



# 5.4 CLEAN AND REMOVE PUNCTUATION
punctuations_list = punctuation

def cleaning_punctuations(data):
    translator = str.maketrans('', '', punctuations_list)
    return data.translate(translator)
data=cleaning_punctuations(data)


    
#5.5 CLEANING NUMBERS
    
def cleaning_numbers(data):
    return re.sub('[0-9]+','',data)
    
data=cleaning_numbers(data)


#5.6 TOKENIZATION OF TWEET TEXT

tweet_tokenizer = TweetTokenizer(
preserve_case = True,
reduce_len    = False,
strip_handles = False)

tweet=data

tokens = tweet_tokenizer.tokenize(tweet)

data = tokens

print(data)

    

#5.8 APPLY LEMMATIZER

tags = pos_tag(data)


print(tags)


def _tag2type(tag):

    if tag.startswith('NN'):
        return 'n'
    elif tag.startswith('VB'):
        return 'v'
    else:
        return 'a'

lemmatizer = WordNetLemmatizer()



print(tags,'asjdhfksjdf',data)



        

data = [lemmatizer.lemmatize(t[0], _tag2type(t[1])) for t in tags] 


print(data)

# X = data


# X_train, X_test, y_train, y_test = train_test_split(X,test_size = 0.1, random_state =50)

#7.1 TRANSFORMING DATASET USING TF-IDF VECTORIZER



vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
# vectoriser.fit(' '.join(data))
vectoriser.fit(data)

print('yooooo')



#7.2 TRANSFORM THE DATA USING TF-IDF VECTORIZER

#data = vectoriser.transform(' '.join(data))
data = vectoriser.transform(data)

print(data)

file = open('/Users/naickercreason/dev/Final_Project_KCD/ML model building/LRmodel.pickle', 'rb')

info = pickle.load(file)

file.close()


# data = np.array(data)

# data.reshape(-1,1)

print(info.predict(data))


# X_train = vectoriser.transform(X_train.apply(lambda x: ' '.join(x)))
# X_test  = vectoriser.transform(X_test.apply(lambda x: ' '.join(x)))






