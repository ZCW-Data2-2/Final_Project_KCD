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
    
    # stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
            #  'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
            #  'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
            #  'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
            #  'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
            #  'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
            #  'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
            #  'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
            #  'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
            #  't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
            #  'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
            #  'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
            #  'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
            #  'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
            #  "youve", 'your', 'yours', 'yourself', 'yourselves']


     # STOPWORDS = set(stopwordlist)
    # def cleaning_stopwords(text):
    #     return " ".join([word for word in str(text).split() if word not in STOPWORDS])
    # data['text'] = data['text'].apply(lambda text: cleaning_stopwords(text))
    # data['text'].head()


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


    
    #def cleaning_repeating_characters(text):
    #     return re.sub(r'(.)1+', r'1', text)
    #data['text']=data['text'].apply(lambda x: cleaning_repeating_characters(x))

    
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

    


#5.8 APPLY LEMMATIZER

tags = pos_tag(data)


def _tag2type(tag):

    if tag.startswith('NN'):
        return 'n'
    elif tag.startswith('VB'):
        return 'v'
    else:
        return 'a'

lemmatizer = WordNetLemmatizer()

# data = [[lemmatizer.lemmatize(word, _tag2type(tag)) for (word, tag) in t] for t in tags]

data = [[lemmatizer.lemmatize(_tag2type(tag)) for (tag) in t] for t in tags]


#5.9 SEPERATE INPUT FEATURE AND LABEL

X=data.text
Y=data.target

#above, this is not working!


type(X)
type(Y)

data_pos=data[data['target']==1]
data_neg=data[data['target']==0]

data_neg

pos_tweets=data_pos['text'].values
neg_tweets=data_pos['text'].values

plt.figured(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
            collocations=False).generate(str(neg_tweets))

plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
            collocations=False).generate(str(pos_tweets))
plt.imshow(wc)
plt.title('Word Cloud for Positive tweets')

#5.9 SEPERATE INPUT FEATURE AND LABEL

def get_all_words(tokens_list):
    
    for tokens in tokens_list:
        for token in tokens:
            yield token

all_pos_words = get_all_words(pos_tweets)
all_neg_words = get_all_words(neg_tweets)

freq_dist_pos = FreqDist(all_pos_words)
freq_dist_neg = FreqDist(all_neg_words)

#7.1 TRANSFORMING DATASET USING TF-IDF VECTORIZER

vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train.apply(lambda x: ' '.join(x)))



#7.2 TRANSFORM THE DATA USING TF-IDF VECTORIZER

X_train = vectoriser.transform(X_train.apply(lambda x: ' '.join(x)))
X_test  = vectoriser.transform(X_test.apply(lambda x: ' '.join(x)))






