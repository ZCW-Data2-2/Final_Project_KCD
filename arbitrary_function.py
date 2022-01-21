# pickle
import pickle
# utilities
import re
# string
from string import punctuation
# nltk
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

file = open('./ML model building/LRmodel.pickle', 'rb')
info = pickle.load(file)
file.close()

file2 = open('ML model building/vectoriser.pickle', 'rb')
info2 = pickle.load(file2)
file2.close()


data = 'i hate driving with no shoes on'
#####
def get_text(data):

    data=data.lower()
    
######

#5.2 REMOVE STOP WORDS
def cleaning_URLs(data):
    return re.sub('(www.[^\s]+)|(http?://[^\s]+)|(https?://[^\s]+)','',data)
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


tags = pos_tag(data)



def _tag2type(tag):

    if tag.startswith('NN'):
        return 'n'
    elif tag.startswith('VB'):
        return 'v'
    else:
        return 'a'

lemmatizer = WordNetLemmatizer()



data = [lemmatizer.lemmatize(t[0], _tag2type(t[1])) for t in tags] 

data = ' '.join(data)

data = [data]

data = info2.transform(data)

print(info.predict(data))








