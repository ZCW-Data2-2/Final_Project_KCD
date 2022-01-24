#!/usr/bin/env python
#
# Copyright 2020 Confluent Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# =============================================================================
#
# Consume messages from Confluent Cloud
# Using Confluent Python Client for Apache Kafka
#
# =============================================================================

from confluent_kafka import Consumer
import json
import ccloud_lib
import pickle
import re
import numpy as np

from secrets import retrieve_secrets

from string import punctuation

from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
import psycopg2





def tag2type(tag):

    if tag.startswith('NN'):
        return 'n'
    elif tag.startswith('VB'):
        return 'v'
    else:
        return 'a'


stopwords1 = stopwords.words('english')
punctuations_list = punctuation


def cleaning_URLs(data):
    return re.sub('(www.[^\s]+)|(http?://[^\s]+)|(https?://[^\s]+)', '', data)


def clean_text(data, stopwords1, punctuations_list):

    data = data.lower()

    data = cleaning_URLs(data)

    data = " ".join([word for word in str(
        data).split() if word not in stopwords1])

    translator = str.maketrans('', '', punctuations_list)
    data = data.translate(translator)
    data = re.sub('[0-9]+', '', data)
    tweet_tokenizer = TweetTokenizer(
        preserve_case=True,
        reduce_len=False,
        strip_handles=False)

    tweet = data

    tokens = tweet_tokenizer.tokenize(tweet)

    data = tokens

    tags = pos_tag(data)
    lemmatizer = WordNetLemmatizer()

    data = [lemmatizer.lemmatize(t[0], tag2type(t[1])) for t in tags]

    data = ' '.join(data)

    data = [data]


    return data


if __name__ == '__main__':


    secrets = retrieve_secrets()

    connection = psycopg2.connect(user="postgres",
                                    password=secrets['postgres_password'],
                                    host="tweets-sentiment.cwbdzmao4m7w.us-east-1.rds.amazonaws.com",
                                    port="5432",
                                    database="tweets_sentiment_db")


    file = open('/Users/naickercreason/dev/Final_Project_KCD/ML model building/LRmodel.pickle', 'rb')
    info = pickle.load(file)
    file.close()

    file2 = open('/Users/naickercreason/dev/Final_Project_KCD/ML model building/vectoriser.pickle', 'rb')
    info2 = pickle.load(file2)
    file2.close()

    # Read arguments and configurations and initialize
    # args = ccloud_lib.parse_args()
    config_file = '/Users/naickercreason/dev/Final_Project_KCD/Kafka/.confluent/python.config'
    topic = 'tweets'
    conf = ccloud_lib.read_ccloud_config(config_file)

    # Create Consumer instance
    # 'auto.offset.reset=earliest' to start reading from the beginning of the
    #   topic if no committed offsets exist
    consumer_conf = ccloud_lib.pop_schema_registry_params_from_config(conf)
    consumer_conf['group.id'] = 'python_example_group_1'
    consumer_conf['auto.offset.reset'] = 'earliest'
    consumer = Consumer(consumer_conf)

    # Subscribe to topic
    consumer.subscribe([topic])

    # Process messages
    total_count = 0
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                # No message available within timeout.
                # Initial message consumption may take up to
                # `session.timeout.ms` for the consumer group to
                # rebalance and start consuming
                print("Waiting for message or event/error in poll()")
                continue
            elif msg.error():
                print('error: {}'.format(msg.error()))
            else:
                # Check for Kafka message
                record_value = msg.value()
                # dictionary will be id & tweet
                data = json.loads(record_value)
                text = clean_text(data['text'], stopwords1, punctuations_list)               
                text = info2.transform(text)
                prediction = info.predict(text)
                prediction = int(prediction[0])
                print(f"The sentiment for ID:{data['id']} is {prediction}")
                cursor = connection.cursor()
                sql_query = """
                    UPDATE tweets SET sentiment = %s WHERE id = %s 
                """
                cursor.execute(sql_query, (prediction, data['id']))
                connection.commit()
                # print(type(prediction))
                # print(type(data['id']))

    except KeyboardInterrupt:
        pass
    finally:
        # Leave group and commit final offsets
        consumer.close()
