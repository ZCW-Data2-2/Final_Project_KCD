import json
import logging
import requests
from confluent_kafka import Producer, KafkaError
import json
import ccloud_lib
import pandas as pd
from secrets import retrieve_secrets
import numpy as np
import datetime

from sqlalchemy import create_engine

secrets = retrieve_secrets()
engine = create_engine(
    f"postgresql://postgres:{secrets['sql_password']}@tweets-sentiment.cwbdzmao4m7w.us-east-1.rds.amazonaws.com/tweets_sentiment_db")
d = datetime.datetime.utcnow()
history = datetime.timedelta(6)
d = d - history
dt = f"{d.isoformat()}Z"
print(dt)
tweet_query = "mask mandate  lang:en"
tweet_auth = f"Bearer {secrets['bearer_token']}"
headers = {'Authorization': tweet_auth, 'Accept': '*/*',
           'Accept-Encoding': 'gzip, deflate, br', 'Connection': 'keep-alive'}
params = {'query': tweet_query, 'start_time': '2022-01-19T10:35:21Z', 'end_time': '2022-01-19T18:14:53Z',
          'tweet.fields': 'created_at', 'max_results': '100'}
url = "https://api.twitter.com/2/tweets/search/recent"


def make_request(url, params, headers, producer):
    response = requests.get(url=url,
                            params=params,
                            headers=headers
                            )
    response_dict = response.json()
    # print(response_dict)
    data_dict = response_dict['data']
    meta_dict = response_dict['meta']
    df = pd.DataFrame(data_dict)
    df = df[['id', 'text', 'created_at']]
    # print(df['text'])
    # print(df)
    # df['sentiment'] = "POS"
    # df['sentiment'] = np.where(df['text'].str.len() > 138, "POS", "NEG")
    # print(df.head(1))
    df.to_sql('tweets2', engine, index=False, if_exists="append")

    params['next_token'] = meta_dict['next_token']

    for index, row in df.iterrows():
        record_value = json.dumps({'id': row['id'], 'text': row['text']})
        print("Producing record: {}".format(record_value))
        producer.produce(topic, value=record_value, on_delivery=acked)
        # p.poll() serves delivery reports (on_delivery)
        # from previous produce() calls.
        producer.poll(0)

    producer.flush()

    return params


def acked(err, msg):
    if err is not None:
        print("Failed to deliver message: {}".format(err))
    else:
        print("Produced record to topic {} partition [{}] @ offset {}"
              .format(msg.topic(), msg.partition(), msg.offset()))


if __name__ == '__main__':
    config_file = "./.confluent/python.config"
    topic = "tweets"
    conf = ccloud_lib.read_ccloud_config(config_file)
    producer_conf = ccloud_lib.pop_schema_registry_params_from_config(conf)
    producer = Producer(producer_conf)



    # while True:
    params = make_request(url, params, headers, producer)
