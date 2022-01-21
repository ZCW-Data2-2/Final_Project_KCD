import json
import logging
import requests
from confluent_kafka import Producer, KafkaError
import json
import ccloud_lib
import pandas as pd

import numpy as np
import datetime

from sqlalchemy import create_engine

engine = create_engine(
    'postgresql://postgres:<password here>@tweets-sentiment.cwbdzmao4m7w.us-east-1.rds.amazonaws.com/tweets_sentiment_db')
d = datetime.datetime.utcnow()
history = datetime.timedelta(6)
d = d - history
dt = f"{d.isoformat()}Z"
print(dt)
tweet_query = "mask mandate  lang:en"
tweet_auth = "Bearer <token here>"
headers = {'Authorization': tweet_auth, 'Accept': '*/*',
           'Accept-Encoding': 'gzip, deflate, br', 'Connection': 'keep-alive'}
params = {'query': tweet_query, 'start_time': '2022-01-19T10:35:21Z', 'end_time': '2022-01-19T18:14:53Z',
          'tweet.fields': 'created_at', 'max_results': '100'}
url="https://api.twitter.com/2/tweets/search/recent"


def make_request(url, params, headers):
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
    return params
if __name__ == '__main__':
    # setup config
    # config_file = ".confluent/python.config"
    # topic = "tweets"
    # conf = ccloud_lib.read_ccloud_config(config_file)
    while True:
        params = make_request(url, params, headers)
    # instantiate producer
    # producer_conf = ccloud_lib.pop_schema_registry_params_from_config(conf)
