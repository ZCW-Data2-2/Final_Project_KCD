import requests
import pandas as pd

import numpy as np
import datetime
from sqlalchemy import create_engine

engine = create_engine(
    'postgresql://postgres:<password here>@tweets-sentiment.cwbdzmao4m7w.us-east-1.rds.amazonaws.com/tweets_sentiment_db')

df = pd.read_sql('SELECT * FROM tweets', engine, index_col="id")
print(df.head())
