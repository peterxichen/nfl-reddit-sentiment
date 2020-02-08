import numpy as np
import pandas as pd
from tinydb import TinyDB, Query
import datetime
import time

query = Query()
# Data
DATA_PATH = "C:/Users/peter/OneDrive/Documents/THESIS/score.csv"
df = pd.read_csv(DATA_PATH)

SUBMISSION_PATH = "C:/Users/peter/OneDrive/Documents/THESIS/submissions.json"
submission_db = TinyDB(SUBMISSION_PATH)
COMMENT_PATH = "C:/Users/peter/OneDrive/Documents/THESIS/comments.json"
comment_db = TinyDB(COMMENT_PATH)

# return submissions from given time period
def sub_query(subreddit, d1, d2):
    utc_date1 = time.mktime(d1.timetuple())
    utc_date2 = time.mktime(d2.timetuple())
    subs = submission_db.search((query['subreddit'] == subreddit) & (query['created_utc']>utc_date1) & (query['created_utc']<utc_date2))
    article_text = []
    for sub in subs:
        thisdict = {
            "id": sub['id'],
            "title": sub['title'],
            "score": sub['score']
            }
        article_text.append(thisdict)
    return article_text

# queries comments from given time period
def com_query(subreddit, d1, d2):
    utc_date1 = time.mktime(d1.timetuple())
    utc_date2 = time.mktime(d2.timetuple())
    coms = comment_db.search((query['subreddit'] == subreddit) & (query['created_utc']>utc_date1) & (query['created_utc']<utc_date2))
    comment_text = []
    for com in coms:
        thisdict = {
            "id": com['id'],
            "body":com['body'],
            "score":com['score']
            }
        comment_text.append(thisdict)
    return comment_text


NUM_ROWS = df.shape[0]
print(NUM_ROWS)
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['Home_Submissions'] = ""
df['Away_Submissions'] = ""
df['Home_Comments'] = ""
df['Away_Comments'] = ""
for i in range(700, NUM_ROWS):
    home = df['Home'][i]
    away = df['Away'][i]
    print("Processing row ", i)
    #print(home, " @ ", away, df['Datetime'][i])
    df['Home_Submissions'][i] = sub_query(home, df['Datetime'][i] - datetime.timedelta(days=1), df['Datetime'][i])
    df['Home_Comments'][i] = com_query(home, df['Datetime'][i] - datetime.timedelta(days=1), df['Datetime'][i])
    df['Away_Submissions'][i] = sub_query(away, df['Datetime'][i] - datetime.timedelta(days=1), df['Datetime'][i])
    df['Away_Comments'][i] = com_query(away, df['Datetime'][i] - datetime.timedelta(days=1), df['Datetime'][i])
    if len(df['Home_Submissions'][i]) == 0:
        print("ERROR:", home, df['Datetime'][i])
    if len(df['Away_Submissions'][i]) == 0:
        print("ERROR:", away, df['Datetime'][i])
df.to_pickle('temp1')
