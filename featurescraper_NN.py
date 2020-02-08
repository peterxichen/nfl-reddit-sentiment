import numpy as np
import pandas as pd
import praw
import psaw
from tinydb import TinyDB, Query
import datetime
import time

query = Query()

# PRAW + PSAW
reddit = praw.Reddit(client_id=secrets.CLIENT_ID,
                     client_secret=secrets.CLIENT_SECRET,
                     user_agent=secrets.USERAGENT
                     )
print("Read only:", reddit.read_only)
ps = psaw.PushshiftAPI(reddit)

# Data
DATA_PATH = "C:/Users/peter/OneDrive/Documents/THESIS/score.csv"
df = pd.read_csv(DATA_PATH)


# return submissions from given time period
def sub_query(team, d1, d2):
    start_epoch = time.mktime(d1.timetuple())
    end_epoch = time.mktime(d2.timetuple())
    submissions = list(ps.search_submissions(after=int(start_epoch),
                                             before=int(end_epoch),
                                             sort='desc',
                                             sort_type='score',
                                             subreddit=str(team),
                                             limit=100))
    if len(submissions) < 1:
        print("NO SUBMISSIONS COLLECTED", team)
    sub_dict = []
    for sub in submissions:
        thisdict = {
            "id": sub.id,
            "title": sub.title,
            "score": sub.score
            }
        sub_dict.append(thisdict)
    return sub_dict

# queries comments from given time period
def com_query(team, d1, d2):
    start_epoch = time.mktime(d1.timetuple())
    end_epoch = time.mktime(d2.timetuple())  
    comments = list(ps.search_comments(after=int(start_epoch),
                                             before=int(end_epoch),
                                             sort='desc',
                                             sort_type='score',
                                             subreddit=str(team),
                                             limit=10))
    if len(comments) < 1:
          print("NO COMMENTS COLLECTED", team)
    comment_dict= []
    for com in comments:
        thisdict = {
            "id": com.id,
            "body":com.body,
            "score":com.score
            }
        comment_dict.append(thisdict)
    return comment_dict



# DICTIONARIES
team_dict1 = {
    "San Francisco 49ers": "49ers",
    "Arizona Cardinals": "AZCardinals",
    "Cincinnati Bengals": "bengals",
    "Cleveland Browns": "Browns",
    "Tampa Bay Buccaneers": "buccaneers",
    "Buffalo Bills": "buffalobills",
    "Los Angeles Chargers": "Chargers",
    "San Diego Chargers": "Chargers",
    "Chicago Bears": "CHIBears",
    "Indianapolis Colts": "Colts",
    "Dallas Cowboys": "cowboys",
    "Denver Broncos": "DenverBroncos",
    "Detroit Lions": "detroitlions",
    "Philadelphia Eagles": "eagles",
    "Atlanta Falcons": "falcons",
    "Green Bay Packers": "GreenBayPackers",
    "Jacksonville Jaguars": "Jaguars",
    "Kansas City Chiefs": "KansasCityChiefs",
    "Los Angeles Rams": "losangelesrams",
    "Miami Dolphins": "miamidolphins",
    "Minnesota Vikings": "minnesotavikings",
    "New York Giants": "NYGiants",
    "New York Jets": "nyjets",
    "Oakland Raiders": "oaklandraiders",
    "Carolina Panthers": "panthers",
    "New England Patriots": "Patriots",
    "Baltimore Ravens": "ravens",
    "Washington Redskins": "Redskins",
    "New Orleans Saints": "Saints",
    "Seattle Seahawks": "Seahawks",
    "Pittsburgh Steelers": "steelers",
    "St. Louis Rams": "StLouisRams",
    "Tennessee Titans": "Tennesseetitans",
    "Houston Texans": "Texans"
	}

team_dict2 = {
    "SF": "49ers",
    "ARI": "AZCardinals",
    "CIN": "bengals",
    "CLE": "Browns",
    "TB": "buccaneers",
    "BUF": "buffalobills",
    "LAC": "Chargers",
    "CHI": "CHIBears",
    "IND": "Colts",
    "DAL": "cowboys",
    "DEN": "DenverBroncos",
    "DET": "detroitlions",
    "PHI": "eagles",
    "ATL": "falcons",
    "GB": "GreenBayPackers",
    "JAX": "Jaguars",
    "KC": "KansasCityChiefs",
    "LAR": "losangelesrams",
    "MIA": "miamidolphins",
    "MIN": "minnesotavikings",
    "NYG": "NYGiants",
    "NYJ": "nyjets",
    "OAK": "oaklandraiders",
    "CAR": "panthers",
    "NE": "Patriots",
    "BAL": "ravens",
    "WAS": "Redskins",
    "NO": "Saints",
    "SEA": "Seahawks",
    "PIT": "steelers",
    "RAM": "StLouisRams",
    "TEN": "Tennesseetitans",
    "HOU": "Texans"
    }



print(df.shape[0])
df['schedule_date'] = pd.to_datetime(df['schedule_date'])
df['home_wts'] = ""
for delta in range(30):
    df['home_submissions_'+str(delta)] = ""
    df['away_submissions_'+str(delta)] = ""
    df['home_comments_'+str(delta)] = ""
    df['away_comments_'+str(delta)] = ""

for i, row in df.iterrows():
    date = df['schedule_date'][i]

    home = team_dict1[df['team_home'][i]]
    away = team_dict1[df['team_away'][i]]
    year = date.year
    
    # rams city change
    if year < 2016 and df['team_favorite_id'][i] == "LAR":
        df['team_favorite_id'][i] = "RAM"
   
    print("Processing row ", i, ":", home, " @ ", away, date)

    for delta in range(30):
        print(delta)
        df['home_submissions_'+str(delta)][i] = sub_query(home, date - datetime.timedelta(days=delta+1), date- datetime.timedelta(days=delta))
        df['home_comments_'+str(delta)][i] = com_query(home, date - datetime.timedelta(days=delta+1), date- datetime.timedelta(days=delta))
        df['away_submissions_'+str(delta)][i] = sub_query(away, date - datetime.timedelta(days=delta+1), date- datetime.timedelta(days=delta))
        df['away_comments_'+str(delta)][i] = com_query(away, date - datetime.timedelta(days=delta+1), date- datetime.timedelta(days=delta))

df.to_pickle('dataNN')
