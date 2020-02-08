from decouple import config
import praw
import psaw
import pdb
import requests
import datetime
from tinydb import TinyDB
from bs4 import BeautifulSoup
import secrets

# PRAW + PSAW
reddit = praw.Reddit(client_id=secrets.CLIENT_ID,
                     client_secret=secrets.CLIENT_SECRET,
                     user_agent=secrets.USERAGENT
                     )
print("Read only:", reddit.read_only)
ps = psaw.PushshiftAPI(reddit)

# Data
SUBMISSION_PATH = "C:/Users/peter/OneDrive/Documents/THESIS/submissions.json"
submission_db = TinyDB(SUBMISSION_PATH)
#submission_db.purge()

COMMENT_PATH = "C:/Users/peter/OneDrive/Documents/THESIS/comments.json"
comment_db = TinyDB(COMMENT_PATH)
#comment_db.purge()

def addData(submissions, comments, submission_db, comment_db):
    count = 0
    for submission in submissions:
        count=count+1
        print(submission.subreddit.display_name, " submission #", count)
        data = {
            'id': submission.id,
            'title': submission.title,
            'created_utc': submission.created_utc,
            'score': submission.score,
            'subreddit': submission.subreddit.display_name
            }
        submission_db.insert(data)

    count = 0
    for comment in comments:
        count=count+1
        print(comment.subreddit.display_name, " comment #:", count)
        comment_data = {
        'id': comment.id,
        'body': comment.body,
        'created_utc': comment.created_utc,
        'score': comment.score,
        'subreddit': comment.subreddit.display_name
        }
        comment_db.insert(comment_data)

            

year_list = [2015, 2016]
end_date_list = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
for year in year_list:
    for i in range(1):
        #leap year
        if (year == 2008 or year == 2012 or year == 2016) and i+1==2:
            end_date = 29
        else:
            end_date = end_date_list[i]        
        for j in range(2):#range(end_date):
            y1 = year
            m1 = i+1
            d1 = j+1 
            if (i+1==12 and j==end_date-1):
                y2 = year+1
                m2 = 1
                d2 = 1
            elif (j==end_date-1):
                y2 = year
                m2= i+2
                d2 = 1
            else:
                y2 = year
                m2= i+1
                d2 = j+2

            print(y1, " ", m1, " ", d1, " to ", y2, " ", m2, " ", d2)

            start_epoch=int(datetime.datetime(y1,m1,d1).timestamp())
            end_epoch=int(datetime.datetime(y2,m2,d2).timestamp())

            # rams to LA in 2016
            teamlist = ['ravens', 'bengals', 'browns', 'steelers',
                         'texans', 'colts', 'jaguars', 'tennesseetitans',
                         'buffalobills', 'miamidolphins', 'patriots', 'nyjets',
                         'denverbroncos', 'kansascitychiefs', 'oaklandraiders', 'chargers',
                         'chibears', 'detroitlions', 'greenbaypackers', 'minnesotavikings',
                         'falcons', 'panthers', 'saints' ,'buccaneers',
                         'cowboys', 'nygiants', 'eagles', 'redskins',
                         'azcardinals', 'stlouisrams', '49ers', 'seahawks']
            if year >= 2016:
                teamlist = ['ravens', 'bengals', 'browns', 'steelers',
                         'texans', 'colts', 'jaguars', 'tennesseetitans',
                         'buffalobills', 'miamidolphins', 'patriots', 'nyjets',
                         'denverbroncos', 'kansascitychiefs', 'oaklandraiders', 'chargers',
                         'chibears', 'detroitlions', 'greenbaypackers', 'minnesotavikings',
                         'falcons', 'panthers', 'saints' ,'buccaneers',
                         'cowboys', 'nygiants', 'eagles', 'redskins',
                         'azcardinals', 'losangelesrams', '49ers', 'seahawks']
            for team in teamlist:
                submissions = list(ps.search_submissions(after=start_epoch,
                                                         before=end_epoch,
                                                         sort='desc',
                                                         sort_type='score',
                                                         subreddit=team,
                                                         limit=10))
                comments = list(ps.search_comments(after=start_epoch,
                                                         before=end_epoch,
                                                         sort='desc',
                                                         sort_type='score',
                                                         subreddit=team,
                                                         limit=10))
                addData(submissions, comments, submission_db, comment_db)
