import numpy as np
import pandas as pd
from tinydb import TinyDB, Query
import datetime
import time

query = Query()

SUBMISSION_PATH = "C:/Users/peter/OneDrive/Documents/THESIS/submissions.json"
submission_db = TinyDB(SUBMISSION_PATH)
COMMENT_PATH = "C:/Users/peter/OneDrive/Documents/THESIS/comments.json"
comment_db = TinyDB(COMMENT_PATH)

"""
# DELETING

d1 = datetime.date(2015,1,1)
utc_date1 = time.mktime(d1.timetuple())
d2 = datetime.date(2015,2,1)
utc_date2 = time.mktime(d2.timetuple())
comment_db.remove(query['created_utc']>utc_date1 and query['created_utc']<utc_date2)
submission_db.remove(query['created_utc']>utc_date1 and query['created_utc']<utc_date2)

# ADDING

SUBMISSION_PATH_INSERT = "C:/Users/peter/OneDrive/Documents/THESIS/submissions091011.json"
sub_insert = TinyDB(SUBMISSION_PATH_INSERT).all()

submission_db.insert_multiple(sub_insert)

COMMENT_PATH_INSERT = "C:/Users/peter/OneDrive/Documents/THESIS/comments091011.json"
com_insert = TinyDB(COMMENT_PATH_INSERT).all()

comment_db.insert_multiple(com_insert)
"""
teamlist = ['49ers','AZCardinals','bengals','Browns','buccaneers','buffalobills','Chargers','CHIBears',
            'Colts','cowboys','DenverBroncos','detroitlions','eagles','falcons','GreenBayPackers','Jaguars',
            'KansasCityChiefs','losangelesrams','miamidolphins','minnesotavikings','NYGiants','nyjets',
            'oaklandraiders','panthers','Patriots','ravens','ravens','Redskins','Saints','Seahawks','steelers',
            'StLouisRams','Tennesseetitans','Texans']



year_list = [2012, 2013, 2014, 2015, 2016, 2017, 2018]
end_date_list = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
for year in year_list:
    for month in range(8,12):
        end_date = end_date_list[month]
        print(year, " ", month+1, " ", 1, " to ", year, " ", month+1, " ", end_date)
        d = datetime.date(year,month+1,1)
        utc_date1 = time.mktime(d.timetuple())
        d = datetime.date(year,month+1,end_date)
        utc_date2 = time.mktime(d.timetuple())
        for team in teamlist:
            print(team)
            print(submission_db.count((query['created_utc'] >= utc_date1) & (query['created_utc'] <= utc_date2) & (query['subreddit'] == team)))

