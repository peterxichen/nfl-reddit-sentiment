import numpy as np
import pandas as pd
import praw
import psaw
from tinydb import TinyDB, Query
import datetime
import time
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import SparsePCA, PCA
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from afinn import Afinn
from keras.layers import Dense, LSTM, Dropout
from keras.layers.convolutional import Conv2D
from keras.wrappers.scikit_learn import KerasClassifier
import warnings

warnings.filterwarnings("ignore")

""" CONFIGURATION """

# PRAW + PSAW configuration
reddit = praw.Reddit(client_id=secrets.CLIENT_ID,
                     client_secret=secrets.CLIENT_SECRET,
                     user_agent=secrets.USERAGENT
                     )
print("Read only:", reddit.read_only)
ps = psaw.PushshiftAPI(reddit)

# directory for data (from sportsline)
DATA_PATH = "XXXX"
df = pd.read_csv(DATA_PATH)

# return top 100 submissions from given time window and subreddit
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

# queries top 100 comments from given time window and subreddit
def com_query(team, d1, d2):
    start_epoch = time.mktime(d1.timetuple())
    end_epoch = time.mktime(d2.timetuple())  
    comments = list(ps.search_comments(after=int(start_epoch),
                                             before=int(end_epoch),
                                             sort='desc',
                                             sort_type='score',
                                             subreddit=str(team),
                                             limit=100))
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


# dictionaries for NFL team names
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


""" DATA COLLECTION """

DELTA = 7 # time window

# scrape relevant text for each game from Reddit
df['schedule_date'] = pd.to_datetime(df['schedule_date'])
df['home_submissions'] = ""
df['away_submissions'] = ""
df['home_comments'] = ""
df['away_comments'] = ""
df['home_wts'] = ""
for i, row in df.iterrows():
    date = df['schedule_date'][i]    
    home = team_dict1[df['team_home'][i]]
    away = team_dict1[df['team_away'][i]]
    year = date.year
    
    # rams city change
    if year < 2016 and df['team_favorite_id'][i] == "LAR":
        df['team_favorite_id'][i] = "RAM"
   
    print("Processing row ", i, ":", home, " @ ", away, date)
    df['home_submissions'][i] = sub_query(home, date - datetime.timedelta(days=DELTA), date)
    df['home_comments'][i] = com_query(home, date - datetime.timedelta(days=DELTA), date)
    df['away_submissions'][i] = sub_query(away, date - datetime.timedelta(days=DELTA), date)
    df['away_comments'][i] = com_query(away, date - datetime.timedelta(days=DELTA), date)

# saves scraped data
df.to_pickle('data7')
df = pd.read_pickle('data7')

# instance
scaler = StandardScaler()
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
vectorizer = CountVectorizer()
analyzer = SentimentIntensityAnalyzer()
transformer = TfidfTransformer(smooth_idf=False)
afinn_analyzer = Afinn()

# preprocess text
def preprocess(text):
    tokens = tokenizer.tokenize(text.lower()) #split into tokens
    tokens = [token for token in tokens if not token.isdigit()] # remove numbers
    tokens = [token for token in tokens if len(token) > 3] # remove single characters
    tokens = [token for token in tokens if not token in stop_words] # remove stop words
    # tokens = [stemmer.stem(tokens) for token in tokens] # "stem" words
    string = ""
    for token in tokens:
        string = " ".join([string, token])
    return string

""" FEATURE ENGINEERING """

home_corpus = []
away_corpus = []
home_vader = []
away_vader = []
home_afinn = []
away_afinn = []
temp = []
df['home_wts'] = '' # 1 for home team wins the spread, 0 for away
df['above_ou'] = '' # 1 for above, 0 for under

# extract features from scraped text
for i in range(df.shape[0]):
    # remove empty rows
    if len(df['home_comments'][i]) == 0 or len(df['away_comments'][i])==0:
        df = df.drop([i], axis=0)
        continue
    
    home = team_dict1[df['team_home'][i]]
    away = team_dict1[df['team_away'][i]]

    # determine WTS outcome
    spread = df['spread_favorite'][i]
    if df['team_favorite_id'][i] != 'PICK':
        favored = team_dict2[df['team_favorite_id'][i]]
        # calculate point spread
        if away == favored:
            spread = -1*spread # away team is favored
    game_spread = df['score_home'][i]-df['score_away'][i]
    if game_spread + spread > 0:
        df['home_wts'][i] = 1 #
    elif game_spread + spread == 0:
        df = df.drop([i], axis=0) # REMOVE PUSH
        continue
    else:
        df['home_wts'][i] = 0

    # determine over-under outcome
    ou = df['over_under_line'][i]
    game_ou = df['score_home'][i]+df['score_away'][i]
    if ou > game_ou:
        df['above_ou'][i] = 1
    elif game_ou == ou:
        df = df.drop([i], axis=0) # REMOVE PUSH
        continue
    else:
        df['above_ou'][i] = 0 
    
    print(i)

    # home team
    string = ""
    for thisdict in df['home_comments'][i]:
        if thisdict["body"] is None:
            continue
        string = string + thisdict["body"] + " "
    string = preprocess(string)
    home_corpus.append(string)
    if len(string) == 0:
        home_vader.append([0,0,0,0])
    else:
        score = analyzer.polarity_scores(string)
        num_words = len(string.split())
        home_vader.append([score['pos'], score['neu'], score['neg'], score['compound']])
    home_afinn.append(afinn_analyzer.score(string))

    # away team
    string = ""
    for thisdict in df['away_comments'][i]:
        if thisdict["body"] is None:
            continue
        string = string + thisdict["body"] + " "
    string = preprocess(string)
    away_corpus.append(string)
    if len(string) == 0:
        away_vader.append([0,0,0,0])
    else:
        score = analyzer.polarity_scores(string)
        num_words = len(string.split())
        away_vader.append([score['pos'], score['neu'], score['neg'], score['compound']])
    away_afinn.append(afinn_analyzer.score(string))

# create bag-of-words + TF-IDF transformations
home_counts  = vectorizer.fit_transform(home_corpus).toarray()
home_tfidf = transformer.fit_transform(home_counts).toarray()
away_counts  = vectorizer.fit_transform(away_corpus).toarray()
away_tfidf = transformer.fit_transform(away_counts).toarray()

# create feature matrices
counts = []
tfidf = []
vader = []
afinn = []
for i in range(home_counts.shape[0]):
    counts.append(np.concatenate((home_counts[i], away_counts[i])))
    tfidf.append(np.concatenate((home_tfidf[i], away_tfidf[i])))
    vader.append(np.concatenate((home_vader[i], away_vader[i])))
    afinn.append([home_afinn[i], away_afinn[i]])
counts = np.matrix(counts)
tfidf = np.matrix(tfidf)
vader = np.matrix(vader)
afinn = np.matrix(afinn)
y = np.array(df['home_wts']).astype(str).astype(int)
#y = np.array(df['above_ou']).astype(str).astype(int)

# perform k-fold validation (k=5) 100 times
def perform_kfold(model, X, y):
    train_score = []
    test_score = []
    for i in range(100):
        kf = KFold(n_splits=5)
        this_train_score = 0
        this_test_score = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            this_train_score = this_train_score + accuracy_score(y_train, model.predict(X_train))
            this_test_score = this_test_score + accuracy_score(y_test, model.predict(X_test))
            #if sum(model.predict(X_test)) == len(model.predict(X_test)):
            #    print("SINGLE CLASSIFICATION")
        train_score.append(this_train_score/5)
        test_score.append(this_test_score/5)
    return np.mean(train_score), np.mean(test_score), np.var(train_score), np.var(test_score)


# determine principal components to use (COUNTS)
pca = PCA().fit(counts)
# plot cumulative sum of explained variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axhline(y=0.9, color='r', linestyle='-')
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Explained Variance for Counts Matrix ('+file_name+')')
plt.show()

# determine principal components to use (TFIDF)
pca = PCA().fit(tfidf)
# plot cumulative sum of explained variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axhline(y=0.9, color='r', linestyle='-')
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Explained Variance for TFIDF Matrix ('+file_name+')')
plt.show()


""" TESTING MODELS """

for X in [PCA(n_components=50).fit_transform(counts), PCA(n_components=1500).fit_transform(tfidf), vader, afinn]:
    print(X.shape)
    # Gaussian Naive Bayes
    model = GaussianNB()
    train_score, test_score, train_var, test_var = perform_kfold(model, X, y)
    #print("GNB (train): ", train_score)
    print("GNB: ", test_score,"(", test_var,")")

    # Random Forest
    for d in [2, 4]:
        model = RandomForestClassifier(n_estimators=100, max_depth=d)
        train_score, test_score, train_var, test_var = perform_kfold(model, X, y)
        #print("Random Forest (train): ", train_score)
        print("Random Forest: ", test_score,"(", test_var,")")

    # Linear SVC
    for C in [10**-3, 10**-2, 10**-1, 1, 10, 100]:
        model = LinearSVC(C=C)
        train_score, test_score, train_var, test_var = perform_kfold(model, X, y)
        #print("Linear SVC (train) C = ", C, ": ", train_score)
        print("Linear SVC C =", C, ": ", test_score,"(", test_var,")")

    # Logistic Regression
    for C in [10**-3, 10**-2, 10**-1, 1, 10, 100]:
        model = LogisticRegression(C=C)
        train_score, test_score, train_var, test_var = perform_kfold(model, X, y)
        #print("Logistic (train) C = ", C, ": ", train_score)
        print("Logistic C =", C, ": ", test_score,"(", test_var,")")

    # K-Nearest Neighbors
    for k in [1, 3, 5, 10]:
        model = KNeighborsClassifier(n_neighbors=k)
        train_score, test_score, train_var, test_var = perform_kfold(model, X, y)
        #print("Random Forest (train): ", train_score)
        print("K Nearest Neighbors K =", k, ": ", test_score,"(", test_var,")")


""" NEURAL NETWORKS """

# start a new dataframe
df = pd.read_csv(DATA_PATH)
df['schedule_date'] = pd.to_datetime(df['schedule_date'])
df['home_wts'] = ""
for delta in range(30):
    df['home_submissions_'+str(delta)] = ""
    df['away_submissions_'+str(delta)] = ""
    df['home_comments_'+str(delta)] = ""
    df['away_comments_'+str(delta)] = ""

# scrape relevant text for each game from Reddit (30 day window)
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
df = pd.read_pickle('dataNN')

""" FEATURE ENGINEERING FOR NEURAL NETWORKS """

corpus = []
vader = []
afinn = []
df['home_wts'] = '' # 1 for home team wins the spread, 0 for away
df['above_ou'] = '' # 1 for above, 0 for under

# extract features from scraped text
for i, row in df.iterrows():
    # remove if empty
    if (((len(df['home_comments_0'][i]) == 0 and len(df['home_comments_1'][i]) == 0 and len(df['home_comments_2'][i]) == 0
        and len(df['home_comments_3'][i]) == 0 and len(df['home_comments_4'][i]) == 0 and len(df['home_comments_5'][i]) == 0
        and len(df['home_comments_6'][i]) == 0 and len(df['home_comments_7'][i]) == 0 and len(df['home_comments_8'][i]) == 0
        and len(df['home_comments_9'][i]) == 0 and len(df['home_comments_10'][i]) == 0 and len(df['home_comments_11'][i]) == 0
        and len(df['home_comments_12'][i]) == 0 and len(df['home_comments_13'][i]) == 0)
        and len(df['home_comments_14'][i]) == 0
        and len(df['home_comments_15'][i]) == 0 and len(df['home_comments_16'][i]) == 0 and len(df['home_comments_17'][i]) == 0
        and len(df['home_comments_18'][i]) == 0 and len(df['home_comments_19'][i]) == 0 and len(df['home_comments_20'][i]) == 0
        and len(df['home_comments_21'][i]) == 0 and len(df['home_comments_22'][i]) == 0 and len(df['home_comments_23'][i]) == 0
        and len(df['home_comments_24'][i]) == 0 and len(df['home_comments_25'][i]) == 0 and len(df['home_comments_26'][i]) == 0
        and len(df['home_comments_27'][i]) == 0 and len(df['home_comments_28'][i]) == 0 and len(df['home_comments_29'][i]) == 0)
        or (len(df['away_comments_0'][i]) == 0 and len(df['away_comments_1'][i]) == 0 and len(df['away_comments_2'][i]) == 0
        and len(df['away_comments_3'][i]) == 0 and len(df['away_comments_4'][i]) == 0 and len(df['away_comments_5'][i]) == 0
        and len(df['away_comments_6'][i]) == 0 and len(df['away_comments_7'][i]) == 0 and len(df['away_comments_8'][i]) == 0
        and len(df['away_comments_9'][i]) == 0 and len(df['away_comments_10'][i]) == 0 and len(df['away_comments_11'][i]) == 0
        and len(df['away_comments_12'][i]) == 0 and len(df['away_comments_13'][i]) == 0
        and len(df['away_comments_14'][i]) == 0
        and len(df['away_comments_15'][i]) == 0 and len(df['away_comments_16'][i]) == 0 and len(df['away_comments_17'][i]) == 0
        and len(df['away_comments_18'][i]) == 0 and len(df['away_comments_19'][i]) == 0 and len(df['away_comments_20'][i]) == 0
        and len(df['away_comments_21'][i]) == 0 and len(df['away_comments_22'][i]) == 0 and len(df['away_comments_23'][i]) == 0
        and len(df['away_comments_24'][i]) == 0 and len(df['away_comments_25'][i]) == 0 and len(df['away_comments_26'][i]) == 0
        and len(df['away_comments_27'][i]) == 0 and len(df['away_comments_28'][i]) == 0 and len(df['away_comments_29'][i]) == 0)):
        df = df.drop([i], axis=0)
        continue

    home = team_dict1[df['team_home'][i]]
    away = team_dict1[df['team_away'][i]]

    # determine WTS outcome
    spread = df['spread_favorite'][i]
    if df['team_favorite_id'][i] != 'PICK':
        favored = team_dict2[df['team_favorite_id'][i]]
        # calculate point spread
        if away == favored:
            spread = -1*spread # away team is favored
    game_spread = df['score_home'][i]-df['score_away'][i]
    if game_spread + spread > 0:
        df['home_wts'][i] = 1 #
    elif game_spread + spread == 0:
        df = df.drop([i], axis=0) # REMOVE PUSH
        continue
    else:
        df['home_wts'][i] = 0

    # determine over-under outcome
    ou = df['over_under_line'][i]
    game_ou = df['score_home'][i]+df['score_away'][i]
    if ou > game_ou:
        df['above_ou'][i] = 1
    elif game_ou == ou:
        df = df.drop([i], axis=0) # REMOVE PUSH
        continue
    else:
        df['above_ou'][i] = 0 
    
    print(i)

    # home team
    for delta in range(30):
        string = ""
        for thisdict in df['home_comments_'+str(delta)][i]:
            if thisdict["body"] is None:
                continue
            string = string + thisdict["body"] + " "
        string = preprocess(string)
        corpus.append(string)

        if len(string) == 0:
            vader.append([0,0,0,0])
        else:
            score = analyzer.polarity_scores(string)
            num_words = len(string.split())
            vader.append([score['pos'], score['neu'], score['neg'], score['compound']])
        afinn.append(afinn_analyzer.score(string))

    # away team
    for delta in range(30):    
        string = ""
        for thisdict in df['away_comments_'+str(delta)][i]:
            if thisdict["body"] is None:
                continue
            string = string + thisdict["body"] + " "
        string = preprocess(string)
        corpus.append(string)
        if len(string) == 0:
            vader.append([0,0,0,0])
        else:
            score = analyzer.polarity_scores(string)
            num_words = len(string.split())
            vader.append([score['pos'], score['neu'], score['neg'], score['compound']])
        afinn.append(afinn_analyzer.score(string))
        
# create bag-of-words + TF-IDF transformations
counts  = vectorizer.fit_transform(corpus)
tfidf = transformer.fit_transform(counts)

# create feature matrices
temp_counts = []
temp_tfidf = []
temp_vader = []
temp_afinn = []
for i in range(int((counts.shape[0])/60)):
    temp_counts.append(np.matrix(counts[i*60:i*60+60]))
    temp_tfidf.append(np.matrix(tfidf[i*60:i*60+60]))
    temp_vader.append(np.matrix(vader[i*60:i*60+60]))
    temp_afinn.append(np.matrix(afinn[i*60:i*60+60]).T)
counts = temp_counts
tfidf = temp_tfidf
vader = temp_vader
afinn = temp_afinn
y = np.array(df['home_wts']).astype(str).astype(int)
y = np.matrix(y).T

# determine principal components to use (COUNTS)
pca = PCA().fit(counts)
# plot cumulative sum of explained variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axhline(y=0.9, color='r', linestyle='-')
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Explained Variance for Counts Matrix ('+file_name+')')
plt.show()

# determine principal components to use (TFIDF)
pca = PCA().fit(tfidf)
# plot cumulative sum of explained variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axhline(y=0.9, color='r', linestyle='-')
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Explained Variance for TFIDF Matrix ('+file_name+')')
plt.show()

# perform PCA (bag-of-words)
pca = TruncatedSVD(20)
pca.fit((sum(counts)/len(counts))[0,0])
X_counts = []
for i in range(len(counts)):
    X_counts.append(np.matrix(pca.transform(counts[i][0,0])))
X_counts = np.array(X_counts)

# perform PCA (TF-IDF)
pca = TruncatedSVD(50)
pca.fit((sum(tfidf)/len(tfidf))[0,0])
X_tfidf = []
for i in range(len(tfidf)):
    X_tfidf.append(np.matrix(pca.transform(tfidf[i][0,0])))
X_tfidf = np.array(X_tfidf)

X_vader = np.array(vader)
X_afinn = np.array(afinn)

""" TESTING NEURAL NETWORKS """
batch_size = 5
for X in [X_counts, X_tfidf, X_vader, X_afinn]:
    print(X[0].shape)
    row, col = X[0].shape
    for neurons in [20, 40]:
        print(neurons, " neurons")

        # LSTM 1-layer Network
        def build_model_1():
            model = Sequential()
            model.add(LSTM(neurons, input_shape=(60,col)))
            model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
            return model
        # evaluate
        model = KerasClassifier(build_fn=build_model_1, epochs=100, batch_size=batch_size, verbose=0)
        test_score, test_var = perform_kfold(model, X, y)
        print("Config 1: ", test_score,"(", test_var,")")

        # LSTM/feedforward 2-layer Network
        def build_model_2():
            model = Sequential()
            model.add(LSTM(neurons, input_shape=(60,col)))
            model.add(Dense(neurons, activation='relu'))
            model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
            return model
        # evaluate
        model = KerasClassifier(build_fn=build_model_2, epochs=100, batch_size=batch_size, verbose=0)
        test_score, test_var = perform_kfold(model, X, y)
        print("Config 2: ", test_score,"(", test_var,")")

        # LSTM 2-layer Network
        def build_model_3():
            model = Sequential()
            model.add(LSTM(neurons, input_shape=(60,col),return_sequences = True))
            model.add(LSTM(neurons))
            model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
            return model
        # evaluate
        model = KerasClassifier(build_fn=build_model_3, epochs=100, batch_size=batch_size, verbose=0)
        test_score, test_var = perform_kfold(model, X, y)
        print("Config 3: ", test_score,"(", test_var,")")

        # LSTM 3-layer Network
        def build_model_4():
            model = Sequential()
            model.add(LSTM(neurons, input_shape=(60,col),return_sequences = True))
            model.add(LSTM(neurons,return_sequences = True))
            model.add(LSTM(neurons))
            model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
            return model
        # evaluate
        model = KerasClassifier(build_fn=build_model_4, epochs=100, batch_size=batch_size, verbose=0)
        test_score, test_var = perform_kfold(model, X, y)
        print("Config 4: ", test_score,"(", test_var,")")
