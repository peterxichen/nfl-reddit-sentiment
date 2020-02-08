import numpy as np
import pandas as pd
from tinydb import TinyDB, Query
import datetime
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
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from afinn import Afinn
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.convolutional import Conv2D
from keras.wrappers.scikit_learn import KerasClassifier

df = pd.read_pickle('dataNN')

scaler = StandardScaler()
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

vectorizer = CountVectorizer()

analyzer = SentimentIntensityAnalyzer()
transformer = TfidfTransformer(smooth_idf=False)
afinn_analyzer = Afinn()

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

corpus = []
vader = []
afinn = []
df['above_ou'] = '' # 1 for above, 0 for under

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

    for delta in range(30):
        
        string = ""
        for thisdict in df['home_comments_'+str(delta)][i]:
            if thisdict["body"] is None:
                continue
            string = string + thisdict["body"] + " "
        string = preprocess(string)
        corpus.append(string)

    for delta in range(30):    
        string = ""
        for thisdict in df['away_comments_'+str(delta)][i]:
            if thisdict["body"] is None:
                continue
            string = string + thisdict["body"] + " "
        string = preprocess(string)
        corpus.append(string)


counts  = vectorizer.fit_transform(corpus)
tfidf = transformer.fit_transform(counts)

# MAKE CHANGES HERE (EVERY 28)
temp_counts = []
temp_tfidf = []
temp_vader = []
temp_afinn = []
for i in range(int((counts.shape[0])/60)):
    temp_counts.append(np.matrix(counts[i*60:i*60+60]))
    temp_tfidf.append(np.matrix(tfidf[i*60:i*60+60]))
counts = temp_counts
tfidf = temp_tfidf

y = np.array(df['above_ou']).astype(str).astype(int)
y = np.matrix(y).T

# DETERMINE # COMPONENTS (COUNTS)
pca = PCA().fit((sum(counts)/len(counts))[0,0].toarray())
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axhline(y=0.9, color='r', linestyle='-')
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Counts Explained Variance')
plt.show()

# DETERMINE # COMPONENTS (TFIDF)
pca = PCA().fit((sum(tfidf)/len(tfidf))[0,0].toarray())
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axhline(y=0.9, color='r', linestyle='-')
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('TFIDF Explained Variance')
plt.show()


def perform_kfold(model, X, y):
    train_score = []
    test_score = []
    
    for i in range(1):
        kf = KFold(n_splits=5)
        this_train_score = 0
        this_test_score = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            #train_score.append(accuracy_score(y_test, model.predict(X_train)))
            this_test_score = this_test_score + accuracy_score(y_test, model.predict(X_test))
            #if sum(model.predict(X_test)) == len(model.predict(X_test)):
            #    print("SINGLE CLASSIFICATION")
        test_score.append(this_test_score/5)
    return np.mean(test_score), np.var(test_score)
    



pca = TruncatedSVD(20)
pca.fit((sum(counts)/len(counts))[0,0])

X_counts = []
# perform PCA
for i in range(len(counts)):
    X_counts.append(np.matrix(pca.transform(counts[i][0,0])))
X_counts = np.array(X_counts)


pca = TruncatedSVD(50)
pca.fit((sum(tfidf)/len(tfidf))[0,0])

X_tfidf = []
# perform PCA
for i in range(len(tfidf)):
    X_tfidf.append(np.matrix(pca.transform(tfidf[i][0,0])))
X_tfidf = np.array(X_tfidf)

X_vader = np.array(vader)
X_afinn = np.array(afinn)


batch_size = 5
for X in [X_counts, X_tfidf, X_vader, X_afinn]:
    print(X[0].shape)
    row, col = X[0].shape
    for neurons in [20,30,40]:
        print(neurons, " neurons")

        # LSTM Network
        def build_model_1():
            model = Sequential()
            model.add(LSTM(neurons, input_shape=(60,col)))
            model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
            return model
        # evaluate
        model = KerasClassifier(build_fn=build_model_1, epochs=100, batch_size=batch_size, verbose=0)
        test_score, test_var = perform_kfold(model, X, y)
        print("`1-layer RNN", test_score,"(", test_var,")")

        # LSTM 2-layer Network
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
        print("2-layer RNN", test_score,"(", test_var,")")

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
        print("2-layer RNN", test_score,"(", test_var,")")

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
        print("2-layer RNN", test_score,"(", test_var,")")

"""
`1-layer RNN 0.5197122191443522 ( 0.0 )
2-layer RNN 0.5280378578024008 ( 0.0 )
2-layer RNN 0.516919052016005 ( 0.0 )
2-layer RNN 0.5230532471529702 ( 0.0 )
30  neurons
`1-layer RNN 0.5113865804863035 ( 0.0 )
2-layer RNN 0.5108479532163742 ( 0.0 )
2-layer RNN 0.5113911972914743 ( 0.0 )
2-layer RNN 0.5197106802092951 ( 0.0 )
40  neurons
`1-layer RNN 0.516942136041859 ( 0.0 )
2-layer RNN 0.5263881194213604 ( 0.0 )
2-layer RNN 0.5147245306248076 ( 0.0 )
"""
