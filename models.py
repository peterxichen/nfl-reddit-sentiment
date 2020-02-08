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
from sklearn.decomposition import SparsePCA, PCA
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from afinn import Afinn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import warnings

warnings.filterwarnings("ignore")

file_name = "30-Day Comments"
print(file_name)
df = pd.read_pickle('data30')

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

home_corpus = []
away_corpus = []
home_vader = []
away_vader = []
home_afinn = []
away_afinn = []
temp = []

df['home_wts'] = ''
for i in range(df.shape[0]):
    # remove empty row
    if len(df['home_comments'][i]) == 0 or len(df['away_comments'][i])==0:
        df = df.drop([i], axis=0)
        continue
    
    home = team_dict1[df['team_home'][i]]
    away = team_dict1[df['team_away'][i]]

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
        
    
    print(i)
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


home_counts  = vectorizer.fit_transform(home_corpus).toarray()
home_tfidf = transformer.fit_transform(home_counts).toarray()
away_counts  = vectorizer.fit_transform(away_corpus).toarray()
away_tfidf = transformer.fit_transform(away_counts).toarray()


counts = []
tfidf = []
vader = []
afinn = []
for i in range(home_counts.shape[0]):
    counts.append(np.concatenate((home_counts[i], away_counts[i])))
    tfidf.append(np.concatenate((home_tfidf[i], away_tfidf[i])))
    vader.append(np.concatenate((home_vader[i], away_vader[i])))
    afinn.append([home_afinn[i], away_afinn[i]])
#counts = scaler.fit_transform(np.matrix(counts))
#tfidf = scaler.fit_transform(np.matrix(tfidf))
#vader = scaler.fit_transform(np.matrix(vader))
#afinn = scaler.fit_transform(np.matrix(afinn))
counts = np.matrix(counts)
tfidf = np.matrix(tfidf)
vader = np.matrix(vader)
afinn = np.matrix(afinn)
y = np.array(df['home_wts']).astype(str).astype(int)

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

"""
# DETERMINE # COMPONENTS (COUNTS)
pca = PCA().fit(counts)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axhline(y=0.9, color='r', linestyle='-')
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Explained Variance for Counts Matrix ('+file_name+')')
plt.show()

# DETERMINE # COMPONENTS (TFIDF)
pca = PCA().fit(tfidf)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axhline(y=0.9, color='r', linestyle='-')
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Explained Variance for TFIDF Matrix ('+file_name+')')
plt.show()
"""
for X in [PCA(n_components=50).fit_transform(counts), PCA(n_components=1500).fit_transform(tfidf), vader, afinn]:
    print(X.shape)
    # Gaussian Naive Bayes
    #model = GaussianNB()
    #train_score, test_score, train_var, test_var = perform_kfold(model, X, y)
    #print("GNB (train): ", train_score)
    #print("GNB: ", test_score,"(", test_var,")")
    

    model = RandomForestClassifier(n_estimators=100, max_depth=4)
    train_score, test_score, train_var, test_var = perform_kfold(model, X, y)
    #print("Random Forest (train): ", train_score)
    print("Random Forest: ", test_score,"(", test_var,")")
"""

    for C in [10**-3, 10**-2, 10**-1, 1, 10, 100]:
        # Linear SVC
        model = LinearSVC(C=C)
        train_score, test_score, train_var, test_var = perform_kfold(model, X, y)
        #print("Linear SVC (train) C = ", C, ": ", train_score)
        print("Linear SVC C =", C, ": ", test_score,"(", test_var,")")

    for C in [10**-3, 10**-2, 10**-1, 1, 10, 100]:
        # Logistic Regression
        model = LogisticRegression(C=C)
        train_score, test_score, train_var, test_var = perform_kfold(model, X, y)
        #print("Logistic (train) C = ", C, ": ", train_score)
        print("Logistic C =", C, ": ", test_score,"(", test_var,")")

    for k in [1, 3, 5, 10]:
        model = KNeighborsClassifier(n_neighbors=k)
        train_score, test_score, train_var, test_var = perform_kfold(model, X, y)
        #print("Random Forest (train): ", train_score)
        print("K Nearest Neighbors K =", k, ": ", test_score,"(", test_var,")")

"""
