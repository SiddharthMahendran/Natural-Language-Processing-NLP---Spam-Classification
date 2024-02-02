import pandas as pd
import seaborn as sns
import nltk
import matplotlib.pyplot as pyplot
import numpy
import re
import sklearn

df = pd.read_csv("C:\\nlp\\spam.csv", encoding='latin-1')
df = df[["v1","v2"]]
df.shape
df.info
df.head()
df.isna().any()
df["v1"].value_counts()
df = df.reset_index(drop=True)
sns.countplot(x = "v1", data = df)
df["v1"] = df["v1"].map({"spam":1, "ham":0})
text = df["v2"]
type(text)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

#Tokenization
from nltk import TweetTokenizer
tk = TweetTokenizer()
text = text.apply(lambda x : tk.tokenize(x)).apply(lambda x:' '.join(x))
text

#Special Character Removal
text = text.str.replace('[^a-zA-Z0-9]+',' ')
text

from nltk import word_tokenize
text = text.apply(lambda x : ' '.join([w for w in word_tokenize(x)if len(x)>=3]))
text

#Stemming
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')
text = text.apply(lambda x : [stemmer.stem(i.lower()) for i in tk.tokenize(x)]).apply(lambda x: ' '.join(x))
text

#Stopwords Removal 
from nltk.corpus import stopwords
stop = stopwords.words('english')
text=text.apply(lambda x: [i for i in tk. tokenize(x) if i not in
stop]).apply(lambda x: ' '.join(x))
text

#Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer() 
data = vec.fit_transform(text)
print(data)
data.shape

y = df["v1"].values
y

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Assuming 'data' and 'y' are defined earlier

x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=0)

knn = KNeighborsClassifier(n_neighbors=5)
nb = MultinomialNB()
svc = SVC()
dec = DecisionTreeClassifier(criterion='entropy')
rfc = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=5, max_features=5)
adb = AdaBoostClassifier(n_estimators=10, learning_rate=1.0)
xgb = XGBClassifier()
lr = LogisticRegression()

list1 = [knn, nb, svc, dec, rfc, adb, xgb, lr]

for classifier in list1:
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print(classifier)
    print(classification_report(y_test, y_pred))
    print('**********************************************************')

import numpy as np 
list1 = [knn, nb, svc, dec, rfc, adb, xgb, lr]
txt = 'Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...'

for classifier in list1:
    pred = classifier.predict(vec.transform(np.array([txt])))
    print(classifier)
    print(pred)
    print('******************************************************')