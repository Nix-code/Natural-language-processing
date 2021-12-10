#import the dataset

import pandas as pd

msg = pd.read_csv('spammsg/spams',sep='\t',names = ['label','message'])

#data cleaning and preprocessing
import re # regex
import nltk # nltk for preprocessing

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# stemming
ps = PorterStemmer()
corpus = []

for i in range(len(msg)):
    texts = re.sub('[^a-zA-Z]',' ',msg['message'][i]) #removing everything expect alphabets
    texts = texts.lower() #lowercase 
    texts = texts.split() #split

    texts = [ps.stem(word) for word in texts if not word in stopwords.words('english')]
    texts = ' '.join(texts)
    corpus.append(texts)


# creating bow
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2500)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(msg['label'])
y = y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 0)

# use naive bayes

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)

ypred = model.predict(X_test)

# accuracy reports
from sklearn.metrics import classification_report,accuracy_score
print(accuracy_score(y_test,ypred))

print(classification_report(y_test, ypred))
