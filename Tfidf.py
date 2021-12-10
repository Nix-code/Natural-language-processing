import nltk

paragraph ="""Early in her academic career, psychologist Wendy Wood noticed a trend: many of her fellow graduate students and professors struggled to get things done in the highly demanding but unstructured academic environment. Intelligence, talent, and motivation didn’t seem to matter—some of those who were struggling to stick to project plans or meet deadlines were among the brightest of the group. Why, she wondered, was it so easy to make the initial decision to change but so hard to persist in the long term? Willpower didn’t seem to be the issue—her colleagues wanted to and were trying to change—so what was? Over the past three decades, Wood has sought the answers to these questions. She recently wrote a book, Good Habits, Bad Habits: The Science of Making Positive Changes that Stick, which details the most important, practical insights from her research. We had the chance to talk about how better understanding how habits form and drive our behavior can help us change—and enjoy—our lives."""


#clearning
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet = WordNetLemmatizer()
sentences =nltk.sent_tokenize(paragraph)
corpus = []

for i in range(len(sentences)):
    tex = re.sub('[^a-zA-Z]',' ',sentences[i]).lower().split()
    tex = [wordnet.lemmatize(word) for word in tex if not word in set(stopwords.words('english'))]
    tex = ' '.join(tex)
    corpus.append(tex)


from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
print(X)
