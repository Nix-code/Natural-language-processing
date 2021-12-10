import nltk

from gensim.models import Word2Vec
from nltk.corpus import stopwords

import re

paragraph = """Let us start with a simple example As a child no one knows anything what’s going on they even have no idea about their gender, not even language and what is right or wrong, what are harmful animals and many more. When you were small child about 3 years old, your parents might have taught you, smoking is injurious to your health, drinking alcohol kills and you should not do drugs. You have been learning since you were child because you have been taught everything about life. Your brain learns these small details. But later, think once, there are many things which you have learned by your own irrespective of the teachers. If you learn how to cook chicken curry then it won’t be difficult for you to cook mutton curry by your own. When you were little, you teacher taught you rule for addition of two numbers. After you practiced many times, you got perfect on addition and you did it. Now You can do addition of any numbers. Not only addition, you probably can do many simplifications and differential equation."""
text = re.sub(r'\[[0-9]*\]',' ',paragraph)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

#preparing the dataset
sentences = nltk.sent_tokenize(text)
sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]

model = Word2Vec(sentences, min_count=1) #discard if the word count is <=1

words = list(model.wv.index_to_key)
vector = model.wv['perfect']

#most similar words

similar = model.wv.most_similar('practiced')


print(similar)







