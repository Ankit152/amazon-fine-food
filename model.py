# importing the libraries
import numpy as np
import pandas as pd
import string
import re
import nltk
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import classification_report

print("All the libraries are imported....")

# loading the dataset
data = pd.read_csv('preprocessed.csv')

print('Dataset loaded....')

def cleaning(text):
    text = str(text)
    text = text.lower()
    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    clean = re.compile('<.*?>')
    text = re.sub(clean,'',text)
    text = pattern.sub('', text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)        
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text) 
    text = re.sub(r"\'ll", " will", text)  
    text = re.sub(r"\'ve", " have", text)  
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"did't", "did not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"have't", "have not", text)

    text = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", text)
    tokens = word_tokenize(text)
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    text = ' '.join(words)
    return text


print('Cleaning of the data taking place....')

data['Text'] = data['Text'].map(cleaning)

data['Score'] = data['Score'].replace({'positive':0,'negative':1})

x = data['Text'].values
y = data['Score'].values


# splitting the data
xtrain, xtest, ytrain, ytest = tts(x,y,test_size=0.2,stratify=y)

# converting it to categorical variable
ytrain = to_categorical(ytrain)
ytest = to_categorical(ytest)

# converting to text to sequences
tokenizer = Tokenizer(25000,lower=True,oov_token='UNK')
tokenizer.fit_on_texts(xtrain)
xtrain = tokenizer.texts_to_sequences(xtrain)
xtest = tokenizer.texts_to_sequences(xtest)

xtrain = pad_sequences(xtrain,maxlen=100,padding='post')
xtest = pad_sequences(xtest,maxlen=100,padding='post')

print("Data preprocessing is over....")

# making the model

print("Making the model....")
model = Sequential()
model.add(Embedding(25000,64,input_length=100))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(64,return_sequences=True)))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(Dense(2,activation="softmax"))
model.compile(optimizer="adam",loss="catagorical_crossentropy",metrics=['accuracy'])
print("The model is defined...")
print(model.summary())

# fitting it into the data
hist=model.fit(xtrain,ytrain,epochs=15,validfation_data=(xtest,ytest))

