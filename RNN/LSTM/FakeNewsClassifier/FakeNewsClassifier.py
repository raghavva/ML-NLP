import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt 

df = pd.read_csv('/Users/raghavvamshithamshetty/Documents/GenAI/NLP/LSTM/train.csv')
print(df.head(2))
print(df.columns)
print(df.isnull().sum())

df = df.dropna()

print(df.shape)

X = df.drop(['label'], axis='columns')
y = df['label']

print(X.shape)
print(y.shape)

import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Input

 #define vocab size
vocab_size = 5000

#OHE 
messages = X.copy()
print(messages['title'][0])

messages = messages.reset_index(drop=True) #just to make sure indxing is working fine

import nltk
import re
from nltk.corpus import stopwords

nltk.download('stopwords')

### Dataset Preprocessing
from nltk.stem.porter import PorterStemmer ##stemming purpose
ps = PorterStemmer()
corpus = []
for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

#print(corpus)

print(corpus[1])

one_hotrepr = [one_hot(words, vocab_size) for words in corpus]
print(one_hotrepr[1])

sent_len = 18
embedded_docs = pad_sequences(one_hotrepr, padding = 'pre', maxlen = sent_len)
print(embedded_docs[2])


##Create the NN 
embedding_features = 35
model = Sequential([
    Input(shape=(sent_len,)),  # Define input shape directly
    Embedding(input_dim=vocab_size, output_dim=35),
    LSTM(100),
    Dense(1,activation = 'relu')
])
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

print(model.summary())


X_final=np.array(embedded_docs)
y_final=np.array(y)
print(X_final.shape,y_final.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)

### Finally Training
model_history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)

y_pred = model.predict(X_test)
y_pred = np.where(y_pred>0.5, 1, 0)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

# summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print(accuracy_score(y_test,y_pred))