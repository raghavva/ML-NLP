import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot

sentences = ['the world is a tough place',
             'it is a really hot day today',
             'software developers are in demand',
             'gemini 1.5pro is really fast'
             ]

print(sentences)

#initialize vocabulary size
vocab_size = 500

#convert to one hot encoding
onehot_repr = [one_hot(words, vocab_size) for words in sentences]
print(onehot_repr)

from tensorflow.keras.layers import Embedding,Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
import numpy as np

#we set the max sentence length and add padding to make them all of equal length , on;y then it can be sent to NN for traing
sent_len = 8
embedded_docs = pad_sequences(onehot_repr, padding = 'pre', maxlen = sent_len)
print(embedded_docs)

#now each word should be converted to feature represtion , i.e each word is represented as 'n' features
feature_dim = 8

#model = Sequential()
#input_layer = Input(shape=(sent_len,), dtype = 'float64')
#model.add(Embedding(vocab_size, 8, input_length = sent_len))(input_layer)
#model.compile('adam', 'mse')
#model.summary()

model = Sequential([
    Input(shape=(sent_len,)),  # Define input shape directly
    Embedding(input_dim=vocab_size, output_dim=12),
])
model.compile('adam','mse')
model.summary()

print(model.predict(embedded_docs))

