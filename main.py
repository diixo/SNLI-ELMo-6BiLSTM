
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from keras.callbacks import Callback
import tensorflow_hub as hub
import tensorflow as tf
import re

from keras import backend as K
#import keras.layers as layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, Embedding, Flatten, Activation, SpatialDropout1D
from keras.layers import Bidirectional, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D, Layer
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import to_categorical
#from keras.engine import Layer
from keras import initializers, regularizers, constraints
from keras.layers import *

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM, GRU, LSTM, Add, Reshape
from keras.layers import MaxPooling1D, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from nltk.tokenize import sent_tokenize, word_tokenize

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['OMP_NUM_THREADS'] = '4'


import re
import math
# set seed
np.random.seed(123)


train = pd.read_csv("datasets/snli-1.0/snli_1.0_train.csv")
test = pd.read_csv("datasets/snli-1.0/snli_1.0_test.csv")
valid = pd.read_csv("datasets/snli-1.0/snli_1.0_dev.csv")

# print("Training loaded on:", train.shape[0], "examples")
# print("Validating loaded on:", test.shape[0], "examples")
# print("Testing loaded on:", valid.shape[0], "examples")
# print(train[:10])

#print(train.isnull().sum())


train = train.dropna(subset = ['sentence2'])
train = train[train["gold_label"] != "-"]
test = test[test["gold_label"] != "-"]
valid = valid[valid["gold_label"] != "-"]

#print(train.nunique())

def get_rnn_data(df):
    x = {
        'sentence1': df["sentence1"],
        #
        'sentence2': df["sentence2"],
        }
    return x

le = LabelEncoder()

X_train = get_rnn_data(train)
Y_train = to_categorical(le.fit_transform(train["gold_label"].values)).astype("int64")

X_valid = get_rnn_data(valid)
Y_valid = to_categorical(le.fit_transform(valid["gold_label"].values)).astype("int64")

X_test = get_rnn_data(test)
Y_test = to_categorical(le.fit_transform(test["gold_label"].values)).astype("int64")


class ElmoEmbeddingLayer(Layer):
    def __init__(self, trainable=False, **kwargs):
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)
        self.dimensions = 1024
        self.trainable = trainable

    def build(self, input_shape):
        # Загружаем ELMo v3 как KerasLayer
        self.elmo = hub.KerasLayer(
            "https://tfhub.dev/google/elmo/3",
            #input_shape=[],  # Вход - один токен или предложение
            dtype=tf.string,
            trainable=self.trainable,
            #name="{}_module".format(self.name)
            signature="default",
            output_key="default",  # pooled embedding
        )
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        # ELMo v3 ожидает shape (batch_size,) или (batch_size, 1)
        x_squeezed = tf.squeeze(tf.cast(x, tf.string), axis=-1)
        return self.elmo(x_squeezed)

    def compute_mask(self, inputs, mask=None):
        tf.print(">>> compute_mask got:", inputs)
        return tf.not_equal(inputs, '--PAD--')

    # def compute_output_shape(self, input_shape):
    #     return (input_shape[0], self.dimensions)


class NonMasking(Layer):   
    def __init__(self, **kwargs):   
        self.supports_masking = True  
        super(NonMasking, self).__init__(**kwargs)   
  
    def build(self, input_shape):   
        input_shape = input_shape   
  
    def compute_mask(self, input, input_mask=None):   
        # do not pass the mask to the next layers   
        return None   
  
    def call(self, x, mask=None):   
        return x   
  
    def get_output_shape_for(self, input_shape):   
        return input_shape

##############################################################################

#### Elmo attempt
def get_model():

    model = Sequential()
    inp1 = Input(shape=(1,), dtype="string", name="sentence1")
    inp2 = Input(shape=(1,), dtype="string", name="sentence2")
    
    def emb_layer(inp, col):
        embd = ElmoEmbeddingLayer()
        x = embd(inp)
        return x

    x = concatenate([
                    emb_layer(inp1, "sen_1"),
                    emb_layer(inp2, "sen_2"),
        ])

    
    x = NonMasking()(x)
    x = Reshape((1, 1024*2), input_shape=(1024*2,))(x)
    x = Bidirectional(LSTM(128, return_sequences=True,recurrent_dropout=0.2))(x)
    x = Bidirectional(LSTM(128, return_sequences=True,recurrent_dropout=0.2))(x)
    x = Bidirectional(LSTM(128, return_sequences=True,recurrent_dropout=0.2))(x)
    x = Bidirectional(LSTM(128, return_sequences=True,recurrent_dropout=0.2))(x)
    x = Bidirectional(LSTM(128, return_sequences=True,recurrent_dropout=0.2))(x)
    x = Bidirectional(LSTM(128, return_sequences=True,recurrent_dropout=0.2))(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])

    outp = Dense(3, activation="softmax", name="final_output")(x)
    
    model = Model(inputs=[inp1,inp2], outputs=outp)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=1e-4),
                  metrics=['accuracy'],
    )
    return model


model = get_model()

model.summary()

#############################################################################################################################

learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_acc', 
    patience=1, 
    verbose=1, 
    factor=0.5,
    min_lr=0.00001)

file_path="checkpoint_SNLI_weights.hdf5"
checkpoint = ModelCheckpoint(
    file_path,
    monitor='val_acc',
    verbose=1,
    save_best_only=True,
    mode='max',
    save_weights_only=True)

early = EarlyStopping(monitor="val_acc", mode="max", patience=1)

model_callbacks = [checkpoint, early, learning_rate_reduction]

model.fit(X_train, Y_train,
    batch_size=128,
    epochs=10,
    verbose=2,
    validation_data=(X_valid, Y_valid),
    callbacks = model_callbacks
    )


model.save_weights("SNLI_weights.hdf5")
model.save("SNLI_model.h5")

test_pred = model.predict(X_test, batch_size=128)

test_acc = (np.argmax(test_pred, axis=1) == np.argmax(Y_test, axis=1)).sum()/Y_test.shape[0] * 100

print(f"Accuracy on test_set: {test_acc}%")
