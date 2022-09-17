#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Import required Libraries


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nlp
import random


# In[4]:


## Import emotion dataset


# In[5]:


print('Using TensorFlow version', tf.__version__)


# In[6]:


dataset=nlp.load_dataset('emotion')


# In[7]:


dataset


# In[8]:


## Divide data into Train, Validation and Test


# In[9]:


train=dataset['train']
val=dataset['validation']
test=dataset['test']


# In[10]:


## Get the tweet data and respective labels


# In[11]:


def get_tweet_data(dataset):
    tweets=[x['text'] for x in dataset]
    labels=[x['label'] for x in dataset]
    return tweets, labels
    


# In[12]:


tweets,labels=get_tweet_data(train)


# In[13]:


## Tokenizer


# In[14]:


from tensorflow.keras.preprocessing.text import Tokenizer


# In[15]:


tokenizer=Tokenizer(num_words=10000, oov_token='<UNK>')
tokenizer.fit_on_texts(tweets)


# In[16]:


lengths=[len(t.split(' '))for t in tweets]


# In[18]:


## Histogram to check the lengths of the tweets


# In[19]:


plt.hist(lengths,bins=len(set(lengths)))
plt.show()


# In[20]:


maxlen=50


# In[21]:


## Padding and Truncating the data to have a same saize input


# In[22]:


from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[23]:


def get_sequences(tokenizer,tweets):
    sequences=tokenizer.texts_to_sequences(tweets)
    padded=pad_sequences(sequences,truncating='post',padding='post',maxlen=maxlen)
    return padded


# In[26]:


padded_train_seq=get_sequences(tokenizer,tweets)


# In[ ]:


## Prepare Labels


# In[27]:


classes = set (labels)


# In[29]:


## Display all unique classes (emotions)


# In[30]:


classes


# In[31]:


class_to_index=dict((c,i)for i,c in enumerate(classes))


# In[32]:


class_to_index


# In[33]:


index_to_class=dict((v,k)for k,v in class_to_index.items())


# In[34]:


index_to_class


# In[35]:


name_to_ids=lambda labels:np.array([class_to_index.get(x)for x in labels])


# In[36]:


train_labels=name_to_ids(labels)


# In[37]:


## Create NN model


# In[38]:


model=tf.keras.models.Sequential([
    tf.keras.layers.Embedding(10000,16,input_length=maxlen),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
    tf.keras.layers.Dense(6,activation='softmax')
    
])
model.compile(
loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# In[39]:


model.summary()


# In[41]:


## Train the model 


# In[42]:


val_tweets,val_labels=get_tweet_data(val)
val_seq=get_sequences(tokenizer,val_tweets)
val_labels=name_to_ids(val_labels)


# In[43]:


h=model.fit(
padded_train_seq, train_labels,
    validation_data=(val_seq,val_labels),
    epochs=20,
    callbacks=[
        
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=2)
    ]
)


# In[ ]:


## Show Training History


# In[44]:


def show_history(h):
    epochs_trained = len(h.history['loss'])
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(0, epochs_trained), h.history.get('accuracy'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get('val_accuracy'), label='Validation')
    plt.ylim([0., 1.])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(0, epochs_trained), h.history.get('loss'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get('val_loss'), label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    
def show_confusion_matrix(y_true, y_pred, classes):
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    plt.figure(figsize=(8, 8))
    sp = plt.subplot(1, 1, 1)
    ctx = sp.matshow(cm)
    plt.xticks(list(range(0, 6)), labels=classes)
    plt.yticks(list(range(0, 6)), labels=classes)
    plt.colorbar(ctx)
    plt.show()


# In[45]:


show_history(h)


# In[46]:


## Predict the emotion


# In[47]:


test_tweets,test_labels=get_tweet_data(test)
test_seq=get_sequences(tokenizer,test_tweets)
test_labels=name_to_ids(test_labels)


# In[48]:


i=random.randint(0,len(test_labels)-1)
print ('Sentence:', test_tweets[i])
print('Emotion:', index_to_class[test_labels[i]])


# In[49]:


p=model.predict(np.expand_dims(test_seq[i],axis=0))[0]
pred_class=index_to_class[np.argmax(p).astype('uint8')]

print('predicted Emotion:',pred_class)


# In[ ]:




