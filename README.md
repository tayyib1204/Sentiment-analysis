# In[1]:
import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split  # function for splitting data to train and test set


import nltk


# In[3]:


from nltk.corpus import stopwords

from nltk.classify import SklearnClassifier


# In[4]:


pip install wordcloud


# In[5]:


from wordcloud import WordCloud,STOPWORDS

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# In[6]:


from subprocess import check_output


# In[7]:


data = pd.read_csv(r'C:\Users\Rafi\Downloads\26th\26th\6. NLP PROJECTS\NLP TASKS\TASK - 56\Sentiment.csv')

data = data[['text', 'sentiment']]

data.head()


# In[8]:


# Splitting the dataset into train and test set

train, test = train_test_split(data, test_size=0.1)


# In[9]:


# Removing the natural sentiments
train = train[train.sentiment != 'Neutral']


# As a next step I separated the Positive and Negative tweets of the training set in order to easily visualize their contained words. After that I cleaned the text from hashtags, mentions and links. Now they were ready for a WordCloud visualization which shows only the most emphatic words of the Positive and Negative tweets.

# In[10]:


train_pos = train[ train['sentiment'] == 'Positive']

train_pos = train_pos['text']

train_neg = train[ train['sentiment'] == 'Negative']

train_neg = train_neg['text']


def wordcloud_draw(data, color = 'black'):
    
    words = ' '.join(data)
    
    cleaned_word = " ".join([word for word in words.split()
                            
                            if 'http' not in word
                                
                                and not word.startswith('@')
                                
                                and not word.startswith('#')
                                
                                and word != 'RT'
                            
                            ])
    
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      
                      background_color=color,
                      
                      width=2500,
                      
                      height=2000
                     
                     ).generate(cleaned_word)
    
    plt.figure(1,figsize=(13, 13))
    
    plt.imshow(wordcloud)
    
    plt.axis('off')
    
    plt.show()
    
print("Positive words")

wordcloud_draw(train_pos,'white')

print("Negative words")

wordcloud_draw(train_neg)

![sentiment analysis 1](https://github.com/tayyib1204/Sentiment-analysis/assets/132560640/a1867f77-32f4-4099-8baf-a721e883872b)



![sentiment analysis 2](https://github.com/tayyib1204/Sentiment-analysis/assets/132560640/bbad70f9-2fdc-4f27-9f82-76c044e267fd)

# Interesting to notice the following words and expressions in the positive word set: truth, strong, legitimate, together, love, job
# 
# In my interpretation, people tend to believe that their ideal candidate is truthful, legitimate, above good and bad.
# 
# At the same time, negative tweets contains words like: influence, news, elevator music, disappointing, softball, makeup, cherry picking, trying
# 
# In my understanding people missed the decisively acting and considered the scolded candidates too soft and cherry picking.

# After the vizualization, I removed the hashtags, mentions, links and stopwords from the training set.
# 
# Stop Word: Stop Words are words which do not contain important significance to be used in Search Queries. Usually these words are filtered out from search queries because they return vast amount of unnecessary information. ( the, for, this etc. )

# In[11]:


from nltk.corpus import stopwords


# In[12]:


nltk.download('stopwords')


# In[13]:


from nltk.corpus import stopwords

nltk.download('stopwords')

tweets = []

stopwords_set = set(stopwords.words("english"))

for index, row in train.iterrows():
    
    words_filtered = [e.lower() for e in row['text'].split() if len(e) >= 3]
    
    words_cleaned = [word for word in words_filtered
        
        if 'http' not in word
        
        and not word.startswith('@')
        
        and not word.startswith('#')
        
        and word != 'RT']
    
    words_without_stopwords = [word for word in words_cleaned if word not in stopwords_set]
    
    tweets.append((words_without_stopwords, row.sentiment))

test_pos = test[ test['sentiment'] == 'Positive']

test_pos = test_pos['text']

test_neg = test[ test['sentiment'] == 'Negative']

test_neg = test_neg['text']


# As a next step I extracted the so called features with nltk lib, first by measuring a frequent distribution and by selecting the resulting keys.

# In[14]:


# Extracting word features

def get_words_in_tweets(tweets):
    
    all = []
    
    for (words, sentiment) in tweets:
        
        all.extend(words)
    
    return all

def get_word_features(wordlist):
    
    wordlist = nltk.FreqDist(wordlist)
    
    features = wordlist.keys()
    
    return features

w_features = get_word_features(get_words_in_tweets(tweets))

def extract_features(document):
    
    document_words = set(document)
    
    features = {}
    
    for word in w_features:
        
        features['contains(%s)' % word] = (word in document_words)
    
    return features


# Hereby I plotted the most frequently distributed words. The most words are centered around debate nights.

# In[15]:


wordcloud_draw(w_features)


![sentiment analysis 3](https://github.com/tayyib1204/Sentiment-analysis/assets/132560640/1b01a43a-1915-4787-9c4a-02d9a1fc4d5a)

# Using the nltk NaiveBayes Classifier I classified the extracted tweet word features.

# Finally, with not-so-intelligent metrics, I tried to measure how the classifier algorithm scored.

# In[23]:


import nltk

from nltk.classify import NaiveBayesClassifier

# Define the extract_features function

def extract_features(words):
    
    # Implement feature extraction logic here
    
    features = {}  # Dictionary to store the features
    
    # Add features based on the words
    
    for word in words:
        
        features[word] = True  # Example: Use a simple bag-of-words approach
    
    return features

# Assuming you have training data for the classifier: train_data

# Train the classifier

training_set = [(extract_features(obj.split()), 'Negative') for obj in train_neg] + [(extract_features(obj.split()), 'Positive') for obj in train_pos]

classifier = NaiveBayesClassifier.train(training_set)

# Use the classifier to classify test data

neg_cnt = 0

pos_cnt = 0

for obj in test_neg:
    
    res = classifier.classify(extract_features(obj.split()))
    
    if res == 'Negative':
        
        neg_cnt += 1

for obj in test_pos:
    res = classifier.classify(extract_features(obj.split()))
    if res == 'Positive':
        pos_cnt += 1

# Print the results
print('[Negative]: %s/%s' % (len(test_neg), neg_cnt))

print('[Positive]: %s/%s' % (len(test_pos), pos_cnt))


# Epilog
# In this project I was curious how well nltk and the NaiveBayes Machine Learning algorithm performs for Sentiment Analysis. In my experience, it works rather well for negative comments. The problems arise when the tweets are ironic, sarcastic has reference or own difficult context.
# 
# Consider the following tweet: "Muhaha, how sad that the Liberals couldn't destroy Trump. Marching forward." As you may already thought, the words sad and destroy highly influences the evaluation, although this tweet should be positive when observing its meaning and context.
