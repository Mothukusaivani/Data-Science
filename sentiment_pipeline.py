#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Install Libraries
get_ipython().system('pip install requests beautifulsoup4 pandas nltk transformers --quiet')


# In[3]:


import requests
from bs4 import BeautifulSoup


# In[4]:


import pandas as pd
import re
import nltk


# In[5]:


from transformers import pipeline


# In[6]:


nltk.download('stopwords')
from nltk.corpus import stopwords


# In[7]:


# Step 1: Fetch Data (Hacker News Headlines)
def fetch_news():
    url = "https://timesofindia.indiatimes.com/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = [tag.text.strip() for tag in soup.select('span.w_tle a') if tag.text.strip()]
    return headlines


# In[8]:


# Step 2: Clean Text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text


# In[9]:


# Step 3: Sentiment Analysis
def analyze_sentiment(text_list):
    sentiment_pipeline = pipeline("sentiment-analysis")
    results = sentiment_pipeline(text_list)
    return results


# In[10]:


# Step 4: Main Pipeline
def main():
    headlines = fetch_news()
    cleaned = [clean_text(headline) for headline in headlines]
    sentiments = analyze_sentiment(cleaned)

    df = pd.DataFrame({
        "Original": headlines,
        "Cleaned": cleaned,
        "Sentiment": [s['label'] for s in sentiments],
        "Score": [s['score'] for s in sentiments]
    })

    display(df)

main()


# In[ ]:




