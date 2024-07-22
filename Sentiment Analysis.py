#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
import nltk


# In[5]:


#data in read
df=pd.read_csv(r'E:\NLP_Projects\Reviews.csv')
df


# In[6]:


df.head()


# In[7]:


df['Text'].values[0]


# In[8]:


df.head(500)
df


# In[9]:


# QUICK EDA
ax=df['Score'].value_counts().sort_index().plot(kind='bar', title="Count of Reviews by stars",figsize=(15,5))
ax.set_xlabel('Review Star')
plt.show()


# In[10]:


# Basic NLTK
example=df['Text'][60]
example


# In[11]:


tokens=nltk.word_tokenize(example)
tokens[:10]


# In[12]:


tagged=nltk.pos_tag(tokens)
tagged[:10]


# In[13]:


entities=nltk.chunk.ne_chunk(tagged)
entities.pprint


# In[14]:


# Vader(Valence Aware Dictionary and sentiment reasoners)-Bag of word approach
# Step1 - VADER sentiment scoring
# We will use NLTK's SentimentIntensityAnalyzer to get the neg/neu/pos scores of the text.

# This uses a "bag of words" approach:
# Stop words are removed
# each word is scored and combined to a total score.
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
sia=SentimentIntensityAnalyzer()


# In[15]:


sia.polarity_scores("I am so happy")


# In[16]:


sia.polarity_scores("This is the worst day of my life")


# In[17]:


sia.polarity_scores(example)


# In[18]:


# Run the polarity score for the entire data
res={}
for i,row in tqdm(df.iterrows(),total=len(df)):
    text=row['Text']
    myid=row['Id']
    res[myid]=sia.polarity_scores(text)
res


# In[19]:


vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')


# In[20]:


vaders.head()


# In[21]:


# Plot VADERS Result
ax=sns.barplot(data=vaders, x='Score', y='compound' )
ax.set_title("Compound Score by Amazon Star Review")
plt.show()


# In[22]:


fig,axs=plt.subplots(1, 3, figsize=(15,5))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()


# In[23]:


# Step 3. Roberta Pretrained Model
# Use a model trained of a large corpus of data.
# Transformer model accounts for the words but also the context related to other words.
get_ipython().system('pip install transformers')
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax


# In[24]:


# This is provided by the huggingface and it is a model
get_ipython().system('pip install torch')
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# In[25]:


# Vader result on example
print(example)
sia.polarity_scores(example)


# In[26]:


# Run for Roberta Model
encoded_text=tokenizer(example, return_tensors='pt')
output=model(**encoded_text)
scores=output[0][0].detach().numpy()
scores=softmax(scores)
scores_dic={
    'roberta_neg':scores[0],
    'roberta_neu':scores[1],
    'roberta_pos':scores[2]
}
scores_dic


# In[27]:


def polarity_scores_roberta(example):
    encoded_text=tokenizer(example, return_tensors='pt')
    output=model(**encoded_text)
    scores=output[0][0].detach().numpy()
    scores=softmax(scores)
    scores_dic={
    'roberta_neg':scores[0],
    'roberta_neu':scores[1],
    'roberta_pos':scores[2]
}
    return scores_dic
    


# In[28]:


res={}
for i,row in tqdm(df.iterrows(),total=len(df)):
    try:
        text=row['Text']
        myid=row['Id']
        res[myid]=sia.polarity_scores(text)
        vader_result=sia.polarity_scores(text)
        vader_result_rename={}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"]=value
        
        roberta_result=polarity_scores_roberta(text)
        both={**vader_result, **roberta_result}
        res[myid]=both
    except RuntimeError:
        print(f'Broke for id{myid}')


# In[29]:


both


# In[30]:


results_df=pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df= results_df.merge(df, how='left')


# In[40]:


results_df.head()


# # Compare Scores between models

# In[32]:


results_df.columns


# # Step 3. Combine and compare

# In[34]:


sns.pairplot(data=results_df,
             vars=['neg', 'neu', 'pos',
                   'roberta_neg', 'roberta_neu', 'roberta_pos' ],
             hue='Score',
             palette='tab10')
plt.show()


# # Step 4: Review Examples:
# . Positive 1-Star and Negative 5-Star Reviews
# Lets look at some examples where the model scoring and review score differ the most.

# In[39]:


results_df.query('Score == 1').sort_values('roberta_pos', ascending=False)['Text'].values[0]


# # Negative Sentence

# In[40]:


results_df.query('Score == 1').sort_values('roberta_neg', ascending=False)['Text'].values[0]


# # Extra: The Transformers Pipeline
# . Quick and easy way to run sentiment predictions

# In[43]:


from transformers import pipeline
sent_pipeline=pipeline("sentiment-analysis")


# In[44]:


sent_pipeline("I love sentiment analysis")


# In[ ]




