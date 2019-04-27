#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import turicreate as tc


# In[2]:


#Read CSV 
unique_beers = pd.read_csv('unique_beers.csv')


# In[7]:


#Create dataframe of required columns then convert to SFrame for turicreate
unique_beers_1 = unique_beers[['userId','beer_beerid','review_overall']]
unique_beers_1 = tc.SFrame(unique_beers_1)
unique_beers_1 = unique_beers_1.dropna()


# In[8]:


#Create SFrame of additional info on beers for model
beer_info = unique_beers[['beer_beerid','beer_style','beer_abv']].drop_duplicates()
beer_info = tc.SFrame(beer_info)


# In[9]:


#Create training and validation set
training_data, validation_data = tc.recommender.util.random_split_by_user(unique_beers_1, 'userId', 'beer_beerid')


# In[10]:


#Create item similarity model
beer_model = tc.item_similarity_recommender.create(training_data, 
                                            user_id="userId", 
                                            item_id="beer_beerid", 
                                            item_data=beer_info,
                                            target="review_overall")


# In[11]:


#Save model
beer_model.save("beer_model")
