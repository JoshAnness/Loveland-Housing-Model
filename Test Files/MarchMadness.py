#!/usr/bin/env python
# coding: utf-8

# # March Madness
# 
# ![Banner](./assets/March-Madness-generic-(Blue-Black).jpg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->
# 
# Guessing a perfect bracket is nearly impossible (1 in 120.2 billion odds if you know a little something about basketball). Winners of the Bracket Challenge Game averaged around 49.8% of correctly guessed games.

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# 
# Could machine learning make a perfect bracket (probably not) but could it beat the 49.8% average of correctly guessed games from the top players in the pool?  

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->
# 
# A machine learning model can beat the average winner rate of 49.8 correctly guessed games.

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->
# 
# KenPom - a statistical archive of stats and ratings based on certain common metrics and custom statistical metrics (very popular among coaches, bookmakers, & bettors)
# 
# Basketball Reference - another statistical archive of college basketball games and stats, has most of the common statistics you are going to find in a game
# 
# Barttorvik - statistical archive of common stats and custom stats, like KenPom but has more custom metrics
# 
# Depedending on what stats/metrics I will be able to pull from the sources, I will relate the data based on team/year (regular season). Each data source has regular season stats/metrics for each year for each team. 

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->
# 
# I will first formulate a method to determine how I will be able to include tournament data for each games/round. From there, I can start aggregating the team stats/metrics I want to test out in the data. Depending on the initial data analysis of these features, I will use some of them for the final model.

# In[1]:


from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import re
import cloudscraper
import lxml


# In[7]:


# Base url, and a lambda func to return url for a given year
base_url = 'http://kenpom.com/index.php'
url_year = lambda x: '%s?y=%s' % (base_url, str(x) if x != 2023 else base_url)

years = range(2002, 2024)


# In[8]:


# Create a method that parses a given year and spits out a raw dataframe
def import_raw_year(year):
    """
    Imports raw data from a ken pom year into a dataframe
    """
    scraper = cloudscraper.create_scraper(browser={'browser': 'firefox','platform': 'windows','mobile': False})
    f = scraper.get(url_year(year))
    soup = BeautifulSoup(f.content)
    table_html = soup.find_all('table', {'id': 'ratings-table'})

    thead = table_html[0].find_all('thead')

    table = table_html[0]
    for x in thead:
        table = str(table).replace(str(x), '')

#    table = "<table id='ratings-table'>%s</table>" % table
    df = pd.read_html(table)[0]
    df['year'] = year
    return df


# In[9]:


# Import all the years into a singular dataframe
df = None
for x in years:
    df = pd.concat( (df, import_raw_year(x)), axis=0) \
        if df is not None else import_raw_year(2002)


# In[10]:


# Column rename based off of original website
df.columns = ['Rank', 'Team', 'Conference', 'W-L', 'Pyth', 
             'AdjustO', 'AdjustO Rank', 'AdjustD', 'AdjustD Rank',
             'AdjustT', 'AdjustT Rank', 'Luck', 'Luck Rank', 
             'SOS Pyth', 'SOS Pyth Rank', 'SOS OppO', 'SOS OppO Rank',
             'SOS OppD', 'SOS OppD Rank', 'NCSOS Pyth', 'NCSOS Pyth Rank', 'Year']


# In[11]:


# Lambda that returns true if given string is a number and a valid seed number (1-16)
valid_seed = lambda x: True if str(x).replace(' ', '').isdigit() \
                and int(x) > 0 and int(x) <= 16 else False

# Use lambda to parse out seed/team
df['Seed'] = df['Team'].apply(lambda x: x[-2:].replace(' ', '') \
                              if valid_seed(x[-2:]) else np.nan )

df['Team'] = df['Team'].apply(lambda x: x[:-2] if valid_seed(x[-2:]) else x)


# In[12]:


# Split W-L column into wins and losses
df['Wins'] = df['W-L'].apply(lambda x: int(re.sub('-.*', '', x)) )
df['Losses'] = df['W-L'].apply(lambda x: int(re.sub('.*-', '', x)) )
df.drop('W-L', inplace=True, axis=1)


# In[13]:


# Reorder columns
df=df[[ 'Year', 'Rank', 'Team', 'Conference', 'Wins', 'Losses', 'Seed','Pyth', 
             'AdjustO', 'AdjustO Rank', 'AdjustD', 'AdjustD Rank',
             'AdjustT', 'AdjustT Rank', 'Luck', 'Luck Rank', 
             'SOS Pyth', 'SOS Pyth Rank', 'SOS OppO', 'SOS OppO Rank',
             'SOS OppD', 'SOS OppD Rank', 'NCSOS Pyth', 'NCSOS Pyth Rank']]


# In[20]:


pd.set_option('display.max_columns', None)
df.head()


# ## This KenPom data includes post-season data. Obviously, I do the model will not have post-season stats to use when predicting an outcome. I want to see if I can create a similar dataset with barttorvik.com (I can filter to only the regular season). 

# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->
# 
# https://www.ncaa.com/news/basketball-men/bracketiq/2022-03-10/perfect-ncaa-bracket-absurd-odds-march-madness-dream
# 
# https://kenpom.com/
# 
# https://www.basketball-reference.com/
# 
# https://barttorvik.com/#
# 
# https://github.com/dylorr/kenpom-scraper

# In[2]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python MarchMadness.ipynb')

