#!/usr/bin/env python
# coding: utf-8

# # {ML vs My Mom - Housing Model}📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->
# 
# The problem being addressed is the difficulty in accurately predicting home prices in the real estate markets. Homebuyers, sellers, and real estate professionals can benefit from a reliable and accurate model that provides an estimation of home prices based on historical sales and relevant attributes.

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# 
# Can a machine learning model predict home prices more accurately than a human with good real estate knowledge (my mom) of the area (Loveland, OH)?

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->
# 
# The hypothesized answer is that the machine learning model, when trained on a comprehensive dataset of historical sales and relevant attributes, can provide more accurate predictions of home prices than a knowledgeable human, due to its ability to consider a larger number of variables and identify complex patterns in the data.

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->
# 
# I will be scrapping Zillow

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->
# 
# 1. Data Collection: Gather a comprehensive dataset of historical home sales in Loveland, OH, including property attributes (e.g., square footage, number of bedrooms, number of bathrooms, location, and other relevant factors).
# 
# 2. Data Preprocessing & Exploratory Data Analysis: Clean and preprocess the data, handling missing values, outliers, and encoding categorical variables as needed. Examine and analyze the data to summarize the main characteristics and identify patterns, relationships, and anomalies.
# 
# 3. Feature Engineering: Identify and create new features that may have predictive power, based on domain knowledge and exploratory data analysis.
# 
# 4. Model Selection: Test various machine learning algorithms (e.g., linear regression, decision trees, support vector machines, and neural networks) and select the one that provides the best performance, based on evaluation metrics like Mean Absolute Error (MAE) and R-squared.
# 
# 5. Model Training: Train the selected model on the preprocessed dataset, tuning hyperparameters as needed to optimize performance.
# 
# 6. Model Evaluation: Evaluate the model's performance on a test dataset not used during training.
# 
# 7. Interpretation and Validation: Analyze the results, interpret the model's predictions, and validate the model's accuracy by comparing it with the human's predictions on a new set of properties not used during training. If the model demonstrates superior performance, the hypothesis will be supported.

# # Data Collection
# **I'm scrapping Zillow to get all the houses sold in the Loveland, OH (45140 Zip Code) area in the past year (4/7/2022 - 4/7/2023)**

# In[2]:


import json
import time
import pandas as pd
import http.client
import numpy as np
from scipy.stats import skew, kurtosis
from redfin import Redfin

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns


mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
plt.style.use("bmh")
pd.set_option('display.max_rows', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


# **Making a request to Zillow to get the houses with the following fields that I've identified as meaningful**

# In[2]:


statusType = []
soldPrice = []
unformattedPrice = []
address = []
addressStreet = []
addressCity = []
addressState = []
addressZipcode = []
beds = []
baths = []
area = []
isZillowOwned = []
variableData = []
hdpData = []
latitude = []
longitude = []
price = []
dateSold = []
homeType = []
homeStatus = []
zestimate = []
rentZestimate = []
priceForHDP = []
currency = []
country = []
taxAssessedValue = []
lotAreaValue = []
lotAreaUnit = []
brokerName = []

def getResponse(page):
  url = f"https://www.zillow.com/search/GetSearchPageState.htm?searchQueryState=%7B%22pagination%22%3A%7B%22currentPage%22%3A{page}%7D%2C%22usersSearchTerm%22%3A%22Loveland%2C%20OH%22%2C%22mapBounds%22%3A%7B%22west%22%3A-84.37974306982422%2C%22east%22%3A-84.10302493017578%2C%22south%22%3A39.152149232712055%2C%22north%22%3A39.36534736841241%7D%2C%22mapZoom%22%3A12%2C%22regionSelection%22%3A%5B%7B%22regionId%22%3A5710%2C%22regionType%22%3A6%7D%5D%2C%22isMapVisible%22%3Atrue%2C%22filterState%22%3A%7B%22doz%22%3A%7B%22value%22%3A%2212m%22%7D%2C%22isCondo%22%3A%7B%22value%22%3Afalse%7D%2C%22isForSaleForeclosure%22%3A%7B%22value%22%3Afalse%7D%2C%22isApartment%22%3A%7B%22value%22%3Afalse%7D%2C%22isMultiFamily%22%3A%7B%22value%22%3Afalse%7D%2C%22isAllHomes%22%3A%7B%22value%22%3Atrue%7D%2C%22sortSelection%22%3A%7B%22value%22%3A%22globalrelevanceex%22%7D%2C%22isAuction%22%3A%7B%22value%22%3Afalse%7D%2C%22isNewConstruction%22%3A%7B%22value%22%3Afalse%7D%2C%22isRecentlySold%22%3A%7B%22value%22%3Atrue%7D%2C%22isLotLand%22%3A%7B%22value%22%3Afalse%7D%2C%22isTownhouse%22%3A%7B%22value%22%3Afalse%7D%2C%22isMiddleSchool%22%3A%7B%22value%22%3Afalse%7D%2C%22isHighSchool%22%3A%7B%22value%22%3Afalse%7D%2C%22includeUnratedSchools%22%3A%7B%22value%22%3Afalse%7D%2C%22isManufactured%22%3A%7B%22value%22%3Afalse%7D%2C%22isForSaleByOwner%22%3A%7B%22value%22%3Afalse%7D%2C%22isComingSoon%22%3A%7B%22value%22%3Afalse%7D%2C%22isPublicSchool%22%3A%7B%22value%22%3Afalse%7D%2C%22isPrivateSchool%22%3A%7B%22value%22%3Afalse%7D%2C%22isApartmentOrCondo%22%3A%7B%22value%22%3Afalse%7D%2C%22isElementarySchool%22%3A%7B%22value%22%3Afalse%7D%2C%22isForSaleByAgent%22%3A%7B%22value%22%3Afalse%7D%2C%22isCharterSchool%22%3A%7B%22value%22%3Afalse%7D%2C%22greatSchoolsRating%22%3A%7B%22min%22%3A%221%22%7D%7D%2C%22isListVisible%22%3Atrue%7D&wants={{%22cat1%22:[%22listResults%22]}}&requestId=2"

  conn = http.client.HTTPSConnection("www.zillow.com")

  payload={}
  headers = {
    'Cookie': 'x-amz-continuous-deployment-state=AYABeHEN0tqCsWupmgl3%2F5ovG78APgACAAFEAB1kM2Jsa2Q0azB3azlvai5jbG91ZGZyb250Lm5ldAABRwAVRzAxMTE2MjczQzlQQzdZU1JDTVBHAAEAAkNEABpDb29raWUAAACAAAAADG%2F52OaQfNIJt5gr7wAwa8ksg4xl78oKkXFQzgMai7yix49fWvQdzMBOStxRZ0nNBNdkcp0yAUNlFIZtgepnAgAAAAAMAAQAAAAAAAAAAAAAAAAAALGapE1G2GC1faDJGo+NNaX%2F%2F%2F%2F%2FAAAAAQAAAAAAAAAAAAAAAQAAAAxE9IK0h9G6jGBvr5xywc2FYfQD9+myuKPFg1MgQD9+myuKPFg1Mg==; search=6|1682459771554%7Crect%3D39.36534736841241%252C-84.10302493017578%252C39.152149232712055%252C-84.37974306982422%26rid%3D5710%26disp%3Dmap%26mdm%3Dauto%26p%3D1%26z%3D1%26listPriceActive%3D1%26days%3D12m%26type%3Dhouse%26fs%3D0%26fr%3D0%26mmm%3D0%26rs%3D1%26ah%3D0%26singlestory%3D0%26housing-connector%3D0%26abo%3D0%26garage%3D0%26pool%3D0%26ac%3D0%26waterfront%3D0%26finished%3D0%26unfinished%3D0%26cityview%3D0%26mountainview%3D0%26parkview%3D0%26waterview%3D0%26hoadata%3D1%26zillow-owned%3D0%263dhome%3D0%26featuredMultiFamilyBuilding%3D0%26commuteMode%3Ddriving%26commuteTimeOfDay%3Dnow%09%09%09%09%09%09%09%09; zgsession=1|e42fe483-86c7-4941-a44a-035f9ad782e7; zguid=24|%2423d0665d-1a19-4c83-95e5-30ab8f8ec133; AWSALB=7o8QDvkzOEBfwgmsA8eFXbIz1xIWMhxgOu0raIK0MDqHFCnifRfm9ictKJBlnRbacFusneyZJ4AduduIXRQEGjgvn1b/8WcE9Sa3zUAmkNpWqTSZPidQJZyNEvnz; AWSALBCORS=7o8QDvkzOEBfwgmsA8eFXbIz1xIWMhxgOu0raIK0MDqHFCnifRfm9ictKJBlnRbacFusneyZJ4AduduIXRQEGjgvn1b/8WcE9Sa3zUAmkNpWqTSZPidQJZyNEvnz; JSESSIONID=B27F9E1F606458A6EEBC079ECEA1DFB6'
  }

  conn.request("GET", url, payload, headers)
  res = conn.getresponse()
  data = res.read()
  data = data.decode("utf-8")
  json_obj = json.loads(data)
  return json_obj

def getData(json_obj):
  data = [x for x in json_obj["cat1"]["searchResults"]['listResults']]
      
  for d in data:
    statusType.append(d['statusType'] if 'statusType' in d else None)
    soldPrice.append(d['soldPrice'] if 'soldPrice' in d else None)
    unformattedPrice.append(d['unformattedPrice'] if 'unformattedPrice' in d else None)
    address.append(d['address'] if 'address' in d else None)
    addressStreet.append(d['addressStreet'] if 'addressStreet' in d else None)
    addressCity.append(d['addressCity'] if 'addressCity' in d else None)
    addressState.append(d['addressState'] if 'addressState' in d else None)
    addressZipcode.append(d['addressZipcode'] if 'addressZipcode' in d else None)
    beds.append(d['beds'] if 'beds' in d else None)
    baths.append(d['baths'] if 'baths' in d else None)
    area.append(d['area'] if 'area' in d else None)
    isZillowOwned.append(d['isZillowOwned'] if 'isZillowOwned' in d else None)
    variableData.append(d['variableData'] if 'variableData' in d else None)
    hdpData.append(d['hdpData'] if 'hdpData' in d else None)
    latitude.append(d['hdpData']['homeInfo']['latitude'] if 'hdpData' in d and 'homeInfo' in d['hdpData'] and 'latitude' in d['hdpData']['homeInfo'] else None)
    longitude.append(d['hdpData']['homeInfo']['longitude'] if 'hdpData' in d and 'homeInfo' in d['hdpData'] and 'longitude' in d['hdpData']['homeInfo'] else None)
    price.append(d['hdpData']['homeInfo']['price'] if 'hdpData' in d and 'homeInfo' in d['hdpData'] and 'price' in d['hdpData']['homeInfo'] else None)
    dateSold.append(d['hdpData']['homeInfo']['dateSold'] if 'hdpData' in d and 'homeInfo' in d['hdpData'] and 'dateSold' in d['hdpData']['homeInfo'] else None)
    homeType.append(d['hdpData']['homeInfo']['homeType'] if 'hdpData' in d and 'homeInfo' in d['hdpData'] and 'homeType' in d['hdpData']['homeInfo'] else None)
    homeStatus.append(d['hdpData']['homeInfo']['homeStatus'] if 'hdpData' in d and 'homeInfo' in d['hdpData'] and 'homeStatus' in d['hdpData']['homeInfo'] else None)
    zestimate.append(d['hdpData']['homeInfo']['zestimate'] if 'hdpData' in d and 'homeInfo' in d['hdpData'] and 'zestimate' in d['hdpData']['homeInfo'] else None)
    rentZestimate.append(d['hdpData']['homeInfo']['rentZestimate'] if 'hdpData' in d and 'homeInfo' in d['hdpData'] and 'rentZestimate' in d['hdpData']['homeInfo'] else None)  
    priceForHDP.append(d['hdpData']['homeInfo']['priceForHDP'] if 'hdpData' in d and 'homeInfo' in d['hdpData'] and 'priceForHDP' in d['hdpData']['homeInfo'] else None)
    currency.append(d['hdpData']['homeInfo']['currency'] if 'hdpData' in d and 'homeInfo' in d['hdpData'] and 'currency' in d['hdpData']['homeInfo'] else None)
    country.append(d['hdpData']['homeInfo']['country'] if 'hdpData' in d and 'homeInfo' in d['hdpData'] and 'country' in d['hdpData']['homeInfo'] else None)
    taxAssessedValue.append(d['hdpData']['homeInfo']['taxAssessedValue'] if 'hdpData' in d and 'homeInfo' in d['hdpData'] and 'taxAssessedValue' in d['hdpData']['homeInfo'] else None)
    lotAreaValue.append(d['hdpData']['homeInfo']['lotAreaValue'] if 'hdpData' in d and 'homeInfo' in d['hdpData'] and 'lotAreaValue' in d['hdpData']['homeInfo'] else None)
    lotAreaUnit.append(d['hdpData']['homeInfo']['lotAreaUnit'] if 'hdpData' in d and 'homeInfo' in d['hdpData'] and 'lotAreaUnit' in d['hdpData']['homeInfo'] else None)
    brokerName.append(d['brokerName'] if 'brokerName' in d else None)


# **Run initially to get the total pages of data in the response**

# In[3]:


url = f"https://www.zillow.com/search/GetSearchPageState.htm?searchQueryState=%7B%22pagination%22%3A%7B%22currentPage%22%3A1%7D%2C%22usersSearchTerm%22%3A%22Loveland%2C%20OH%22%2C%22mapBounds%22%3A%7B%22west%22%3A-84.37974306982422%2C%22east%22%3A-84.10302493017578%2C%22south%22%3A39.152149232712055%2C%22north%22%3A39.36534736841241%7D%2C%22mapZoom%22%3A12%2C%22regionSelection%22%3A%5B%7B%22regionId%22%3A5710%2C%22regionType%22%3A6%7D%5D%2C%22isMapVisible%22%3Atrue%2C%22filterState%22%3A%7B%22doz%22%3A%7B%22value%22%3A%2212m%22%7D%2C%22isCondo%22%3A%7B%22value%22%3Afalse%7D%2C%22isForSaleForeclosure%22%3A%7B%22value%22%3Afalse%7D%2C%22isApartment%22%3A%7B%22value%22%3Afalse%7D%2C%22isMultiFamily%22%3A%7B%22value%22%3Afalse%7D%2C%22isAllHomes%22%3A%7B%22value%22%3Atrue%7D%2C%22sortSelection%22%3A%7B%22value%22%3A%22globalrelevanceex%22%7D%2C%22isAuction%22%3A%7B%22value%22%3Afalse%7D%2C%22isNewConstruction%22%3A%7B%22value%22%3Afalse%7D%2C%22isRecentlySold%22%3A%7B%22value%22%3Atrue%7D%2C%22isLotLand%22%3A%7B%22value%22%3Afalse%7D%2C%22isTownhouse%22%3A%7B%22value%22%3Afalse%7D%2C%22isMiddleSchool%22%3A%7B%22value%22%3Afalse%7D%2C%22isHighSchool%22%3A%7B%22value%22%3Afalse%7D%2C%22includeUnratedSchools%22%3A%7B%22value%22%3Afalse%7D%2C%22isManufactured%22%3A%7B%22value%22%3Afalse%7D%2C%22isForSaleByOwner%22%3A%7B%22value%22%3Afalse%7D%2C%22isComingSoon%22%3A%7B%22value%22%3Afalse%7D%2C%22isPublicSchool%22%3A%7B%22value%22%3Afalse%7D%2C%22isPrivateSchool%22%3A%7B%22value%22%3Afalse%7D%2C%22isApartmentOrCondo%22%3A%7B%22value%22%3Afalse%7D%2C%22isElementarySchool%22%3A%7B%22value%22%3Afalse%7D%2C%22isForSaleByAgent%22%3A%7B%22value%22%3Afalse%7D%2C%22isCharterSchool%22%3A%7B%22value%22%3Afalse%7D%2C%22greatSchoolsRating%22%3A%7B%22min%22%3A%221%22%7D%7D%2C%22isListVisible%22%3Atrue%7D&wants={{%22cat1%22:[%22listResults%22]}}&requestId=2"

conn = http.client.HTTPSConnection("www.zillow.com")

payload={}
headers = {
'Cookie': 'x-amz-continuous-deployment-state=AYABeHEN0tqCsWupmgl3%2F5ovG78APgACAAFEAB1kM2Jsa2Q0azB3azlvai5jbG91ZGZyb250Lm5ldAABRwAVRzAxMTE2MjczQzlQQzdZU1JDTVBHAAEAAkNEABpDb29raWUAAACAAAAADG%2F52OaQfNIJt5gr7wAwa8ksg4xl78oKkXFQzgMai7yix49fWvQdzMBOStxRZ0nNBNdkcp0yAUNlFIZtgepnAgAAAAAMAAQAAAAAAAAAAAAAAAAAALGapE1G2GC1faDJGo+NNaX%2F%2F%2F%2F%2FAAAAAQAAAAAAAAAAAAAAAQAAAAxE9IK0h9G6jGBvr5xywc2FYfQD9+myuKPFg1MgQD9+myuKPFg1Mg==; search=6|1682459771554%7Crect%3D39.36534736841241%252C-84.10302493017578%252C39.152149232712055%252C-84.37974306982422%26rid%3D5710%26disp%3Dmap%26mdm%3Dauto%26p%3D1%26z%3D1%26listPriceActive%3D1%26days%3D12m%26type%3Dhouse%26fs%3D0%26fr%3D0%26mmm%3D0%26rs%3D1%26ah%3D0%26singlestory%3D0%26housing-connector%3D0%26abo%3D0%26garage%3D0%26pool%3D0%26ac%3D0%26waterfront%3D0%26finished%3D0%26unfinished%3D0%26cityview%3D0%26mountainview%3D0%26parkview%3D0%26waterview%3D0%26hoadata%3D1%26zillow-owned%3D0%263dhome%3D0%26featuredMultiFamilyBuilding%3D0%26commuteMode%3Ddriving%26commuteTimeOfDay%3Dnow%09%09%09%09%09%09%09%09; zgsession=1|e42fe483-86c7-4941-a44a-035f9ad782e7; zguid=24|%2423d0665d-1a19-4c83-95e5-30ab8f8ec133; AWSALB=7o8QDvkzOEBfwgmsA8eFXbIz1xIWMhxgOu0raIK0MDqHFCnifRfm9ictKJBlnRbacFusneyZJ4AduduIXRQEGjgvn1b/8WcE9Sa3zUAmkNpWqTSZPidQJZyNEvnz; AWSALBCORS=7o8QDvkzOEBfwgmsA8eFXbIz1xIWMhxgOu0raIK0MDqHFCnifRfm9ictKJBlnRbacFusneyZJ4AduduIXRQEGjgvn1b/8WcE9Sa3zUAmkNpWqTSZPidQJZyNEvnz; JSESSIONID=B27F9E1F606458A6EEBC079ECEA1DFB6'
}

conn.request("GET", url, payload, headers)
res = conn.getresponse()
data = res.read()
data = data.decode("utf-8")
json_obj = json.loads(data)

total_pages = json_obj.get('cat1', dict()).get('searchList', dict()).get('totalPages')

print(total_pages)


# **Zillow does not like scraping, so we need to sleep for a while between requests**

# In[4]:


for page in range(1, total_pages+1):
    json_obj = getResponse(page)
    getData(json_obj)
    time.sleep(100)


# **Let's put our scraped data into a csv for future use**

# In[5]:


holder = (list(zip(statusType,soldPrice,unformattedPrice,address,addressStreet,addressCity,addressState,addressZipcode,beds,baths,area,isZillowOwned,variableData,hdpData,latitude,longitude,price,dateSold,homeType,homeStatus,zestimate,rentZestimate,priceForHDP,currency,country,taxAssessedValue,lotAreaValue,lotAreaUnit,brokerName)))
columns = ['statusType', 'soldPrice', 'unformattedPrice', 'address', 'addressStreet', 'addressCity', 'addressState', 'addressZipcode', 'beds', 'baths', 'area', 'isZillowOwned', 'variableData', 'hdpData', 'latitude', 'longitude', 'price', 'dateSold', 'homeType', 'homeStatus', 'zestimate', 'rentZestimate', 'priceForHDP', 'currency', 'country', 'taxAssessedValue', 'lotAreaValue', 'lotAreaUnit', 'brokerName']
houses = pd.DataFrame(holder, columns=columns)
houses.to_csv('houses.csv', index=False)


# # Data Preprocessing & Exploratory Data Analysis

# In[63]:


houses = pd.read_csv('houses.csv')
houses = houses[houses['addressZipcode'] == 45140]
houses.info()


# **Dropping some unuseful columns**

# In[64]:


houses.drop(['statusType', 'soldPrice', 'address', 'addressStreet', 'addressCity', 'addressState', 'addressZipcode', 'isZillowOwned', 'variableData', 'hdpData', 'price', 'homeType', 'homeStatus', 'priceForHDP', 'currency', 'country', 'brokerName'], axis=1, inplace=True)
houses.info()


# **Changing column names to make more sense**

# In[65]:


houses.rename(columns={'unformattedPrice': 'price', 'area': 'sqft'}, inplace=True)
houses.head()


# **Some of the area values are in sqft and some are in acres, let's convert all of them to acres**

# In[66]:


def convert_to_acres(row):
    if row['lotAreaUnit'] == 'sqft':
        return row['lotAreaValue'] / 43560
    else:
        return row['lotAreaValue']

houses['lotAreaValue'] = houses.apply(convert_to_acres, axis=1)

houses.drop(['lotAreaUnit'], axis=1, inplace=True)


# **The price is what we're focused on, so let's take a further look at it**

# In[67]:


print(houses['price'].describe())
print('Median: ', houses['price'].median())


# **Wow, a house sold for `$`10.3 million and one sold for only `$`1,000! Something isn't right, I did some digging and it looks like Zillow had some wrong data for some "houses". I checked the `$`10.3 million dollar "house" and it turns out it isn't a house. To negate these issues, I've decided to filter my dataset to prices under `$`10,000,000.**
# 
# I filtered by the zip code 45140 above so this house ins't in our data frame but it is in the raw data, go ahead and check out the price ;) - https://www.zillow.com/homedetails/6091-2nd-St-Miamiville-OH-45147/2062599063_zpid/

# In[68]:


houses = houses[(houses['price'] < 5000000)]
print(houses['price'].describe())
print('Median: ', houses['price'].median())


# **For the houses on the lower end of the price spectrum (like our $1,000 house), I found that either the data was wrong or the houses were complete trash (fixer-uppers if you will). So the cells below were used to help me determine an appropriate cutoff**

# In[69]:


p1 = np.percentile(houses['price'], 1)
p5 = np.percentile(houses['price'], 5)
p10 = np.percentile(houses['price'], 10)

print('1st percentile: ', p1)
print('5th percentile: ', p5)
print('10th percentile: ', p10)

sns.histplot(data=houses, x='price', kde=True)
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Sales Price Distribution')
plt.axvline(x=p1, color='red', linestyle='--', label='1st percentile')
plt.axvline(x=p5, color='green', linestyle='--', label='5th percentile')
plt.axvline(x=p10, color='blue', linestyle='--', label='10th percentile')
plt.legend()
plt.show()


# **1st percentile: $15,000 - Choosing this value would retain 99% of the data, but the value still seems quite low for a home price.**
# **5th percentile: $85,000 - This option retains 95% of the data and seems a little more reasonable as a minimum value for a house.**
# **10th percentile: $149,900 - This choice retains 90% of the data and might be more representative of the lower end of the housing market.**
# 
# **Due to our fixer-upper dillema, I'm gonna go with $149,900 as our minimum value for price.**

# In[70]:


houses = houses[(houses['price'] >= 149900)]


# **Let's check out a histogram of the price again**

# In[71]:


ax = sns.histplot(houses['price'], kde=True)
ax.xaxis.set_major_formatter(plt.FuncFormatter('{:,.0f}'.format))
plt.show()


# **This looks better but it is still skewed to the right, let's look at this further**

# In[72]:


print('Skewness: ', houses['price'].skew())
print('Kurtosis: ', houses['price'].kurt())


# **We can normalize the price using a log transformation. This makes the data more symettryic and follow a normal-like distribution, helping reduce the impact of outliers, handle skewed data, and stabilize the variance across diffferent levels of the variable.**

# In[73]:


houses['log_price'] = np.log(houses['price'])

log_skewness = skew(houses['log_price'])
log_kurtosis = kurtosis(houses['log_price'])

print("Log Skewness: ", log_skewness)
print("Log Kurtosis: ", log_kurtosis)

sns.histplot(data=houses, x='log_price', kde=True)
plt.xlabel('Log Price')
plt.ylabel('Frequency')
plt.title('Log Sales Price Distribution')
plt.show()


# **Let's start digging into some other stuff, let's make sure there are no duplicates.**

# In[74]:


houses.duplicated().sum()


# **Let's look at the correlation between the price and our features.**

# In[75]:


houses.corr()['log_price'].sort_values(ascending=False)


# In[76]:


fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(houses.corr(), annot=True, ax=ax)


# **The Zillow estimates are highly correlated, however these are estimates/predictions themselves. I'd like to create a model that doesn't use another estimate/prediction system. Using them may improve the accuracy of my model, however these are current estimates and not estimates from when the house was sold. I may revisit this, still thinking through it. I'm going to drop the taxAssessedValue as well.**

# In[77]:


houses.drop(['zestimate', 'rentZestimate', 'taxAssessedValue'], axis=1, inplace=True)


# **The dateSold and the latitude/longitude weren't very correlated to the price. I'd like to use latitude and longititude in some way, may revisit later.**

# In[78]:


houses.drop(['dateSold', 'latitude', 'longitude'], axis=1, inplace=True)


# **I'm going to drop price to since we are using the log_price.**

# In[79]:


houses.drop(['price'], axis=1, inplace=True)


# **Lot's of good features here, however some of them have missing values. Let's figure out what to do with these.**

# In[80]:


houses.isna().sum()


# **Option 1: Remove rows with missing values**

# In[81]:


houses_clean = houses.dropna()


# In[82]:


houses_clean.info()


# **Option 2: Fill in missing values**

# In[83]:


houses_filled = houses.copy()
houses_filled['beds'].fillna(houses_filled['beds'].median(), inplace=True)
houses_filled['baths'].fillna(houses_filled['baths'].median(), inplace=True)
houses_filled['sqft'].fillna(houses_filled['sqft'].median(), inplace=True)
houses_filled['lotAreaValue'].fillna(houses_filled['lotAreaValue'].median(), inplace=True)


# In[84]:


houses_filled.info()


# **I'm gonna go with the houses_clean for now, let's see how we are looking.**

# In[87]:


houses_clean.corr()['log_price'].sort_values(ascending=False)


# # Feature Engineering

# **I thought lotAreaValue would be more meaningful, let's **

# In[88]:


# Interaction term
houses_clean['beds_baths_interaction'] = houses_clean['beds'] * houses_clean['baths']

# Polynomial features
houses_clean['sqft_squared'] = houses_clean['sqft'] ** 2
houses_clean['sqft_cubed'] = houses_clean['sqft'] ** 3

# Ratio feature
houses_clean['baths_per_bed'] = houses_clean['baths'] / houses_clean['beds']

houses_clean.corr()['log_price'].sort_values(ascending=False)


# # Model Selection

# # Model Training

# # Model Evaluation

# # Interpretation and Validation

# In[59]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')


# In[15]:


from redfin import Redfin

client = Redfin()

address = '4544 Radnor St, Detroit Michigan'

response = client.search(address)

url = '/OH/Loveland/6872-Clubside-Dr-45140/home/63713622'
initial_info = client.initial_info(url)

property_id = initial_info['payload']['propertyId']
mls_data = client.below_the_fold(property_id)

listing_id = initial_info['payload']['listingId']
avm_details = client.avm_details(property_id, listing_id)


# In[17]:


mls_data


# In[23]:


# Replace 'data' with the actual name of your dictionary
data = mls_data

amenity_info = data['payload']['amenitiesInfo']

amenities = {}

for super_group in amenity_info['superGroups']:
    for amenity_group in super_group['amenityGroups']:
        group_title = amenity_group['groupTitle']
        amenities[group_title] = {}
        for amenity_entry in amenity_group['amenityEntries']:
            if 'amenityName' in amenity_entry:
                amenity_name = amenity_entry['amenityName']
                amenity_values = amenity_entry['amenityValues']
                amenities[group_title][amenity_name] = amenity_values

# Print the extracted amenities
for group_title, group_amenities in amenities.items():
    print(group_title)
    for amenity_name, amenity_values in group_amenities.items():
        print(f"  {amenity_name}: {', '.join(amenity_values)}")
    print()

