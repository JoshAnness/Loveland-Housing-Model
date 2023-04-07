#!/usr/bin/env python
# coding: utf-8

# # {ML vs My Mom - Housing Model}📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->
# 
# People are always either trying to sell a house or buy a house, can a machine learning model accurately predict the price? This predicted price is not only useful to potenital buyers/sellers but can be useful to those in the real estate business like real estate agents and property developers.

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# 
# Can my model more accurately predict the price of houses than a human with good knowledge of the area (my mom)?

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->
# 
# I will take a random sample of houses for my model to predict and my mom to predict (she will be able to use all available info) and compare to see who is more accurate. I think my mom will probably be more accurate. 

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->
# 
# Scraping Zillow

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->
# 
# I am planning on creating a regression model based on the Zillow dataset and features I am able to scrape.

# In[114]:


import json
import time
import pandas as pd
import http.client

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


# # Data Scraping
# My Zillow query only looks at houses sold in the Loveland, OH area in the past year (3/26/2022-2/36/2023) 

# In[57]:


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


# In[60]:


# Run initially to get the total pages
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


# In[62]:


for page in range(1, total_pages+1):
    json_obj = getResponse(page)
    print(json_obj)
    getData(json_obj)
    time.sleep(200)


# In[2]:


holder = (list(zip(statusType,soldPrice,unformattedPrice,address,addressStreet,addressCity,addressState,addressZipcode,beds,baths,area,isZillowOwned,variableData,hdpData,latitude,longitude,price,dateSold,homeType,homeStatus,zestimate,rentZestimate,priceForHDP,currency,country,taxAssessedValue,lotAreaValue,lotAreaUnit,brokerName)))
columns = ['statusType', 'soldPrice', 'unformattedPrice', 'address', 'addressStreet', 'addressCity', 'addressState', 'addressZipcode', 'beds', 'baths', 'area', 'isZillowOwned', 'variableData', 'hdpData', 'latitude', 'longitude', 'price', 'dateSold', 'homeType', 'homeStatus', 'zestimate', 'rentZestimate', 'priceForHDP', 'currency', 'country', 'taxAssessedValue', 'lotAreaValue', 'lotAreaUnit', 'brokerName']
houses = pd.DataFrame(holder, columns=columns)
houses.to_csv('houses.csv', index=False)


# # Exploratory Data Analysis/Data Cleaning and Transformations

# In[91]:


houses = pd.read_csv('houses.csv')
houses.head(2)


# In[92]:


houses.info()


# In[93]:


# Dropping some unuseful columns
houses.drop(['statusType', 'soldPrice', 'address', 'addressStreet', 'addressCity', 'addressState', 'addressZipcode', 'isZillowOwned', 'variableData', 'hdpData', 'price', 'homeType', 'homeStatus', 'priceForHDP', 'currency', 'country'], axis=1, inplace=True)


# In[94]:


# Taking a look at the data again
houses.info()


# In[95]:


houses.drop(['brokerName', 'dateSold', 'taxAssessedValue'], axis=1, inplace=True)

# Changing column names to make more sense
houses.rename(columns={'unformattedPrice': 'price', 'area': 'sqft'}, inplace=True)

houses.head(2)


# In[96]:


houses.head(10)


# In[97]:


# Some of the area values are in sqft and some are in acres, let's convert all of them to acres
def convert_to_acres(row):
    if row['lotAreaUnit'] == 'sqft':
        return row['lotAreaValue'] / 43560
    else:
        return row['lotAreaValue']

houses['lotAreaValue'] = houses.apply(convert_to_acres, axis=1)

houses.head(10)


# In[98]:


houses.drop(['lotAreaUnit'], axis=1, inplace=True)


# In[99]:


# Check for duplicates
houses.duplicated().sum()


# In[100]:


# Looks like there are duplicates, some of the initial houses were repeated
houses[houses.duplicated()]


# In[101]:


houses.drop_duplicates(inplace=True)
houses.duplicated().sum()


# In[102]:


houses.isna().sum()


# In[103]:


houses.drop(['zestimate', 'rentZestimate'], axis=1, inplace=True)


# In[104]:


print(houses[houses.isna().any(axis=1)])


# In[105]:


houses.hist(bins=50, figsize=(20,15))


# In[106]:


# Remove the record if it doesn't have the number of bedrooms
houses.dropna(subset=['beds'], inplace=True)
houses.info()


# In[107]:


# Going to remove the remaning records with missing values
houses.dropna(subset=['latitude', 'longitude', 'sqft', 'lotAreaValue'], inplace=True)
houses.info()


# In[110]:


houses.describe()


# In[109]:


houses["price"].plot.hist(bins=50, figsize=(10,5))
plt.show()


# In[113]:


fig = go.Figure(data=go.Scattergeo(
        lon = houses['longitude'],
        lat = houses['latitude'],
        mode = 'markers'
        ))

fig.update_layout(
        title = 'Housing Data in Loveland, OH',
        geo_scope='usa'
    )

fig.show()


# In[111]:


# Correlation data for the price
houses.corr()['price'].sort_values(ascending=False)


# In[117]:


fig, ax = plt.subplots(figsize=(6,6))
sns.heatmap(houses.corr(), annot=True, ax=ax)


# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# In[118]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python LovelandHousing.ipynb')

