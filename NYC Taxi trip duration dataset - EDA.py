#!/usr/bin/env python
# coding: utf-8

# ## Data Preparation and pre-processing

# In[1]:


# importing the required libraries and packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# reading the dataset
data = pd.read_csv('nyc_taxi_trip_duration.csv')

# also creating a copy of the dataset
data_copy = data.copy(deep=True)


# In[3]:


# looking at the head of the dataset to make sure it has been read properly
data.head()


# In[4]:


data.tail()


# In[5]:


# shape of our dataset (number of records and number of columns)
data.shape


# In[6]:


# columns that we have in our dataset
data.columns.tolist()


# In[7]:


# data type of each column
data.dtypes


# In[8]:


# number of unique values in each column
data.nunique()


# In[9]:


# missing value % in each column
data.isnull().mean()*100


# There are no missing values in the dataset

# In[10]:


# looking for duplicates in the dataset
data.duplicated(subset=data.columns.tolist()[1:]).sum()


# * There are no duplicates in the dataset

# In[11]:


# changing pickup and dropoff datetime columns to datetime values
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
data['dropoff_datetime'] = pd.to_datetime(data['dropoff_datetime'])

# creating separate variables for month, date, and hour of pickup
data['pickup_year'] = data['pickup_datetime'].dt.year
data['pickup_month'] = data['pickup_datetime'].dt.month
data['pickup_date'] = data['pickup_datetime'].dt.day
data['pickup_dow'] = data['pickup_datetime'].dt.weekday
data['pickup_hour'] = data['pickup_datetime'].dt.hour

# creating separate variables for month, date, and hour of drop
data['dropoff_year'] = data['dropoff_datetime'].dt.year
data['dropoff_month'] = data['dropoff_datetime'].dt.month
data['dropoff_date'] = data['dropoff_datetime'].dt.day
data['dropoff_dow'] = data['dropoff_datetime'].dt.weekday
data['dropoff_hour'] = data['dropoff_datetime'].dt.hour

# dropping the pickup and dropoff datetime columns
data.drop(columns=['pickup_datetime', 'dropoff_datetime'], inplace=True)


# In[12]:


# mapping Y and N values in store_and_fwd_flag column to 0 and 1
data['store_and_fwd_flag'] = data['store_and_fwd_flag'].map({'N': 0, 'Y':1})


# In[13]:


# creating a variable for difference between pickup and dropoff coordinates
data['distance']=abs(data.pickup_longitude-data.dropoff_longitude)+abs(data.pickup_latitude-data.dropoff_latitude)


# In[14]:


# having a look at the dataset after above changes
data.head()


# In[15]:


# final check of the data types of newly created variables
data.dtypes


# ## Univariate Analysis

# * We can automate the process of EDA using the sweetviz library, which will generate an HTML report of all variables

# In[16]:


# importing sweetviz library for generating EDA report
import sweetviz as sv


# In[17]:


report = sv.analyze(data)
report.show_html()


# * We will also go through each variable manually

# In[18]:


data.vendor_id.value_counts(normalize=True).sort_index().plot(kind='bar')
plt.xticks(rotation=0)
plt.xlabel('vendor_id');


# * More rides have been booked through vendor 2

# In[19]:


data.passenger_count.value_counts(normalize=True).sort_index().plot(kind='bar')
plt.xticks(rotation=0)
plt.xlabel('passenger_count');


# * Most of the rides have 1 passenger
# 
# * 0, 7, and 9 are the least frequent categories

# In[20]:


data.pickup_longitude.describe()


# In[21]:


sns.boxplot(x=data.pickup_longitude);


# * New York City longitude is -74 degrees. So we need to remove those values that are less than -74.5 and more than -73.5

# In[22]:


# number of records that satisfy this criteria
len(data.loc[(data.pickup_longitude < -74.5) | (data.pickup_longitude > -73.5)].index.tolist())


# In[23]:


# dropping the records that satify this criteria
data.drop(data.loc[(data.pickup_longitude < -74.5) | (data.pickup_longitude > -73.5)].index, inplace=True)


# In[24]:


sns.boxplot(x=data.pickup_longitude);


# * Although there still seem to be a lot of outlier values, we will continue to keep them in the dataset assuming these are trips to places around New York

# In[25]:


data.pickup_latitude.describe()


# In[26]:


sns.boxplot(x=data.pickup_latitude);


# * New York City longitude is 40.7 degrees. So we need to remove those values that are less than 40.2 and more than 41.2

# In[27]:


# number of records that satisfy this criteria
len(data.loc[(data.pickup_latitude < 40.2) | (data.pickup_latitude > 41.2)].index.tolist())


# In[28]:


# dropping the records that satify this criteria
data.drop(data.loc[(data.pickup_latitude < 40.2) | (data.pickup_latitude > 41.2)].index, inplace=True)


# In[29]:


sns.boxplot(x=data.pickup_latitude);


# * Although there still seem to be a lot of outlier values, we will continue to keep them in the dataset assuming these are trips to places around New York

# In[30]:


data.dropoff_longitude.describe()


# In[31]:


sns.boxplot(x=data.dropoff_longitude);


# * Although data seems to contain a few outliers, we will continue to keep them in the dataset for now

# In[32]:


data.dropoff_latitude.describe()


# In[33]:


sns.boxplot(x=data.dropoff_latitude);


# * Although data seems to contain a few outliers, we will continue to keep them in the dataset for now

# In[34]:


data.store_and_fwd_flag.value_counts(normalize=True).plot(kind='bar')
plt.xticks(rotation=0)
plt.xlabel('store_and_fwd_flag');


# In[35]:


data.store_and_fwd_flag.value_counts(normalize=True)*100


# * Almost all records have store_and_fwd_flag value as zero. We will drop this variable because it will not add any value to our analysis

# In[36]:


# dropping the store_and_fwd_flag column
data.drop(columns='store_and_fwd_flag', inplace=True)


# In[37]:


sns.boxplot(x=data.trip_duration);


# * Clearly the maximum value is too large. We will drop that record and then look at the data

# In[38]:


# dropping the record with the largest value
data.drop(data.loc[data.trip_duration == data.trip_duration.max()].index, inplace=True)


# In[39]:


sns.boxplot(x=data.trip_duration);


# * There seem to be a lot of outlier values in this data. We will drop all those records where trip duration is more than 6 hours i.e. 21600 seconds

# In[40]:


# number of records that satify this criteria
len(data.loc[data.trip_duration > 21600].index.tolist())


# In[41]:


# dropping the records that satify this criteria
data.drop(data.loc[data.trip_duration > 21600].index, inplace=True)


# In[42]:


sns.boxplot(x=data.trip_duration);


# * Data looks much better now

# In[43]:


data.pickup_year.value_counts(normalize=True)*100


# * All records are from 2016, so we will drop this feature

# In[44]:


# dropping the pickup_year column
data.drop(columns='pickup_year', inplace=True)


# In[45]:


data.pickup_month.value_counts(normalize=True).sort_index().plot(kind='bar')
plt.xticks(rotation=0)
plt.xlabel('pickup_month');


# * Number of trips are fairly evenly distributed throughout the 6 months

# In[46]:


data.pickup_date.value_counts().sort_index().plot(kind='line', marker='o')
plt.xlabel('pickup_date');


# * Number of trips seem to drop off in the second half of the month

# In[47]:


data.pickup_dow.value_counts(normalize=True).sort_index().plot(kind='bar')
plt.xticks(rotation=0);


# * Number of trips increase from Monday to Saturday, and then drop on Sunday

# In[48]:


data.pickup_hour.value_counts().sort_index().plot(kind='line', marker='o')
plt.xlabel('pickup_hour');


# * Number of rides are the least during late night and early morning hours (12AM-6AM)
# 
# * Number of rides seem to pick up after 5 AM, and peak during the evening (5PM-8 PM)

# In[49]:


data.dropoff_year.value_counts(normalize=True)*100


# * All records are from 2016, so we will drop this feature

# In[50]:


# dropping the dropoff_year column
data.drop(columns='dropoff_year', inplace=True)


# In[51]:


data.dropoff_month.value_counts(normalize=True).sort_index().plot(kind='bar')
plt.xticks(rotation=0)
plt.xlabel('dropoff_month');


# In[52]:


data.dropoff_date.value_counts().sort_index().plot(kind='line', marker='o')
plt.xlabel('dropoff_date');


# In[53]:


data.dropoff_dow.value_counts(normalize=True).sort_index().plot(kind='bar')
plt.xticks(rotation=0);


# In[54]:


data.dropoff_hour.value_counts().sort_index().plot(kind='line', marker='o')
plt.xlabel('dropoff_hour');


# In[55]:


data.distance.describe()


# In[56]:


sns.boxplot(x=data.distance);


# * Although it looks like there are a lot of outliers in this data, we will keep all records for now

# In[57]:


data.trip_duration.describe()


# In[58]:


sns.boxplot(x=data.trip_duration);


# In[59]:


sns.kdeplot(data.trip_duration, shade=True);


# ## Bivariate Analysis

# In[60]:


data.groupby('vendor_id')['trip_duration'].mean()


# * Trip duration is ~ 12 seconds more in case of rides booked through vendor 2

# In[61]:


data.groupby('passenger_count')['trip_duration'].agg(['count', 'mean'])


# In[62]:


data.groupby('passenger_count')['trip_duration'].mean().plot(kind='bar', label='Trip duration')
plt.plot([data.trip_duration.mean()]*data.passenger_count.nunique(), 'r--', label='Avg trip duration')
plt.xticks(rotation=0)
plt.legend();


# * passenger_count 0, 7, and 9 are very less frequent
# 
# * Categories 1-6 have trip duration more or less around the average trip duration

# In[63]:


data.groupby('pickup_month')['trip_duration'].mean().plot(marker='o')
plt.ylim(600);


# * Trip duration has increased over the months

# In[64]:


data.groupby('pickup_date')['trip_duration'].mean().plot(marker='o', label='Trip duration')
plt.plot(data.groupby('pickup_date')['trip_duration'].mean().index, [data.trip_duration.mean()]*data.pickup_date.nunique(), 'r--', label='Avg trip duration');
plt.legend();


# * Trip duration is varying quite a lot with pickup date

# In[65]:


data.groupby('pickup_dow')['trip_duration'].mean().plot(kind='bar', label='Trip duration')
plt.plot([data.trip_duration.mean()]*data.pickup_dow.nunique(), 'r--', label='Avg trip duration')
plt.xticks(rotation=0)
plt.legend();


# * Trip duration during weekends is lesser than average

# In[66]:


data.groupby('pickup_hour')['trip_duration'].mean().plot(marker='o', label='Trip duration')
plt.plot(data.groupby('pickup_hour')['trip_duration'].mean().index, [data.trip_duration.mean()]*data.pickup_hour.nunique(), 'r--', label='Avg trip duration');
plt.legend();


# * Trip duration in early morning trips is much less
# 
# * Trip duration starts picking up after 6 AM, and peak around 2-6 PM

# * Dropoff and pickup date time variables are strongly related. So we will only look at dropoff_hour

# In[67]:


data.groupby('dropoff_hour')['trip_duration'].mean().plot(marker='o', label='Trip duration')
plt.plot(data.groupby('dropoff_hour')['trip_duration'].mean().index, [data.trip_duration.mean()]*data.dropoff_hour.nunique(), 'r--', label='Avg trip duration');
plt.legend();


# * Dropoff hour plot looks very similar to pickup hour plot

# In[68]:


plt.scatter(data.distance, data.trip_duration)
plt.xlabel('distance')
plt.ylabel('trip_duration');


# ## Multivariate Analysis

# In[69]:


data['pickup_hour_cat'] = 'str'
data['pickup_hour_cat'] = np.where(data['pickup_hour'].isin([0, 1, 2, 3]), 'Late Night', data['pickup_hour_cat'])
data['pickup_hour_cat'] = np.where(data['pickup_hour'].isin([4, 5, 6, 7]), 'Early Morning', data['pickup_hour_cat'])
data['pickup_hour_cat'] = np.where(data['pickup_hour'].isin([8, 9, 10, 11]), 'Morning', data['pickup_hour_cat'])
data['pickup_hour_cat'] = np.where(data['pickup_hour'].isin([12, 13, 14, 15]), 'Afternoon', data['pickup_hour_cat'])
data['pickup_hour_cat'] = np.where(data['pickup_hour'].isin([16, 17, 18, 19]), 'Evening', data['pickup_hour_cat'])
data['pickup_hour_cat'] = np.where(data['pickup_hour'].isin([20, 21, 22, 23]), 'Night', data['pickup_hour_cat'])


# In[70]:


data['dropoff_hour_cat'] = 'str'
data['dropoff_hour_cat'] = np.where(data['dropoff_hour'].isin([0, 1, 2, 3]), 'Late Night', data['dropoff_hour_cat'])
data['dropoff_hour_cat'] = np.where(data['dropoff_hour'].isin([4, 5, 6, 7]), 'Early Morning', data['dropoff_hour_cat'])
data['dropoff_hour_cat'] = np.where(data['dropoff_hour'].isin([8, 9, 10, 11]), 'Morning', data['dropoff_hour_cat'])
data['dropoff_hour_cat'] = np.where(data['dropoff_hour'].isin([12, 13, 14, 15]), 'Afternoon', data['dropoff_hour_cat'])
data['dropoff_hour_cat'] = np.where(data['dropoff_hour'].isin([16, 17, 18, 19]), 'Evening', data['dropoff_hour_cat'])
data['dropoff_hour_cat'] = np.where(data['dropoff_hour'].isin([20, 21, 22, 23]), 'Night', data['dropoff_hour_cat'])


# In[71]:


data.groupby(['pickup_hour_cat', 'pickup_dow'])['trip_duration'].mean().unstack()


# In[72]:


data.groupby(['pickup_hour_cat', 'pickup_dow'])['trip_duration'].mean().unstack().T.plot(marker='o')
plt.plot([data.trip_duration.mean()]*data.pickup_dow.nunique(), 'r--o', label='Avg trip duration')
plt.legend(loc='lower center');


# * Afternoon trips are the longest on average
# 
# * During weekend, trip durations are lesser than average
# 
# * Morning trips are longer in duration during weekdays but smaller during weekends (because weekends are generally off)

# In[73]:


data.groupby(['dropoff_hour_cat', 'dropoff_dow'])['trip_duration'].mean().unstack().T.plot(marker='o')
plt.plot([data.trip_duration.mean()]*data.pickup_dow.nunique(), 'r--o', label='Avg trip duration')
plt.legend(loc='lower center');


# * Dropoff plot looks very similar to pickup plot

# ## Correlation Heatmap

# In[74]:


# correlation between variables
corr = data.corr()


# In[75]:


# heatmap of correlation coefficients
plt.figure(dpi=120)
sns.heatmap(corr, vmin=0, cmap='YlGnBu', linewidth=0.5);


# * Pickup and dropoff date and time variables are highly correlated
# 
# * Pickup and dropoff latitude and longitude variables are slightly correlated
# 
# * Distance and pickup and dropoff longitude are slightly correlated
# 
# * Other variables are very less correlated, which is a good thing from a modeling perspective

# In[ ]:




