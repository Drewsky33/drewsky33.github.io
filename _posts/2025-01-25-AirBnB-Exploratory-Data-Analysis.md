---
layout: post
title: Analyzing Paris Airbnb Listings and the Impact of 2015 Regulations
image: "/posts/paris_airbnb_market_analysis.jpg"
tags: [Airbnb, Data Analysis, Pricing Trends, Regulations, Python]
---

In this project, I analyze Airbnb listings in Paris with a focus on pricing trends and the impact of regulations introduced in 2015. 

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results & Discussion](#overview-results)
- [01. Data Overview & Preparation](#data-overview)
- [02. Pricing Trends in Paris](#pricing-trends)
- [03. Impact of 2015 Regulations](#regulation-impact)
- [04. Key Insights](#key-insights)
- [05. Conclusion](#conclusion)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

As Airbnb has grown in popularity, many cities have introduced regulations to limit the number of properties listed. Paris, one of Airbnb’s most active markets, saw significant regulatory changes in **2015**, aimed at curbing short-term rentals. This analysis seeks to explore **pricing trends**, **market dynamics**, and **the impact of these regulations on new host registrations**.

To achieve this, I analyzed **Airbnb listing data for Paris**, focusing on:

- **Price trends across neighborhoods**
- **Price differences based on accommodation capacity**
- **The number of new hosts over time**
- **The effect of 2015 regulations on host activity and pricing**

### Actions <a name="overview-actions"></a>

To conduct this analysis, the following steps were taken:
- **Data was imported and cleaned**
- **Data was QA'd to ensure accurate analysis.**
- **I filtered down the listings dataset to only Paris listings.**
- **Data was reshaped (aggregated) for analysis.**
- **A helper function to add a formatted subtitle was created.**
- **A 1 page report with the relevant visualizations was created.**
___

# Analysis TL;DR <a name="analysis-summary"></a>

- **New host registrations dropped significantly in response to 2015 regulatory changes**, which illustrated the impact of the regulatory changes.
- **Despite a drop in listings in response to regulartory changes, the average listing price continued to rise until a recent drop in 2022**, this could potentially be attributed to market adjustments.
- **Elysee remains the most expensive neighborhood**, with higher-priced listings persisting even after the regulations.
___

# Data Overview & Preparation <a name="data-overview"></a>

### **Loading and Cleaning Data**

First, I had to import the data. I had to ensure the encoding was correct and also parsed host_since to conver to a datetime.

```python
import pandas as pd

# Load dataset
# Read in AirBnB data
path = '/######/######/######/######/######/######/Airbnb Data/Listings.csv'
df = pd.read_csv(path,
                 encoding = "ISO-8859-1",
                 low_memory = False,
                 parse_dates = ["host_since"]
                )
# Explore first few rows
df.head()

```
##### Add Photo

After that, I wanted to ensure that the data was read in correctly. So I used `.info()` to look for missing data and ensure the columns were of the correct data type. 

``` python

# Check the data types of columns
df.info()

```
##### Add Photo

Following that, I filtered the data down to records where the `city` was Paris to examine Paris. Finally, I grabbed every row for a select set of relevant columns. 

``` Python

# Use Query to filter down
paris_listings = (
    df
    .query("city == 'Paris'")
    .loc[:, ["host_since", "neighbourhood", "city",
             "latitude", "longitude", "property_type", "room_type",
             "accommodates", "price"]]
)

paris_listings.head(10)

```
##### Add Photo

### **QA the Data**
Next, I wanted to look at the missing entries of the dataset to see if it would potentially skew our data. There were 33 missing entries in the `host_since` column. Not enough to impact our data so I proceeded.

``` python
# Check the missing listings
paris_listings.isna().sum()

```
##### Add Photo

Then I profiled the numerical columns of our datasets using the `.describe()` call. 

```python
# Profile our numeric columns
paris_listings.describe()

```

I noticed the `accommodates` column had a minimum was zero so I wanted to see just how many. So did the `price` column. 

``` python

# Count How Many times Accommodates = 0
paris_listings.query("accommodates == 0 ").count()

# Count How Many times price = 0
paris_listings.query("price == 0 ").count()

```
Accommodates had 54 rows with zero and price had 62. Neither had significant records so, I opted to keep them. 
##### Add Photo

### **Data Aggregation**
First, I needed to create a dataset that would allow me to visualize the average price of each neighborhood. 

```python
# Group by Neighbourhood
paris_listings_neighborhood = (
    df.groupby('neighbourhood')
    .agg({'price': 'mean'})
    .sort_values(by="price", ascending=False)
)

```

Then, I needed to look  at the average price for different accommodation levels in the most expensive neighborhood "Elysee".

``` python
# Filter Data for Accommodations in Elysee
paris_listings_accommodations = (
    df.query("neighbourhood == 'Elysee'")
    .groupby("accommodates")
    .agg({"price": "mean"})
    .sort_values("price", ascending=False)
)

```

Finally, I created a time series to look at the average price and the number of listings over time at the yearly level. 

``` python
# Create a time-series dataset
paris_listings_over_time = (
    df.set_index("host_since")
    .resample("YE")
    .agg({
        "neighbourhood": "count",
        "price": "mean", 
    })
)
```
___

# **STOPPED HERE RESUME TOMORROW**

# Pricing Trends in Paris <a name="pricing-trends"></a>

### **Average Price by Neighborhood**
```python
import matplotlib.pyplot as plt

# Plot neighborhood price data
fig, ax = plt.subplots(figsize=(10,6))
ax.barh(paris_listings_neighborhood.index, paris_listings_neighborhood['price'], color="navy")
ax.set_title("Average Price by Neighborhood", loc='left', fontsize=14, fontweight='bold')
ax.set_xlabel("Average Price (€)")
ax.set_ylabel("Neighborhood")
plt.show()
```

**_Screenshot Placeholder: Average Price by Neighborhood_**

### **Average Price by Accommodates in Elysee**
```python
# Plot accommodates price data
fig, ax = plt.subplots(figsize=(10,6))
ax.barh(paris_listings_accommodations.index, paris_listings_accommodations['price'], color="navy")
ax.set_title("Avg Price by Accommodates in Elysee", loc='left', fontsize=14, fontweight='bold')
ax.set_xlabel("Average Price (€)")
ax.set_ylabel("Accommodates")
plt.show()
```

**_Screenshot Placeholder: Average Price by Accommodates in Elysee_**

___

# Impact of 2015 Regulations <a name="regulation-impact"></a>

### **New Hosts Over Time**
```python
# Plot new hosts over time
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(paris_listings_over_time.index, paris_listings_over_time["neighbourhood"], color="navy", marker="v", linewidth=2)
ax.axvspan(pd.Timestamp("2015-01-01"), pd.Timestamp("2016-01-01"), color="grey", alpha=0.2, label="Regulation Period")
ax.set_title("Count of New Hosts Over Time", loc='left', fontsize=14, fontweight='bold')
ax.set_xlabel("Year")
ax.set_ylabel("New Hosts")
ax.legend()
plt.show()
```

**_Screenshot Placeholder: New Hosts Over Time_**

### **Average Price Over Time**
```python
# Plot average price over time
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(paris_listings_over_time.index, paris_listings_over_time["price"], color="navy", marker="v", linewidth=2)
max_price = paris_listings_over_time["price"].max()
ax.axhspan(max_price - 5, max_price + 5, color="grey", alpha=0.2, label="Max Price Range")
ax.set_title("Average Price Over Time", loc='left', fontsize=14, fontweight='bold')
ax.set_xlabel("Year")
ax.set_ylabel("Average Price (€)")
ax.legend()
plt.show()
```

**_Screenshot Placeholder: Average Price Over Time_**

___

# Key Insights <a name="key-insights"></a>

1. **Regulations Introduced in 2015 Significantly Reduced New Host Registrations**
2. **Despite Regulations, Prices Have Increased Over Time**
3. **Elysee Continues to be the Most Expensive Neighborhood**
4. **Fewer Listings, More Competition**

___

# Conclusion <a name="conclusion"></a>

- **The 2015 Airbnb regulations had a strong impact on market entry**, reducing new host participation.
- **Pricing trends indicate that Airbnb remains lucrative in Paris**, especially in high-demand neighborhoods.
- **Further research could explore guest booking behaviors** to understand whether **regulations have impacted consumer demand or only host supply**.

The results of this analysis can be useful for **policymakers**, **Airbnb hosts**, and **investors**, helping them better understand how regulations shape market dynamics in short-term rentals.

___



