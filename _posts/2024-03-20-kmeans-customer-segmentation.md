---
layout: post
title: The "You Are What You Eat" Customer Segmentation
image: "/posts/clustering-title-img.png"
tags: [Customer Segmentation, Machine Learning, Clustering, Python]
---

In this project, I use k-means clustering to segment our client customer base in order to increase business understanding, and to enhance the relevancy of targeted messaging & customer communications.

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [01. Data Overview](#data-overview)
- [02. K-Means](#kmeans-title)
    - [Concept Overview](#kmeans-overview)
    - [Data Preprocessing](#kmeans-preprocessing)
    - [Finding A Good Value For K](#kmeans-k-value)
    - [Model Fitting](#kmeans-model-fitting)
    - [Appending Clusters To Customers](#kmeans-append-clusters)
    - [Segment Profiling](#kmeans-cluster-profiling)
- [03. Application](#kmeans-application)
- [04. Growth & Next Steps](#growth-next-steps)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

The Senior Management team from our client, a supermarket chain, can't come to a consensus about how customers are shopping, and how lifestyle choices may affect which food areas customers are shopping into, or more interestingly, not shopping into.

We've been asked to segment the customer based on engagement with each of the major food categories. This will help the business better understand their customer base and improve messaging targeting. 



### Actions <a name="overview-actions"></a>

We firstly needed to compile the necessary data from multiple tables in the database, namely the `transactions` table and the `product_areas` table.  

We joined together the relevant information using Pandas, and then aggregated the transactional data across product areas, from the most recent six month to a customer level. 

The final data for clustering is, for each customer, the percentage of sales allocated to each product area.

As a starting point, I wanted to test & apply k-means clustering for this task.  To do this there will be some data pre-processing required. 

The most important type of pre-processing is feature scaling to ensure all variables exist on the same scale - a very important consideration for distance based algorithms such as k-means.

K-means is an **unsupervised learning** approach, in other words there are no labels - we use a process known as **Within Cluster Sum of Squares (WCSS)** to understand what a "good" number of clusters or segments is.

Based upon this, we can then apply the k-means algorithm onto the product area data, append the clusters to our customer base, and then profile the resulting customer segments to understand what the differentiating factors were!


### Results <a name="overview-results"></a>

Based upon iterative testing using WCSS we settled on a customer segmentation with 3 clusters.  These clusters ranged in size, with Cluster 0 accounting for 73.6% of the customer base, Cluster 2 accounting for 14.6%, and Cluster 1 accounting for 11.8%.

There were some extremely interesting findings from profiling the clusters.

For **Cluster 0** we saw a significant portion of spend being allocated to each of the product areas - showing customers without any particular dietary preference. 

For **Cluster 1** we saw quite high proportions of spend being allocated to Fruit & Vegetables, but very little to the Dairy & Meat product areas.  A hypothesis that I thought of was that these customers are following a vegan diet.  

Finally customers in **Cluster 2** spent significant portions within Dairy, Fruit & Vegetables, but very little in the Meat product area - so similarly, there's evidence for a potential hypothesis that these customers are more along the lines of those following a vegetarian diet.

To help this segmentation drive the business, we have proposed to call this the **"You Are What You Eat"** segmentation.

### Growth/Next Steps <a name="overview-growth"></a>

Potential uses for this clustering/segmentation would be to run it at a lower level of product areas, so rather than just the four areas of Meat, Dairy, Fruit, Vegetables - clustering spend across the sub-categories **below** those categories.  

This would allow for more specific clusters, and in turn would allow both us and our client to get an even more granular understanding of dietary preferences within the customer base.

Here we've just focused on variables that are linked directly to sales - it could be interesting to also include customer metrics such as `distance to store`, `gender`, etc to give a even more well-rounded customer segmentation.

It would be useful to compare the outputs from a kmeans clustering test  to other clustering approaches such as hierarchical clustering or DBSCAN to compare the results.

___

# Data Overview  <a name="data-overview"></a>

We are primarily looking to discover segments of customers based upon their transactions within **food** based product areas so we will need to only select those.

In the code below, we:

* Import the required python packages & libraries
* Import the tables from the database
* Merge the tables to tag on **product_area_name** which only exists in the **product_areas** table
* Drop the non-food categories
* Aggregate the sales data for each product area, at customer level
* Pivot the data to get it into the right format for clustering
* Rescale the values from raw dollars, into a percentage of spend for each customer (to ensure each customer is comparable)

```python

# import required Python packages
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

# import tables from database
transactions = ...
product_areas = ...

# merge product_area_name on
transactions = pd.merge(transactions, product_areas, how = "inner", on = "product_area_id")

# drop the non-food category
transactions.drop(transactions[transactions["product_area_name"] == "Non-Food"].index, inplace = True)

# aggregate sales at customer level (by product area)
transaction_summary = transactions.groupby(["customer_id", "product_area_name"])["sales_cost"].sum().reset_index()

# pivot data to place product areas as columns
transaction_summary_pivot = transactions.pivot_table(index = "customer_id",
                                                    columns = "product_area_name",
                                                    values = "sales_cost",
                                                    aggfunc = "sum",
                                                    fill_value = 0,
                                                    margins = True,
                                                    margins_name = "Total").rename_axis(None,axis = 1)

# transform sales into % sales
transaction_summary_pivot = transaction_summary_pivot.div(transaction_summary_pivot["Total"], axis = 0)

# drop the "total" column as we don't need that for clustering
data_for_clustering = transaction_summary_pivot.drop(["Total"], axis = 1)

```
After the data pre-processing using Pandas, we have a dataset for clustering that looks like the below sample:


| **customer_id** | **dairy** | **fruit** | **meat** | **vegetables** |
|---|---|---|---|---|
| 2 | 0.246 | 0.198 | 0.394 | 0.162  |
| 3 | 0.142 | 0.233 | 0.528 | 0.097  |
| 4 | 0.341 | 0.245 | 0.272 | 0.142  |
| 5 | 0.213 | 0.250 | 0.430 | 0.107  |
| 6 | 0.180 | 0.178 | 0.546 | 0.095  |
| 7 | 0.000 | 0.517 | 0.000 | 0.483  |


The data is at customer level, and we have a column for each of the highest level food product areas.  Within each of those we have the *percentage* of sales that each customer allocated to that product area over the past six months.
___

# K-Means <a name="kmeans-title"></a>


### Concept Overview <a name="kmeans-overview"></a>

K-Means is an **unsupervised learning** algorithm, meaning that it does not look to predict known labels or values, but instead looks to isolate patterns within unlabelled data.

The algorithm works in a way where it partitions data-points into distinct groups (clusters) based upon their **similarity** to each other.

This similarity is called the euclidean (straight-line) distance between data-points in n-dimensional space.  

Each variable that is included lies on one of the dimensions in space.

The number of distinct groups (clusters) is determined by the value that is set for "k".

The algorithm does this by iterating over four key steps, namely:

1. It selects "k" random points in space (these points are known as centroids)
2. It then assigns each of the data points to the nearest centroid (based upon euclidean distance)
3. It then repositions the centroids to the **mean** dimension values of it's cluster
4. It then reassigns each data-point to the nearest centroid

Steps 3 & 4 continue to iterate until no data-points are reassigned to a closer centroid.


### Data Preprocessing <a name="kmeans-preprocessing"></a>

There are three vital preprocessing steps for k-means, namely:

* Dealing with missing values in the data
* Mitigating the effect of outliers
* Feature Scaling


##### Missing Values

Missing values can cause issues for k-means, as the algorithm won't know where to plot those data-points along the dimension where the value is not present.  

When there are missing values, there are several common options to deal with these:

These are to either remove the observations, or to impute to fill-in or to estimate what those value might be.

As we aggregated our data for each customer, we actually don't have this issue so we don't have to perform any pre-processing.


##### Outliers

As k-means is a distance based algorithm, outliers are one of the biggest culprits of accuracy problems. The main issue we face is when we come to scale our input variables, a very important step for a distance based algorithm.

We want to avoid “bunched up” variables due to a single outlier value, as this will make it hard to compare their values to the other input variables. We should always investigate outliers rigorously. Since we are dealing with percentages, outliers won't be a consideration.

##### Feature Scaling

Again, as k-means is a distance based algorithm, in other words it is reliant on an understanding of how similar or different data points are across different dimensions in n-dimensional space, feature scaling is one of the most important pre-processing techniques.

Feature Scaling is where we force the values from different columns to exist on the same scale, in order to improve the learning capabilities of the model. The two most common approaches for this are standardization, and normalization.

Standardization rescales data to have a mean of 0, and a standard deviation of 1 - meaning most datapoints will most often fall between values of around -4 and +4.

Normalization rescales datapoints so that they exist in a range between 0 and 1. In our case we are examining percentages between 0 and 1. This makes normalization a potentially more useful technique here.

Now, either approach is going to be *far better* for k-means clustering than no scaling at all.  I decided to use normalization as this will ensure all variables will end up having the same range, fixed between 0 and 1, and therefore the k-means algorithm can judge each variable in the same context.  Standardization **could** result in different ranges, variable to variable, and this is not so useful even though this isn't true in every scenario.

Another reason for choosing Normalization over Standardization is that our scaled data will **all** exist between 0 and 1, and these will then be compatible with any categorical variables that we have encoded as 1’s and 0’s (although we don't have any variables of this type in our task here).

In our specific task here, we are using percentages, so our values are **already** spread between 0 and 1.  We will still apply normalization for the following reason.  One of the product areas might commonly make up a large proportion of customer sales, and this may end up dominating the clustering space and algorithm's thinking.  If we normalize all of our variables, even product areas that make up smaller volumes, will be spread proportionately between 0 and 1 and create a less biased model.

The below code uses the in-built MinMaxScaler functionality from scikit-learn to apply Normalization to all of our variables.  The reason we create a new object (here called `data_for_clustering_scaled`) is that we want to use the scaled data for clustering, but when profiling the clusters later on, we may want to use the actual percentages as this may make more intuitive business sense, so it's good to have both options available!

```python

# create our scaler object
scale_norm = MinMaxScaler()

# normalise the data
data_for_clustering_scaled = pd.DataFrame(scale_norm.fit_transform(data_for_clustering), columns = data_for_clustering.columns)

```

### Finding A Good Value For k <a name="kmeans-k-value"></a>

At this point here, our data is ready to be fed into the k-means clustering algorithm.  Before that however, we want to understand what number of clusters we want the data split into.

In the world of unsupervised learning, there is no **right or wrong** value for this - it really depends on the data you are dealing with, as well as the unique scenario you're utilizing the algorithm for.  

From our client, having a very high number of clusters might not be appropriate as it would create too much granularity for the decision-makers of the business to make choices on.

Finding the "right" value for k, can feel more like art than science, but there are some data driven approaches that can help us!  

The approach we will choose here is known as **Within Cluster Sum of Squares (WCSS)** which measures the sum of the squared euclidean distances that data points lie from their closest centroid.  WCSS can help us understand the point where adding **more clusters** provides little extra benefit in terms of separating our data.

By default, the k-means algorithm within scikit-learn will use k = 8 meaning that it will look to split the data into eight distinct clusters.  We want to find a better value that fits our data, and our task!

So below, we tested multiple values for k, and plotted how this WCSS metric changes.  As we increase the value for k (in other words, as we increase the number or centroids or clusters) the WCSS value will always decrease.

However, these decreases will get smaller and smaller each time we add another centroid and we are looking for a point where this decrease is quite prominent **before** this point of diminishing returns.

```python

# set up range for search, and empty list to append wcss scores to
k_values = list(range(1,10))
wcss_list = []

# loop through each possible value of k, fit to the data, append the wcss score
for k in k_values:
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(data_for_clustering_scaled)
    wcss_list.append(kmeans.inertia_)

# plot wcss by k
plt.plot(k_values, wcss_list)
plt.title("Within Cluster Sum of Squares -  by k")
plt.xlabel("k")
plt.ylabel("WCSS Score")
plt.tight_layout()
plt.show()

```
That code gives us the below plot - which visualises our results!

![alt text](/img/posts/kmeans-optimal-k-value-plot.png "K-Means Optimal k Value Plot")


Based upon the shape of the above plot - there does appear to be an **elbow** (which we are searching for) at k = 3.  Prior to that we see a significant drop in the WCSS score, but following the decreases are much smaller, meaning this could be a point that suggests adding **more clusters** will provide little extra benefit in terms of separating our data.  

A small number of clusters can be beneficial when considering how easy it is for the business to focus on, and understand, each - so we will continue on, and fit our k-means clustering solution with k = 3. This will be the value we pass into our k-means clustering model.

### Model Fitting <a name="kmeans-model-fitting"></a>

The below code will instantiate our k-means object using a value for k equal to 3.  Then I fit this object into my scaled dataset to separate our data into three distinct segments or clusters.

```python

# instantiate our k-means object
kmeans = KMeans(n_clusters = 3, random_state = 42)

# fit to our data
kmeans.fit(data_for_clustering_scaled)

```


### Append Clusters To Customers <a name="kmeans-append-clusters"></a>

With the k-means algorithm fitted to our data, we can now append those clusters to the original dataset, meaning that each customer will be tagged with the cluster number that they most closely fit into based upon their sales data over each product area. This creates a classifier that can be used to cut our data in visualizations!

In the code below we tag this cluster number onto our original dataframe.

```python

# add cluster labels to our original data
data_for_clustering["cluster"] = kmeans.labels_

```

### Cluster Profiling <a name="kmeans-cluster-profiling"></a>

Once we have our data separated into distinct clusters, our client needs to understand **what is is** that is driving the separation.  

This provides valuable context for stakeholders within the business so they can understand the customers within each, and what type of behavior makes each segmentation unique.

##### Cluster Sizes

In the below code we firstly assess the number of customers that fall into each cluster.

```python

# check cluster sizes
data_for_clustering["cluster"].value_counts(normalize=True)

```

Running that gives the following outputs. We have three clusters are different in size, with the following proportions:

* Cluster 0: **73.6%** of customers
* Cluster 2: **14.6%** of customers
* Cluster 1: **11.8%** of customers

Based on these results, it does appear we do have a skew toward Cluster 0 with Cluster 1 & Cluster 2 being proportionally smaller.  This isn't right or wrong, it is simply showing up pockets of the customer base that are exhibiting different behaviors - and this is **exactly** what we and the client want. This helps drive strategic decision-making with data-backed segmentation.


##### Cluster Attributes

To understand what these different behaviors or characteristics are, we can look to analyze the attributes of each cluster, in terms of the variables we fed into the k-means algorithm.

```python

# profile clusters (mean % sales for each product area)
cluster_summary = data_for_clustering.groupby("cluster")[["Dairy","Fruit","Meat","Vegetables"]].mean().reset_index()

```
The code shows the mean spend percentages for the different food product areas for each cluster.

| **Cluster** | **Dairy** | **Fruit** | **Meat** | **Vegetables** |
|---|---|---|---|---|
| 0 | 22.1% | 26.5% | 37.7% | 13.8%  |
| 1 | 0.2% | 63.8% | 0.4% | 35.6%  |
| 2 | 36.4% | 39.4% | 2.9% | 21.3%  |


For **Cluster 0** we see a reasonably significant portion of spend being allocated to each of the product areas.  For **Cluster 1** we see quite high proportions of spend being allocated to Fruit & Vegetables, but very little to the Dairy & Meat product areas.  Again, as seen in the tl:dr a hypothesis could be formed that these customers are following a vegan diet.  

Finally customers in **Cluster 2** spend, on average, significant portions within Dairy, Fruit & Vegetables, but very little in the Meat product area - so similarly, an early hypothesis could be that these customers are more along the lines of those following a vegetarian diet which would yield some interesting results.

___
# Application <a name="kmeans-application"></a>

Even those this is a simple solution, having a look at high level product areas and creating customer profiles will help leaders in the business, and category managers gain a clearer understanding of the customer base.

Tracking these clusters over time would allow the client to more quickly react to dietary trends, adjust their messaging, and even improve inventory re-stocking cadences.

Based upon these clusters, the client will be able to target customers more accurately - promoting products & discounts to customers that are truly relevant to them - overall enabling a more customer focused communication strategy.

___
# Growth & Next Steps <a name="growth-next-steps"></a>

It would be interesting to run this clustering/segmentation at a lower level of product areas. For example, clustering spend across the sub-categories *below* the four categories we had a look at.  This would mean we could create more specific clusters, and get an even more granular understanding of dietary preferences within the customer base.

Here we've just focused on variables that are linked directly to sales - it could be interesting to also include other customer metrics such as `distance to store`, `gender` etc to give a even more well-rounded customer segmentation.

These applications could be applied to a variety of industries like marketing to see similar customers based on page viewing habits or app usage, cluster athletes based on physical profiles or statistical performance and use this to inform draft strategy. There's a ton of different high-value ideas. 
