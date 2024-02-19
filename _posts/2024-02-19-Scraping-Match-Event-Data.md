---
layout: post
title: Scraping Soccer Match Event Data
image: "/posts/Saka.jpeg"
tags: [Web Scraping, Sports Analytics, Database Creation, Python]
---

In this project, I create a web scraping pipeline for scraping data from [whoscored](https://www.whoscored.com/)

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results & Discussion](#overview-results)
- [01. Concept Overview](#concept-overview)
	- [Web Scraping](#concept-Webscraping)
	- [Data Modeling](#concept-Datamodeling)
	- [Automation](#concept-Automation)
- [02. Data Overview & Preparation](#data-overview)
- [03. Data Model with Pydantic](#data-model)
- [04. Automating the Pipeline](#data-automation)
- [05. Next Steps](#discussion)

___

# Project Overview  <a name="overview-main"></a>

#### Context <a name="overview-context"></a>

I've been wanting to find a way to build my own database so that I can get soccer match data to make predictions on, visualize, and extend analysis into the world of sports betting. I found a course done by `Mckay Johns that taught me the basics of webscraping. 

The goal is to create pipeline for match event data (this specific post) and then do the same for historical data. 


#### Actions <a name="overview-actions"></a>

For this project, I used beautiful soup for scraping along with selenium, pandas and numpy for transformations and string manipulation, json for dealing with html data and converting them to dictionaries, pydantic for creating data models, time for responsible scraping, supbabase where I'll host the data and typing for bringing in lists. 


The goal is to create 4 tables: match_events, players, teams, and match. They'll be linked through id's that map each to one another. 


___

# Concept Overview  <a name="concept-overview"></a>


#### Web Scraping <a name="concept-Webscraping"></a>

Web scraping at a high levels involves extracting data from websites using packages like BeautifulSoup and requests. It allows us to automate data collection from web pages by parsing the HTML and extracting the required information. 



#### Data Modeling <a name="concept-Datamodeling"></a>

Data modeling is to structure and organize data in a way that makes it easily manageable, understandable, and usable. It is crucial for this task because we need to be able to find information about players outside of a certain match, teams outside of a certain match, or look up a specific match to get that information. 
By creating a data model with Pydantic we can store our data in an efficient in and meaningful manner for easier analysis and data retrieval going forward. 



#### Using functions to automate a pipeline <a name="concept-Automation"></a>

The use of functions to automate this pipeline will be useful because it will allow me to parse a page, clean the data, and extract information for different variants of that page. In this case, getting different matches associated to a team based on url's for that team.


___


# Data Overview & Preparation  <a name="data-overview"></a>

#### Import Packages <a name="prep-import"></a>
First, I'll import all of the packages mentioned:

``` python

# Import Packages Needed
import json
import time

# Import statistical packages
import numpy as np
import pandas as pd

# Import Scraper
from bs4 import BeautifulSoup
from selenium import webdriver

# Prepare data for insertion
from pydantic import BaseModel
from typing import List, Optional
from supabase import create_client, Client

```
#### Initiate Driver and Pass Url <a name="prep-initiate"></a>

* Load in the Python libraries we require for scraping data, transforming it, and pushing to a cloud PostgreSQL database.
* **Next,** I need to initiate the driver. 

``` python
driver = webdriver.Chrome()
```

From there I'll pass in the url for the site I want to scrape which who scored. 

```python
# WhoScored Url
whoscored_url = "https://www.whoscored.com/Matches/1729476/Live/England-Premier-League-2023-2024-Manchester-City-Everton"
```

From there, I pass the driver into the URL:

``` python
# Pass the url into the driver
driver.get(whoscored_url)
```

#### Parse Webpage with Beautiful Soup <a name="prep-soup"></a>

``` python
# Use Soup
soup = BeautifulSoup(driver.page_source, 'html.parser')

# We can't access stuff from a `<div> object` So we need to grab keywords from that script
element = soup.select_one('script:-soup-contains("matchCentreData")') # let's use css selector
```
Now that we have I have the element part of the html, we are going to split on the line break and get only teh first element. 

``` python 

# Pull in match dictionary
# Splitting the string on the phrase below and selection everything after that split
# Second split is splitting on the line break phrase \n to remove the next dictionary , first element

matchdict = json.loads(element.text.split("matchCentreData: ")[1].split(",\n")[0])

``` 

When we look at the new match output dictionary, we can see that we have a a variety of key value pairs related to the match. we have a nested dictionary for player's names and id's, venueName, elapsed time, scores at ht, ft, a dictionary for home team, one for away, and even more. 

#### Parse Webpage with Beautiful Soup <a name="prep-cleaning"></a>

After looking at a few of the keys within the `matchdict` I located the information I need within `events`. So next, I'm going to pass the  events portion of the dictionary into a pandas dataframe. I also need to drop null player id columns. 

``` python
match_events = matchdict['events']

df = pd.DataFrame(match_events)

## I want to see every column of my data frame
# Set the option to display all columns
pd.set_option('display.max_columns', None)

df[df['playerId'].isna()]

# Drop player id columns where there are nulls
df.dropna(subset = 'playerId', inplace = True)
```

Now, we still have null values in other columns. What we we want to do is assign a value of None so that we can use this within a postgresSQL database.

``` python
df = df.where(pd.notnull(df), None)

# Observe the columns we have
df.columns

``` 
`['id', 'eventId', 'minute', 'second', 'teamId', 'x', 'y',
       'expandedMinute', 'period', 'type', 'outcomeType', 'qualifiers',
       'satisfiedEventsTypes', 'isTouch', 'playerId', 'endX', 'endY',
       'relatedEventId', 'relatedPlayerId', 'blockedX', 'blockedY',
       'goalMouthZ', 'goalMouthY', 'isShot', 'cardType', 'isGoal']`

Then, I'll make my column names more readable. 

``` python
df = df.rename(
    {
        'eventId': 'event_id',
        'expandedMinute': 'expanded_minute',
        'outcomeType': 'outcome_type',
        'isTouch': 'is_touch',
        'playerId': 'player_id',
        'teamId': 'team_id',
        'endX': 'end_x',
        'endY': 'end_y',
        'blockedX': 'blocked_x',
        'blockedY': 'blocked_y',
        'goalMouthZ': 'goal_mouth_z',
        'goalMouthY': 'goal_mouth_y',
        'isShot': 'is_shot',
        'cardType': 'card_type',
        'isGoal': 'is_goal'
        
    },
    axis = 1

)
```
We also have to clean up the display names for each event that happens. We need this to know the action for that event. 

``` python
# Grab every single display name from period

df['period_display_name'] = df['period'].apply(lambda x: x['displayName'])

# Grab every single display name from type at row level
df['type_display_name'] = df['type'].apply(lambda x: x['displayName'])

# Check the outcome type
df['outcome_type_display_name'] = df['outcome_type'].apply(lambda x: x['displayName'])

# Drop unneeded columns
df.drop(columns = ['period', 'type', 'outcome_type'], inplace = True)

# Create new dataframe with only columns I want
# Subset data with wanted column names
df = df[[
    'id', 'event_id', 'minute', 'second', 'team_id', 'player_id', 'x', 'y', 'end_x', 'end_y', 
    'qualifiers', 'is_touch', 'blocked_x', 'blocked_y', 'goal_mouth_z', 'goal_mouth_y', 'is_shot', 
    'card_type', 'is_goal', 'type_display_name', 'outcome_type_display_name', 'period_display_name'
]]
```

After I've done this, I need to ensure that id's are of the correct type. Id's need to be integers, seconds, x/y coordinates should be floats, make sure that binary outcomes are booleans. Also need to 

``` python
# Clean id's to be integer type
df[['id', 'event_id', 'minute', 'team_id', 'player_id']] = df[['id', 'event_id', 'minute', 'team_id', 'player_id']].astype(int)

# Cast seconds, times, or spacial data as floats
df[['second', 'x', 'y', 'end_x', 'end_y']] = df[['second', 'x', 'y', 'end_x', 'end_y']].astype(float)

# Cast binary outcomes as booleans
df[['is_shot', 'is_goal', 'card_type']] = df[['is_shot', 'is_goal', 'card_type']].astype(bool)


# Remove null values for the binary outcomes
df['is_goal'] = df['is_goal'].fillna(False)
df['is_shot'] = df['is_shot'].fillna(False)
df.iloc[0].to_dict()

```
Finally, we need to make sure that float fields are filled with a None value for ingestion into the database.



# Data Modeling with Pydantic  <a name="data-model"></a>

``` python
class MatchEvent(BaseModel):
    id: int
    event_id: int
    minute: int
    second: float
    team_id: int
    player_id: int
    x: float
    y: float
    end_x: float
    end_y: float
    qualifiers: List[dict]
    is_touch: bool
    blocked_x: Optional[float] = None
    blocked_y: Optional[float] = None
    goal_mouth_z: Optional[float] = None
    goal_mouth_y: Optional[float] = None
    is_shot: bool
    card_type: bool
    is_goal: bool
    type_display_name: str
    outcome_type_display_name: str
    period_display_name: str

class Player(BaseModel):
    player_id: int
    shirt_no: int
    name: str
    age: int
    position: str
    team_id: int
    height: int
    weight: int

class Team(BaseModel):
    team_id: int
    name: str
    country_name: str
    manager_name: str

```

I haven't finished testing the match table, but below are the scripts for creating the table in Supa Base:

``` sql
CREATE TABLE match_event (

id BIGINT PRIMARY KEY,

event_id INTEGER NOT NULL,

minute INTEGER NOT NULL,

second FLOAT,

team_id INTEGER NOT NULL,

player_id INTEGER NOT NULL,

x FLOAT NOT NULL,

y FLOAT NOT NULL,

end_x FLOAT,

end_y FLOAT,

-- For qualifiers, use JSONB as it allows storing a list of dictionaries

qualifiers JSONB NOT NULL,

is_touch BOOLEAN NOT NULL,

blocked_x FLOAT,

blocked_y FLOAT,

goal_mouth_z FLOAT,

goal_mouth_y FLOAT,

is_shot BOOLEAN NOT NULL,

card_type BOOLEAN NOT NULL,

is_goal BOOLEAN NOT NULL,

type_display_name TEXT NOT NULL,

outcome_type_display_name TEXT NOT NULL,

period_display_name TEXT NOT NULL



-- Players Table
CREATE TABLE IF NOT exists players (

player_id BIGINT PRIMARY KEY,

shirt_no INTEGER NOT NULL,

name TEXT NOT NULL,

age INTEGER NOT NULL,

position TEXT NOT NULL,

team_id INTEGER NOT NULL,

height INTEGER NOT NULL,

weight INTEGER NOT NULL

);


-- Team
CREATE TABLE teams (

team_id INTEGER PRIMARY KEY,

name TEXT NOT NULL,

country_name TEXT NOT NULL,

manager_name TEXT NOT NULL

);

```

The tables are built based on the models generated from Pydantic. Then I created function insert the data into my supabase tables. 

``` python

def insert_match_events(df, supabase):
    
    # Create a list of verified dictionaries
    events = [
        
        MatchEvent(**x).dict()
        for x in df.to_dict(orient = 'records')
    ]
    
    execution = supabase.table('match_event').upsert(events).execute() # insert any new rows, and update any old rows with the same primary key

# Initialize connection to supabase using the project url and api_key. 
supabase = create_client(project_url, api_key)

# Insert match events
insert_match_events(df, supabase)

```

#### For players:

``` python
team_info = []

team_info.append({
    'team_id': matchdict['home']['teamId'],
    'name': matchdict['home']['name'],
    'country_name': matchdict['home']['countryName'],
    'manager_name': matchdict['home']['managerName'],
    'players': matchdict['home']['players'],
})

team_info.append({
    'team_id': matchdict['away']['teamId'],
    'name': matchdict['away']['name'],
    'country_name': matchdict['away']['countryName'],
    'manager_name': matchdict['away']['managerName'],
    'players': matchdict['away']['players'],
})


def insert_players(team_info, supabase):
    players = []
    
    for team in team_info:
        for player in team['players']:
            players.append({
                'player_id': player['playerId'],
                'team_id': team['team_id'],
                'shirt_no': player['shirtNo'],
                'name': player['name'],
                'position': player['position'],
                'age': player['age'],
                'height': player['height'],
                'weight': player['weight']
            })
    execution = supabase.table('players').upsert(players).execute()

insert_players(team_info, supabase) 

```


___

# Automating the Pipeline  <a name="data-automation"></a>

#### Create a function to automate the entire process.

Now that I have the scraper working, the data models created, the tables created, and the functions to insert data into the tables. Now I need to tie all of this together into one pipeline function

```python

def scrape_match_events(whoscored_url, driver):
    
    # Visit the provided URL using the Selenium WebDriver
    driver.get(whoscored_url)
    
    # Parse the HTML of the loaded page using BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # Extract the script tag containing matchCentreData using BeautifulSoup
    element = soup.select_one('script:-soup-contains("matchCentreData")')
    
    # Extract the match data as a dictionary from the script element
    matchdict = json.loads(element.text.split("matchCentreData: ")[1].split(',\n')[0])
    
    # Extract the events from the match data dictionary
    match_events = matchdict['events']
    
    # Convert the events data into a Pandas DataFrame
    df = pd.DataFrame(match_events)
    
    # Drop rows with missing playerId
    df.dropna(subset='playerId', inplace=True)
    
    # Replace NaN values with None
    df = df.where(pd.notnull(df), None)
    
    # Rename columns to match the Pydantic model
    df = df.rename(
        {
        'eventId': 'event_id',
        'expandedMinute': 'expanded_minute',
        'outcomeType': 'outcome_type',
        'isTouch': 'is_touch',
        'playerId': 'player_id',
        'teamId': 'team_id',
        'endX': 'end_x',
        'endY': 'end_y',
        'blockedX': 'blocked_x',
        'blockedY': 'blocked_y',
        'goalMouthZ': 'goal_mouth_z',
        'goalMouthY': 'goal_mouth_y',
        'isShot': 'is_shot',
        'cardType': 'card_type',
        'isGoal': 'is_goal'
    },
        axis=1
    )
    
    # Extract additional information from nested dictionaries and drop unnecessary columns
    df['period_display_name'] = df['period'].apply(lambda x: x['displayName'])
    df['type_display_name'] = df['type'].apply(lambda x: x['displayName'])
    df['outcome_type_display_name'] = df['outcome_type'].apply(lambda x: x['displayName'])
    df.drop(columns=["period", "type", "outcome_type"], inplace=True)
    
    # Fill missing columns and values
    if 'is_goal' not in df.columns:
        df['is_goal'] = False
        
    if 'is_card' not in df.columns:
        df['is_card'] = False
        df['card_type'] = False
        
    # Filter out events with type_display_name 'OffsideGiven'
    df = df[~(df['type_display_name'] == "OffsideGiven")]
    
    # Reorder columns
    df = df[[
        'id', 'event_id', 'minute', 'second', 'team_id', 'player_id', 'x', 'y', 'end_x', 'end_y',
        'qualifiers', 'is_touch', 'blocked_x', 'blocked_y', 'goal_mouth_z', 'goal_mouth_y', 'is_shot',
        'card_type', 'is_goal', 'type_display_name', 'outcome_type_display_name',
        'period_display_name'
    ]]
    
    # Convert data types of selected columns
    df[['id', 'event_id', 'minute', 'team_id', 'player_id']] = df[['id', 'event_id', 'minute', 'team_id', 'player_id']].astype(np.int64)
    df[['second', 'x', 'y', 'end_x', 'end_y']] = df[['second', 'x', 'y', 'end_x', 'end_y']].astype(float)
    df[['is_shot', 'is_goal', 'card_type']] = df[['is_shot', 'is_goal', 'card_type']].astype(bool)
    
    # Fill NaN values with None for columns of float data type
    for column in df.columns:
        if df[column].dtype == np.float64 or df[column].dtype == np.float32:
            df[column] = np.where(
                np.isnan(df[column]),
                None,
                df[column]
            )
    
    # Insert match events data into the database
    insert_match_events(df, supabase)
    
    # Extract team information and insert players data into the database
    team_info = []
    team_info.append({
        'team_id': matchdict['home']['teamId'],
        'name': matchdict['home']['name'],
        'country_name': matchdict['home']['countryName'],
        'manager_name': matchdict['home']['managerName'],
        'players': matchdict['home']['players'],
    })

    team_info.append({
        'team_id': matchdict['away']['teamId'],
        'name': matchdict['away']['name'],
        'country_name': matchdict['away']['countryName'],
        'manager_name': matchdict['away']['managerName'],
        'players': matchdict['away']['players'],
    })
    
    insert_players(team_info, supabase)
    
    insert_teams(team_info, supabase)
    
    
    # Return a success message
    return print('Success')

```
#### Scrape for all games for one team. 

The final part of this process is scraping data relevant to the team I want. I'm going to scrape for Arsenal. So similarly, I'll create a beautiful soup instance, and use the `.select` functionality to to find all elements where an anchor tag `<a>` has a substring called live. The specifies that we are looking for hyperlinks. We are looking for this hyperlink for every game on the Arsenal page. 

``` python
# Initiate new driver
driver.get('https://www.whoscored.com/Teams/13/Fixtures/England-Arsenal')

# Create Soup Instance
soup = BeautifulSoup(driver.page_source, 'html.parser')

# Extract urls
all_urls = soup.select('a[href*= "\/Live\/"]')

# Add website url with sub hyper link

# List comprehension
all_urls = list(set([
    
    'https://www.whoscored.com' + x.attrs['href']
    
    for x in all_urls
])

```

Lastly, I loop through each url in the list, print the url name, and run the scrape_match_events script. With a sleep passed in to slow the scrape down. 

``` python
for url in all_urls:
    print(url)
    scrape_match_events(
        whoscored_url = url,
        driver = driver
    )
    time.sleep(2)
```

# Discussion <a name="discussion"></a>

Here is my database! I'll be running this script regularly manually, but next steps can be how to update this automatically or set a schedule run time. That's another depth that I'm not quite ready to look at yet. But that is an area of extension that I'm interested in learning.

!("img/posts/Saka.jpeg")

