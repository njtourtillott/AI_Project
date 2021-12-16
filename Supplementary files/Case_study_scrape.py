import os
import pandas as pd

company_names = ['Hersheys', 'DominionEnergy', 'HP', 'amazon', 'netflix',
                 'TysonFoods', 'MarathonPetroCo', 'WasteManagement']

# Create an empty dataframe to store all tweets
all_tweets = pd.DataFrame(columns=['user', 'date', 'content'])

# Using OS library to call commands in Python
for name in company_names:

    # Create a command to get X tweets from the username given
    command = "snscrape --jsonl --max-results 150 --since 2011-01-01 twitter-search 'from:" + name + "'> " + name + ".json"
    os.system(command)

    # Reads the json generated from the CLI command above and creates a pandas dataframe
    file_name = name + ".json"
    company_tweets = pd.read_json(file_name, lines=True)

    # Get the date and the content of the tweet
    company_tweets = company_tweets[['content', 'date']]

    # Add a column with the name of the company and reorder columns
    company_tweets['user'] = name
    company_tweets = company_tweets[['user', 'date', 'content']]
    company_tweets = pd.DataFrame(company_tweets)

    # Append all the tweets to the main dataframe
    all_tweets = all_tweets.append(company_tweets, ignore_index=True)

all_tweets.to_csv('case_study_tweets.csv', index=False)



