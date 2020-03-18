

import sys
import csv
import tweepy
import matplotlib.pyplot as plt

from collections import Counter
from aylienapiclient import textapi



## Twitter credentials
consumer_key = 	'Del4H2xEpmeDEtOm4hn3hZObD'
consumer_secret = 'jE7zFuRdj9FvkSk1DLzpvR2MhmlczWKGyXnKsBvK95tJkN7IUG'

access_token = '825230799238742016-hI7EV1rmbKG01qaQUfkRhF93OpNpnsK'

access_token_secret = 	'JT9y9vc23IaRXOdZEi2b6JouBuCZmrIyUdMWwcToafwej'

## AYLIEN credentials
application_id = "89635eae"
application_key = "b216f6bd01aaebbf5f5331fcb8498114"

## set up an instance of Tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

## set up an instance of the AYLIEN Text API
client = textapi.Client(application_id, application_key)

## search Twitter for something that interests you
query = input("Enter ant tag you want to analyze? \n")
number = input("How many Tweets do you want to analyze? \n")

results = api.search(
   lang="en",
   q=query + " -rt",
   count=number,
   result_type="recent"
)

print("Gathered Tweets \n")

## open a csv file to store the Tweets and their sentiment
file_name = 'Myanalysis.csv'.format(number, query)

with open('Myanalysis.csv', 'w', newline='') as csvfile:
   csv_writer = csv.DictWriter(
       f=csvfile,
       fieldnames=["Tweet", "Sentiment"]
   )
   csv_writer.writeheader()

   print("--- Opened a CSV file to store the results of your sentiment analysis... \n")

## tidy up the Tweets and send each to the AYLIEN Text API
   for c, result in enumerate(results, start=1):
       tweet = result.text
       tidy_tweet = tweet.strip().encode('ascii', 'ignore')



       response = client.Sentiment({'text': tidy_tweet})
       csv_writer.writerow({
           'Tweet': response['text'],
           'Sentiment': response['polarity']
       })

       print("Analyzed Tweet {}".format(c))

## count the data in the Sentiment column of the CSV file
with open(file_name, 'r') as data:
   counter = Counter()
   for row in csv.DictReader(data):
       counter[row['Sentiment']] += 1

   positive = counter['positive']
   negative = counter['negative']
   neutral = counter['neutral']

## declare the variables for the pie chart, using the Counter variables for "sizes"
colors = ['green', 'red', 'grey']
sizes = [positive, negative, neutral]
labels = 'Positive', 'Negative', 'Neutral'

## use matplotlib to plot the chart
plt.pie(
   x=sizes,
   shadow=True,
   colors=colors,
   labels=labels,
   startangle=90
)

plt.title("Myanalysis.csv".format(number, query))
plt.show()
