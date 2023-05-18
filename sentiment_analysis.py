import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def nltkSentiment(sentence):
    sid = SentimentIntensityAnalyzer()
    senti = sid.polarity_scores(sentence)  
    return senti['neg'], senti['pos']

tweets_by_topic_csv_path = "./output/2023-05-17-15-33-14_tweets_by_topic.csv"
data = pd.read_csv(tweets_by_topic_csv_path)

num_topic = 20

import numpy as np
total_neg = np.zeros(num_topic)
total_pos = np.zeros(num_topic)

i = 0
for index, row in data.iterrows():
    if i % 10000 == 0:
        print(f"finished {i}/{data.shape[0]}")
    topic_id_str = row["topic"]
    if len(topic_id_str) == 6:
        topic_id = int(topic_id_str[-1])
    else:
        topic_id = int(topic_id_str[-2:])
        
    sentence = row['tweet']
    
    neg, pos = nltkSentiment(sentence)
    if neg > pos:
        total_neg[topic_id] += 1
    else:
        total_pos[topic_id] += 1    
    i += 1
print(f"total_neg: {total_neg} \ntotal_pos: {total_pos}")

ratio = total_pos / (total_pos + total_neg)
print(f"positive ratio: {ratio}")

sentiment_df = pd.DataFrame({'negative': total_neg, 'positive': total_pos, 'sentiment_ratio': ratio})
sentiment_df

save_sentiment_result_path = 'output/2023-05-17-15-33-14_sentiment_result.csv'
sentiment_df.to_csv(save_sentiment_result_path, sep=',', index=False,header=True)