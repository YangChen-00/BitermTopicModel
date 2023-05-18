import pandas as pd
import pickle
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from operator import itemgetter

# path init
after_preprocess_dataset_path = "./data/2023-05-17-15-30-06_after_preprocess_dataset_clean_english_only_new.csv"
topic_result_path = "./output/topic_result_2023-05-17-15-33-14_1iter.txt"

model_path = "./models/btm_model_2023-05-17-15-33-14_1iter.pkl"
topics_path = "./models/btm_topics_2023-05-17-15-33-14_1iter.pkl"

topic_result_path = "./output/topic_result_2023-05-17-15-33-14_1iter.txt"

timestamp = '2023-05-17-15-33-14'
save_topic_keywords_path = f'./output/{timestamp}_topic_keywords.csv'
save_tweets_by_topic_path = f'./output/{timestamp}_tweets_by_topic.csv'
save_n_tweets_in_topic = f'./output/{timestamp}_n_tweets_in_topic.csv'
save_keywords = f'./output/{timestamp}_keywords.csv'

# Read the CSV file
def read_corpus_and_result():
    """
        tweets - '.csv' organization of the original corpus
        tweets_btm - Topic model text file after obtaining the topic classification results, organized as "document (Topic: 8)"
    """
    print("reading corpus and results...")
    tweets = pd.read_csv(after_preprocess_dataset_path)
    tweets.rename(columns={"text_clean": "text"}, inplace=True) # Change the table header to 'text'
    
    tweets_btm = open(topic_result_path).read().splitlines()
    # tweets.head(5), tweets_btm[:5], tweets.shape, len(tweets_btm)
    
    return tweets, tweets_btm

def load_btm_model():
    print("loading btm model and model topics...")
    # Load the BTM model file
    f_model = open(model_path,'rb')
    biterm_model = pickle.load(f_model)
    
    f_topics = open(topics_path,'rb')
    biterm_model_topics = pickle.load(f_topics)
    
    return biterm_model, biterm_model_topics

# Find the topM words for each topic
def generate_topic_top_word(tweets, biterm_model, M = 10):
        """
        Args:
            topic_top_prob - The top probability list of the most likely words for each topic
            V [List] - A list of all the words
            M - Take the first M words
        Returns:
            topic_top_word Dict(List[Tuple()]) - The names of the top M words in probability
        """
        # vectorize texts
        vec = CountVectorizer(stop_words='english')
        tweets_list = [i for item in tweets.values for i in item]
        X = vec.fit_transform(tweets_list).toarray()

        # vocab - Get all words
        V = np.array([t for t, i in sorted(vec.vocabulary_.items(),
                                            key=itemgetter(1))])

        # The top probability list of the most likely words for each topic
        topic_top_prob = biterm_model.phi_wz # phi_wz: [word, topic] Word distribution probability on each topic
    
        topic_top_word = dict()
        for z, P_wzi in enumerate(topic_top_prob.T): 
            """
                z - z-th topic
                P_wzi - The probability distribution of all words on the z-th topic
            """
            topic_top_word[z] = [] # Each topic consists of multiple tuples (word, prob)
            V_z_prob = np.sort(P_wzi)[:-(M + 1):-1] # Sort the probability distribution
            V_z = np.argsort(P_wzi)[:-(M + 1):-1] # Sort the probability distribution and find the index of the top words
            W_z = V[V_z] # Find the name of the word at the top of the list
            for prob, word in zip(V_z_prob, W_z): # Form the innermost tuple, meaning tuple(word, prob).
                topic_top_word[z].append((word, prob))
        return topic_top_word
    
def save_topic_keywords_to_csv(tweets, biterm_model):
    print("saving topic keywords to csv...")
    # top-M words for each topic
    topic_top_word = generate_topic_top_word(tweets, biterm_model, 20)

    topic_all_words = []
    for i in range(20):
        print_str = ""
        # print_str = f"topic {i}"
        for j in range(len(topic_top_word[i])):
            print_str += f"{topic_top_word[i][j][0]},"
        topic_all_words.append(print_str)
        print(f"topic {i}: {print_str}")
        
    topic_all_words_df = pd.DataFrame(np.array(topic_all_words).reshape(-1, 1), columns=["keywords"])
    topic_all_words_df.to_csv(save_topic_keywords_path, sep=',', index=False,header=True)
        
    return topic_all_words


def save_tweets_by_topic_to_csv(tweets, topic_all_words):
    print("saving tweets by topic to csv...")
    tweets_by_topic_df = pd.DataFrame()
    
    reader = open(topic_result_path, "r")
    i = 0
    for line in reader.readlines():
        if i % 10000 == 0:
            print(f"finished {i}/{tweets.shape[0]}")
        row = []

        split_line = line.split("(")
        doc = split_line[0]
        topic_id = int(split_line[1].split(" ")[1].split(')')[0])
        row.append(f"topic{topic_id}")
        row.append(topic_all_words[topic_id])
        row.append(doc)
        temp_df = pd.DataFrame([row], columns=['topic', 'keywords', 'tweet'])
        tweets_by_topic_df = pd.concat([tweets_by_topic_df, temp_df], ignore_index=True)
        i += 1

    # sort dataframe by 'topic'
    df_mapping = pd.DataFrame({
        'size': ['topic{}'.format(i) for i in range(20)],
    })
    sort_mapping = df_mapping.reset_index().set_index('size')

    tweets_by_topic_df['topic_num'] = tweets_by_topic_df['topic'].map(sort_mapping['index'])

    tweets_by_topic_df = tweets_by_topic_df.sort_values('topic_num').drop('topic_num', axis=1)
    print(tweets_by_topic_df)

    tweets_by_topic_df.to_csv(save_tweets_by_topic_path, sep=',', index=False,header=True)
    
    return tweets_by_topic_df
    
def save_n_tweets_in_topic_to_csv(tweets_by_topic_df):
    print("saving n tweets in topic to csv...")
    
    # statistic the number of every topic (statistic n_tweets_in_topic)
    total_topic_num = int(tweets_by_topic_df.iloc[-1]['topic'].split('c')[1]) + 1
    n_tweets_in_topic = pd.DataFrame(np.zeros(total_topic_num).reshape(1, -1))
    n_tweets_in_topic.columns = ['topic{}'.format(i) for i in range(20)]

    for i in range(len(tweets_by_topic_df)):
        topic_i = tweets_by_topic_df.iloc[i]['topic']
        n_tweets_in_topic[topic_i] += 1
    print(n_tweets_in_topic)

    # save_n_tweets_in_topic
    n_tweets_in_topic.to_csv(save_n_tweets_in_topic, sep=',', index=False,header=True)

def save_keywords_of_every_topic_to_csv(tweets, biterm_model, tweets_by_topic_df):
    print("saving keywords of every topic to csv...")
    
    # statistic the keywords of every topic
    total_topic_num = int(tweets_by_topic_df.iloc[-1]['topic'].split('c')[1]) + 1

    columns_df = ['Keyword_number']
    columns_df.extend(['topic{}'.format(i) for i in range(20)])
    keywords_df = pd.DataFrame(columns=columns_df, index=np.arange(1, total_topic_num+1))
    keywords_df['Keyword_number'] = np.arange(1, total_topic_num + 1)

    # top-M words for each topic
    topic_top_word = generate_topic_top_word(tweets, biterm_model, 20)
    
    for i in range(total_topic_num):
        topic_str = 'topic' + str(i)
        keywords_num = len(topic_top_word[i])
        keywords_list = []
        for j in range(keywords_num):
            keywords_list.append(topic_top_word[i][j][0])
        keywords_df[topic_str] = keywords_list
        
    print(keywords_df)
    
    # save_n_tweets_in_topic
    keywords_df.to_csv(save_keywords, sep=',', index=False,header=True)

if __name__ == "__main__":
    tweets, tweets_btm = read_corpus_and_result()
    biterm_model, biterm_model_topics = load_btm_model()
    
    topic_all_words = save_topic_keywords_to_csv(tweets, biterm_model)
    
    tweets_by_topic_df = save_tweets_by_topic_to_csv(tweets, topic_all_words)
    
    save_n_tweets_in_topic_to_csv(tweets_by_topic_df)
    
    save_keywords_of_every_topic_to_csv(tweets, biterm_model, tweets_by_topic_df)
    
    