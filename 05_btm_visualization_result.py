import pandas as pd
import pickle
import itertools
import plotly.graph_objects as go
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from operator import itemgetter
from sentence_transformers import SentenceTransformer
from typing import List
from umap import UMAP
from plotly.subplots import make_subplots

# path init
after_preprocess_dataset_path = "./data/2023-05-17-15-30-06_after_preprocess_dataset_clean_english_only_new.csv"
topic_result_path = "./output/topic_result_2023-06-13-20-46-08_5iter_24t.txt"

model_path = "./models/btm_model_2023-06-13-20-46-08_5iter_24t.pkl"
topics_path = "./models/btm_model_2023-06-13-20-46-08_5iter_24t.pkl"

timestamp = '2023-06-13-20-46-08'
save_vis_barchart_path = f"./analysis/vis/topics_vis/{timestamp}_vis_barchart.html"
save_vis_documents_and_topics_path = f"./analysis/vis/topics_vis/{timestamp}_vis_documents_and_topics.html"

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

def corpus_embedding(tweets):
    """
    The embedding of each document is required for subsequent drawing and dimensionality reduction
    """
    print("courpus embedding...")
    
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(list(tweets["text"].values), show_progress_bar=False)
    
    return embeddings

def load_btm_model():
    print("loading btm model and model topics...")
    # Load the BTM model file
    f_model = open(model_path,'rb')
    biterm_model = pickle.load(f_model)
    
    f_topics = open(topics_path,'rb')
    biterm_model_topics = pickle.load(f_topics)
    
    return biterm_model, biterm_model_topics

"""
Gets the topic number for each document classification
"""
def get_topic_per_doc(tweets_btm):
    """
    Args:
        tweets_btm - Topic model text file after obtaining the topic classification results, organized as "document (Topic: 8)"
    Returns:
        topic_per_doc [List] - Gets the topic number for each document classification
    """
    topic_per_doc = [] # Gets the topic for each doc
    for item in tweets_btm:
        split_item = item.split(" ")
        for i in range(len(split_item)):
            if split_item[i] == "(topic:":
                topic_per_doc.append(int(split_item[i + 1].split(')')[0]))
    return topic_per_doc

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

# Visual documents and topics
def visualize_documents(docs: List[str],
                        docs_btm: List[str],
                        topic_top_word,
                        topics: List[int] = None,
                        embeddings: np.ndarray = None,
                        reduced_embeddings: np.ndarray = None,
                        sample: float = None,
                        hide_annotations: bool = False,
                        hide_document_hover: bool = False,
                        title: str = "<b>Documents and Topics</b>",
                        width: int = 1200,
                        height: int = 750):
    """
    Args:
        docs - Original corpus
        docs_btm - Topic model text file after obtaining the topic classification results, organized as "document (Topic: 8)"
        topic_top_word - The names of the top M words in probability
        topics - Topic number (or custom topic name)
        embeddings - The corpus code is used for dimensionality reduction
        reduced_embeddings - Coding after reduced dimension
        sample - Sample data to optimize visualization and dimensionality reduction
        hide_annotations - 
        hide_document_hover - 
        title - Graph Title
        width, height - 
        
    Returns:
        fig (go.Figure) - figure
    """

    topic_per_doc = get_topic_per_doc(docs_btm) # Gets the topic number for each document classification
    
    # Sample the data to optimize for visualization and dimensionality reduction
    if sample is None or sample > 1:
        sample = 1

    indices = []
    for topic in set(topic_per_doc):
        s = np.where(np.array(topic_per_doc) == topic)[0]
        size = len(s) if len(s) < 100 else int(len(s) * sample)
        indices.extend(np.random.choice(s, size=size, replace=False))
    indices = np.array(indices)

    df = pd.DataFrame({"topic": np.array(topic_per_doc)[indices]})
    df["doc"] = [docs[index] for index in indices]
    df["topic"] = [topic_per_doc[index] for index in indices]

    if sample is None:
            embeddings_to_reduce = embeddings
    else:
        embeddings_to_reduce = embeddings[indices]

    # Reduce input embeddings
    if reduced_embeddings is None: # Dimension reduction is used to display on a two-dimensional graph
        umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit(embeddings_to_reduce)
        embeddings_2d = umap_model.embedding_
    elif sample is not None and reduced_embeddings is not None:
        embeddings_2d = reduced_embeddings[indices]
    elif sample is None and reduced_embeddings is not None:
        embeddings_2d = reduced_embeddings

    unique_topics = set(topic_per_doc) # Pick out how many subject numbers there are (do not repeat)
    if topics is None:
        topics = unique_topics

    # Combine data
    df["x"] = embeddings_2d[:, 0]
    df["y"] = embeddings_2d[:, 1]

    # Prepare text and names
    # topic_model.get_topic(topic) - Retrieve the word frequency order table on the topic
    # Name the topic points
    names = [f"{topic}_" + "_".join([word for word, value in topic_top_word[topic]][:3]) for topic in unique_topics]
    
    # Visualize
    fig = go.Figure()

    # Outliers and non-selected topics
    non_selected_topics = set(unique_topics).difference(topics)
    if len(non_selected_topics) == 0:
        non_selected_topics = [-1]

    selection = df.loc[df.topic.isin(non_selected_topics), :]
    selection["text"] = ""
    selection.loc[len(selection), :] = [None, None, selection.x.mean(), selection.y.mean(), "Other documents"]

    fig.add_trace(
        go.Scattergl(
            x=selection.x,
            y=selection.y,
            hovertext=selection.doc if not hide_document_hover else None,
            hoverinfo="text",
            mode='markers+text',
            name="other",
            showlegend=False,
            marker=dict(color='#CFD8DC', size=5, opacity=0.5)
        )
    )

    # Selected topics
    for name, topic in zip(names, unique_topics):
        if topic in topics and topic != -1:
            selection = df.loc[df.topic == topic, :]
            selection["text"] = ""

            if not hide_annotations:
                selection.loc[len(selection), :] = [None, None, selection.x.mean(), selection.y.mean(), name]

            fig.add_trace(
                go.Scattergl(
                    x=selection.x,
                    y=selection.y,
                    hovertext=selection.doc if not hide_document_hover else None,
                    hoverinfo="text",
                    text=selection.text,
                    mode='markers+text',
                    name=name,
                    textfont=dict(
                        size=12,
                    ),
                    marker=dict(size=5, opacity=0.5)
                )
            )

    # Add grid in a 'plus' shape
    x_range = (df.x.min() - abs((df.x.min()) * .15), df.x.max() + abs((df.x.max()) * .15))
    y_range = (df.y.min() - abs((df.y.min()) * .15), df.y.max() + abs((df.y.max()) * .15))
    fig.add_shape(type="line",
                  x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1],
                  line=dict(color="#CFD8DC", width=2))
    fig.add_shape(type="line",
                  x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2,
                  line=dict(color="#9E9E9E", width=2))
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)

    # Stylize layout
    fig.update_layout(
        template="simple_white",
        title={
            'text': f"{title}",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width,
        height=height
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    
    # 保存成html有两种方式
    fig.write_html(save_vis_documents_and_topics_path)  # 直接通过fig.write_html

    return fig


# Visual bar chart
def visualize_barchart(docs_btm,
                       topic_top_word,
                       topics: List[int] = None,
                       n_words: int = 10,
                       title: str = "<b>Topic Word Scores</b>",
                       width: int = 250,
                       height: int = 250) -> go.Figure:
    
    """
    Args:
        docs_btm - Topic model text file after obtaining the topic classification results, organized as "document (Topic: 8)"
        topic_top_word - The names of the top M words in probability
        topics - Topic number (or custom topic name)
        n_words - Displays the probability of the first n_words
        title - Graph Title
        width, height - 
        
    Returns:
        fig (go.Figure) - figure
    """

    colors = itertools.cycle(["#D55E00", "#0072B2", "#CC79A7", "#E69F00", "#56B4E9", "#009E73", "#F0E442"])

    topic_per_doc = get_topic_per_doc(docs_btm) # Gets the topic number for each document classification
    topics = set(topic_per_doc)  # Pick out how many subject numbers there are (do not repeat)
    
    # Select topics based on top_n and topics args
    # freq_df = topic_model.get_topic_freq()
    # freq_df = freq_df.loc[freq_df.Topic != -1, :]
    # if topics is not None:
    #     topics = list(topics)
    # elif top_n_topics is not None:
    #     topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    # else:
    #     topics = sorted(freq_df.Topic.to_list()[0:6])

    # Initialize figure
    subplot_titles = [f"Topic {topic}" for topic in topics] # Set the title of each column subgraph
    
    columns = 4 
    rows = int(np.ceil(len(topics) / columns))
    fig = make_subplots(rows=rows,
                        cols=columns,
                        shared_xaxes=False,
                        horizontal_spacing=.1,
                        vertical_spacing=.4 / rows if rows > 1 else 0,
                        subplot_titles=subplot_titles)

    # Add barchart for each topic
    row = 1
    column = 1
    for topic in topics:
        # Gets the specific names and distribution probabilities of the first M words under each topic
        words = [word + "  " for word, _ in topic_top_word[topic]][:n_words][::-1]
        scores = [score for _, score in topic_top_word[topic]][:n_words][::-1]

        fig.add_trace(
            go.Bar(x=scores,
                   y=words,
                   orientation='h',
                   marker_color=next(colors)),
            row=row, col=column)

        if column == columns:
            column = 1
            row += 1
        else:
            column += 1

    # Stylize graph
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        title={
            'text': f"{title}",
            'x': .5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width*4,
        height=height*rows if rows > 1 else height * 1.3,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    
    fig.write_html(save_vis_barchart_path)
    
    return fig

    
if __name__ == "__main__":
    tweets, tweets_btm = read_corpus_and_result()
    biterm_model, biterm_model_topics = load_btm_model()
    
    topic_top_word = generate_topic_top_word(tweets, biterm_model, M = 10)
    embeddings = corpus_embedding(tweets)
    
    visualize_documents(tweets["text"].values, tweets_btm, topic_top_word, embeddings=embeddings)
    
    visualize_barchart(tweets_btm, topic_top_word, n_words=10)
    
    