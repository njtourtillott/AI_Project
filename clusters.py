import umap
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import string


def preprocess_tweet(tweet):
    stop_words = set(stopwords.words('english'))
    tweet.lower()
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in filtered_words]

    return " ".join(lemma_words)


def preprocess_corpus():
    companies = pd.read_csv("companies_and_social_media.csv")
    tweets = pd.read_csv("all_tweets.csv")
    companies = companies.dropna(subset=["Twitter"])
    twitter_score = companies.set_index("Twitter").to_dict()["Performance.Band"]
    tweets["score"] = tweets["user"].map(twitter_score)
    tweets["content"] = tweets["content"].apply(preprocess_tweet)
    corpus = tweets[["user", "content", "score"]]
    return corpus


def generate_set():
    corpus = preprocess_corpus()
    tfidf_vect = TfidfVectorizer()
    tfidf_vect.fit(corpus["content"])
    set = tfidf_vect.transform(corpus["content"])
    return set, corpus


def umap_cluster(set, corpus):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(set)
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette()[x] for x in corpus.score.map({"A": 0, "A-": 1, "B": 2, "C": 3, "D": 4, "E": 5})])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the Tweet dataset', fontsize=24)


def tsne_cluster(set, corpus):
    embedding = TSNE().fit_transform(set)
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette()[x] for x in corpus.score.map({"A": 0, "A-": 1, "B": 2, "C": 3, "D": 4, "E": 5})])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('TSNE projection of the Tweet dataset', fontsize=24)


def main():
    set, corpus = generate_set()
    plt.figure(1)
    umap_cluster(set, corpus)
    plt.figure(2)
    tsne_cluster(set, corpus)
    plt.show()


if __name__ == '__main__':
    main()
