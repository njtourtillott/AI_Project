from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
from sklearn import model_selection, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
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
    companies = companies.dropna(subset=["Twitter", "Scope.1..metric.tonnes.CO2e."])
    twitter_emission = companies.set_index("Twitter").to_dict()["Scope.1..metric.tonnes.CO2e."]
    tweets["emission"] = tweets["user"].map(twitter_emission)
    tweets["content"] = tweets["content"].apply(preprocess_tweet)
    corpus = tweets[["user", "content", "emission"]]
    corpus = corpus.dropna(subset=["emission"])
    return corpus


def generate_sets(test_size):
    corpus = preprocess_corpus()
    train_set, test_set, train_label, test_label = model_selection.train_test_split(corpus["content"],
                                                                                    corpus["emission"], test_size=test_size)
    tfidf_vect = TfidfVectorizer()
    tfidf_vect.fit(corpus["content"])
    train_set_tfidf = tfidf_vect.transform(train_set)
    test_set_tfidf = tfidf_vect.transform(test_set)
    return train_set_tfidf, test_set_tfidf, train_label, test_label


def develop_model(n, train_set, test_set, train_label, test_label):
    dr = TruncatedSVD(n_components=n)

    train_set = dr.fit_transform(train_set)
    test_set = dr.transform(test_set)

    rm = linear_model.LinearRegression()
    rm.fit(train_set, train_label)
    train_predictions = rm.predict(train_set)
    test_predictions = rm.predict(test_set)

    train_score = rm.score(train_set, train_label)
    test_score = rm.score(test_set, test_label)

    return train_score, test_score


def main():
    train_set, test_set, train_label, test_label = generate_sets(0.2)
    train, test = develop_model(4000, train_set, test_set, train_label, test_label)
    print(train)
    print(test)


if __name__ == '__main__':
    main()
