from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import re
from sklearn import model_selection, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
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


def develop_model():
    corpus = preprocess_corpus()
    train_set, test_set, train_label, test_label = model_selection.train_test_split(corpus["content"], corpus["score"], test_size=0.2)

    encoder = LabelEncoder()
    encoder.fit(corpus["score"])
    test_label = encoder.transform(test_label)
    train_label = encoder.transform(train_label)

    tfidf_vect = TfidfVectorizer()
    tfidf_vect.fit(corpus["content"])
    train_set_tfidf = tfidf_vect.transform(train_set)
    test_set_tfidf = tfidf_vect.transform(test_set)

    clf = svm.SVC()
    clf.fit(train_set_tfidf, train_label)
    train_predictions = clf.predict(train_set_tfidf)
    test_predictions = clf.predict(test_set_tfidf)

    print(accuracy_score(train_predictions, train_label))
    print(accuracy_score(test_predictions, test_label))

    d = {"content": train_set, "prediction": train_label}
    f = {"content": test_set, "prediction": test_label}
    df = pd.DataFrame(data=d)
    df2 = pd.DataFrame(data=f)
    df = df.append(df2, ignore_index=True)
    tweets_summary = corpus.join(df.set_index("content"), on="content")

    tweets_summary["score"] = encoder.transform(tweets_summary["score"].tolist())
    companies = np.unique(tweets_summary["user"])
    companies_summary = pd.DataFrame(columns=["user", "score", "prediction"])
    for entry in companies:
        company = tweets_summary[tweets_summary["user"] == entry]
        companies_summary = companies_summary.append({"user": entry, "score": company["score"].mode(), "prediction": company["prediction"].mode()}, ignore_index=True)
    print(accuracy_score(np.array(companies_summary["score"].tolist()), np.array(companies_summary["prediction"].tolist())))

    tweets_summary["score"] = encoder.inverse_transform(tweets_summary["score"].tolist())
    tweets_summary["prediction"] = encoder.inverse_transform(tweets_summary["prediction"].tolist())
    companies_summary["score"] = encoder.inverse_transform(companies_summary["score"].tolist())
    companies_summary["prediction"] = encoder.inverse_transform(companies_summary["prediction"].tolist())

    tweets_summary.to_csv("tweets_summary.csv")
    companies_summary.to_csv("companies_summary.csv")


def main():
    develop_model()


if __name__ == '__main__':
    main()
