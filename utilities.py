from os import path
import numpy as np
import pandas as pd

# text preprocessing (Feature Engineering)
from sklearn.feature_extraction.text import TfidfVectorizer

# Feature extraction
import re
import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# saving and loading trained models
import pickle

nltk.download([
    'stopwords',
    'punkt',
    'wordnet',
    'omw-1.4',
    'vader_lexicon'
])

MODEL_PATH = 'models'


def save_model(model, filename):
    try:
        pickle.dump(model, open(f'{filename}', 'wb'))
        print('Saved')
    except Exception as err:
        print(err)


def load_model(filename):
    try:
        # print(f'Model path: .....................{filename}')
        model = pickle.load(open(f'{filename}', 'rb'))
        return model
    except Exception as err:
        print(err)
        return None


def remove_links(review):
    coma_split = review.split(',')

    review_split = []
    for sentence in coma_split:
        review_split.extend(sentence.split(' '))

    for word in review_split:
        if word.lower().startswith('http') or word.lower().startswith('https'):
            review_split.remove(word)
    else:
        review = ' '.join(review_split)

    return review


def preprocess_review(review):
    # Remove links from review
    review = remove_links(review)

    # Tokenize review
    tokenized_review = nltk.tokenize.RegexpTokenizer(
        '[a-zA-Z0-9\']+').tokenize(review)

    # Remove stopwords
    eng_stopwords = list(ENGLISH_STOP_WORDS)
    cleaned_tokens = [word.lower()
                      for word in tokenized_review if word not in eng_stopwords]

    # Stemming
    stemmer = nltk.stem.PorterStemmer()
    stemmed_review = [stemmer.stem(word) for word in cleaned_tokens]

    return ' '.join(stemmed_review)


def get_sentiment_score(review):
    sentAnalyzer = SentimentIntensityAnalyzer()
    scores = sentAnalyzer.polarity_scores(review)
    return scores['compound']


def get_sentiment_polarity(score):
    if score > 0.25:
        # positive
        return 1
    elif score < -0.25:
        # negative
        return -1
    else:
        # neutral
        return 0


def encode_tfidf_single(text_data, max_word_size=200):
    # Generating Corpus
    corpus_text = pd.Series(data=text_data, dtype='U')

    # loading trained encoder
    tfidf = load_model(filename='models/tfidf_encoding_model.pickle')

    tfidf_encoded = tfidf.transform(corpus_text)

#     print(f'TfIdf-encoded data: {tfidf_encoded.shape}')

    return pd.DataFrame(data=tfidf_encoded.toarray())


def get_predicted_class(label):
    encoded = {1: "positive", 0: "neutral", -1: "negative"}
    return encoded[label]


def detect_anomaly(review, model_path='models'):
    preprocessed_review = preprocess_review(review=review)
    encoded_review = encode_tfidf_single(preprocessed_review)
    model = load_model(filename='models/anomaly_IForest.pickle')
    prediction = model.predict(encoded_review)[0]
    category = {1: "non-anomalous", -1: "anomalous"}
    return (prediction, category[prediction])


def predict_sentiment(review, model_type='rf'):
    score = get_sentiment_score(review=review)
    polarity = get_sentiment_polarity(score)
    category = get_predicted_class(polarity)
    confidence = np.round(abs(score) * 100, 2)
    anomalous = detect_anomaly(review=review)
    #     predicted = model.predict(encoded_review)[0]
    #     category = get_predicted_class(predicted)
    #     confidence = (model.predict_proba(encoded_review).max() * 100).round(2)

    # print(f'Sentiment predicted label: {polarity}')
    # print(f'Sentiment predicted category: {category}')
    # print(f'Sentiment Confidence: {confidence}%')
    # print(f'Anomalous Status: {anomalous}')

    return {
        'review-sentiment-label': polarity,
        'review-sentiment': category,
        'sentiment-confidence': f'{confidence}%',
        'anomaly-status': anomalous
    }


def predict_sentiment2(review, model_type='rf', model_path='models'):
    preprocessed_review = preprocess_review(review=review)
    encoded_review = encode_tfidf_single(preprocessed_review)
    model = None
    if model_type == 'rf':
        model = load_model(filename=f'{model_path}/rf.pickle')
    elif model_type == 'dt':
        model = load_model(filename=f'{model_path}/dtree.pickle')
    elif model_type == 'svc':
        model = load_model(filename=f'{model_path}/svc.pickle')
    elif model_type == 'nb':
        model = load_model(filename=f'{model_path}/nb.pickle')

    predicted = model.predict(encoded_review)[0]
    category = get_predicted_class(predicted)
    confidence = (model.predict_proba(encoded_review).max() * 100).round(2)

    print(f'Sentiment predicted label: {predicted}')
    print(f'Sentiment predicted category: {category}')
    print(f'Sentiment Confidence: {confidence}%')


def interpretResult(results):
    if results['anomaly-status'][1].startswith('non') and \
            results['review-sentiment'].startswith('neg'):

        return f'The review is <b>Negative (Confidence: \
            {results["sentiment-confidence"]})</b> but \
                it is <b><i>Not an Anomaly</i></b>.'

    elif results['anomaly-status'][1].startswith('non') and \
            results['review-sentiment'].startswith('pos'):

        return f'The review is <b>Positive (Confidence: \
            {results["sentiment-confidence"]})</b> and \
                it is <b><i>Not an Anomaly</i></b>.'

    elif results['anomaly-status'][1].startswith('non') and \
            results['review-sentiment'].startswith('neu'):

        return f'The review is <b>Neutral (Confidence: \
            {results["sentiment-confidence"]})</b> but \
                it is <b><i>Not an Anomaly</i></b>.'

    elif results['anomaly-status'][1].startswith('anom') and \
            results['review-sentiment'].startswith('neg'):

        return f'The review is <b>Negative (Confidence: \
            {results["sentiment-confidence"]})</b> and \
                also <b><i>An Anomaly</i></b>.'

    elif results['anomaly-status'][1].startswith('anom') and \
            results['review-sentiment'].startswith('pos'):

        return f'The review is <b>Positive (Confidence: \
            {results["sentiment-confidence"]})</b> but \
                it is <b><i>An Anomaly</i></b>.'

    elif results['anomaly-status'][1].startswith('anom') and \
            results['review-sentiment'].startswith('neu'):

        return f'The review is <b>Neutral (Confidence: \
            {results["sentiment-confidence"]})</b> and \
                it is <b><i>An Anomaly</i></b>.'

    else:
        return f'The review is <b>{results["review-sentiment"]} (Confidence: \
            {results["sentiment-confidence"]})</b> and \
                it is <b><i>{results["anomaly-status"][1]}</i></b>.'


# print(detect_anomaly('I am feeling good about this product. Happy!'))
