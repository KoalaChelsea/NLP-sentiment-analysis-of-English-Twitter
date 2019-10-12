from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer
from nltk.stem import PorterStemmer
from nltk.sentiment.util import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from nltk.tokenize import TweetTokenizer
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer

stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")


# clean tweet to help with statistics
def processTweet(tweet):
    # Emoji patterns
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)

    # Remove url
    tweet = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', '', tweet)
    # Remove hashtags
    # only removing the hash # sign from the word, we believe hashtags contains useful information
    tweet = re.sub(r'#', '', tweet)
    # Remove HTML special entities (e.g. amp;)
    tweet = re.sub(r'\\w*;', '', tweet)
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # convert @username to AT_USER
    tweet = re.sub('@[^\s]+', '', tweet)
    # remove mentions
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)
    # replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+', ' ', tweet)
    # remove emojis from tweet
    tweet = emoji_pattern.sub(r'', tweet)

    return tweet

# Preprocess for model
def pre_process_for_model(file, output):
    # load input data line by line with utf8

    with open(file, encoding='utf-8') as f:
        lines = f.readlines()
        # If there is no data in the file, don't go any further.
        if not len(lines):
            exit

    # get a dictionary of each tweet by regex
    tweets_dict = {"tweet_id": [], "tweet_sentiment": [], "tweet_content": []}
    for line in lines:
        matches = re.compile(r'(^\d*)(\s)([^\s]+)(\s)(.*)')
        tweets_dict["tweet_id"].append(matches.match(line).group(1))
        tweets_dict["tweet_sentiment"].append(matches.match(line).group(3))
        tweets_dict["tweet_content"].append(matches.match(line).group(5))

    # create a data frame of tweet
    tweets_df = pd.DataFrame(tweets_dict)

    # clean dataframe text column
    tweets_df['clean_text'] = tweets_df['tweet_content'].apply(processTweet)
    tweets_df['token'] = tweets_df['clean_text'].apply(TweetTokenizer().tokenize) \
        .apply(lambda x: [item for item in x if item.isalpha()])
    tweets_df['token count'] = tweets_df['token'].apply(len)

    # after remove stopwords and stemmer
    stop = stopwords.words('english')
    porter_stemmer = PorterStemmer()
    tweets_df['token deep clean'] = tweets_df['token'].apply(lambda x: [item for item in x if item not in stop]) \
        .apply(lambda x: [porter_stemmer.stem(item) for item in x])

    tweets_df.to_csv(output, index=None, header=True, encoding='utf-8')


# Check if a string has a number
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


# Normalize text data
def stem_stop(text, stem=False):
    # Remove link,user and special characters
    text = str(text).lower()
    # text=nltk.word_tokenize(text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = []

    for token in text.split():

        if wordnet.synsets(token) and hasNumbers(token) == False:
            if token not in stop_words:
                if stem:

                    tokens.append(stemmer.stem(token))
                else:
                    tokens.append(token)
    return " ".join(tokens)


# Navie Bayes
def NB(df_train, df_dev):
    # Feature extraction
    # n=1200
    df_train['clean_text'] = df_train['clean_text'].apply(lambda x: stem_stop(x))
    df_dev['clean_text'] = df_dev['clean_text'].apply(lambda x: stem_stop(x))

    df_pos_train = df_train[df_train['tweet_sentiment'] == 'positive']
    # df_pos_train= df_pos_train.sample(n=n, random_state=1)
    pos_tweets = df_pos_train['clean_text'].tolist()

    df_neg_train = df_train[df_train['tweet_sentiment'] == 'negative']
    # df_neg_train= df_neg_train.sample(n=n, random_state=1)
    neg_tweets = df_neg_train['clean_text'].tolist()

    df_neutral_train = df_train[df_train['tweet_sentiment'] == 'neutral']
    # df_neutral_train= df_neutral_train.sample(n=n, random_state=1)
    neutral_tweets = df_neutral_train['clean_text'].tolist()

    positive_featuresets = [(features(tweet), 'positive') for tweet in pos_tweets]
    negative_featuresets = [(features(tweet), 'negative') for tweet in neg_tweets]
    neutral_featuresets = [(features(tweet), 'neutral') for tweet in neutral_tweets]
    training_features = positive_featuresets + negative_featuresets + neutral_featuresets
    ngram_vectorizer = CountVectorizer(analyzer='word', binary=True, lowercase=False, ngram_range=(1, 2))

    # train the model
    sentiment_analyzer = SentimentAnalyzer()
    trainer = NaiveBayesClassifier.train

    classifier = sentiment_analyzer.train(trainer, training_features)
    truth_list = list(df_dev[['clean_text', 'tweet_sentiment']].itertuples(index=False, name=None))

    # test the model
    for i, (text, expected) in enumerate(truth_list):
        text_feats = features(text)
        truth_list[i] = (text_feats, expected)
    re = sentiment_analyzer.evaluate(truth_list, classifier)
    print(re)
    return classifier


def features(sentence):
    words = sentence.lower().split()
    return dict(('contains(%s)' % w, True) for w in words)


# Logistic regression
def Logreg(df_train, df_dev):
    df_train['clean_text'] = df_train['clean_text'].apply(lambda x: stem_stop(x))
    df_dev['clean_text'] = df_dev['clean_text'].apply(lambda x: stem_stop(x))
    datat = df_train[['clean_text', 'tweet_sentiment']]

    ngram_vectorizer = CountVectorizer(analyzer='word', binary=True, lowercase=False, ngram_range=(1, 2))

    y_train = datat['tweet_sentiment']
    # log_model = LogisticRegression()
    # log_model = log_model.fit(X=X_train, y=y_train)
    datatest = df_dev[['clean_text', 'tweet_sentiment']]

    X_train = ngram_vectorizer.fit_transform(df_train['clean_text'])
    X_test = ngram_vectorizer.transform(df_dev['clean_text'])

    y_test = datatest['tweet_sentiment']
    # find the best parameter: cf
    cf = 0
    accutemp = 0
    for c in [0.01, 0.05, 0.25, 0.3, 0.5, 1]:
        lr = LogisticRegression(multi_class='multinomial', solver='newton-cg', C=c)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        accu = accuracy_score(y_test, y_pred)
        if accu > accutemp:
            cf = c
    lr = LogisticRegression(multi_class='multinomial', solver='newton-cg', C=cf)
    lr.fit(X_train, y_train)
    print('Accuracy of Logistic regression is ', accuracy_score(y_test, y_pred))
    f1 = f1_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    precision = precision_score(y_test, y_pred, average=None)
    print('the f1 scores for negative, netural, postive are : ', f1)
    print('the recall for negative, netural, postive are : ', recall)
    print('the precision for negative, netural, postive are : ', precision)
    return lr


def main():
    # Processing train and test dataset to csv
    filelist = ['devtest.txt', 'dev.txt', 'train.txt', 'test.txt']
    outputlist = ['devtest.csv', 'dev.csv', 'train.csv', 'test.csv']
    for i in range(len(filelist)):
        output = 'data/Gold/' + outputlist[i]
        file = 'data/Gold/' + filelist[i]
        pre_process_for_model(file, output)

    # load data
    train = pd.read_csv("data/Gold/train.csv")
    dev = pd.read_csv("data/Gold/dev.csv")
    test = pd.read_csv("data/Gold/test.csv")
    devtest = pd.read_csv("data/Gold/devtest.csv")

    # combine train and validation for get larger train dataset
    frames = [train, dev]
    train = pd.concat(frames)

    # remove highly frequent data
    freq = pd.Series(' '.join(train['clean_text']).split()).value_counts()[:10]
    freq = list(freq.index)
    train['clean_text'] = train['clean_text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
    train['clean_text'].head()

    # remove rarely frequent data
    freq = pd.Series(' '.join(train['clean_text']).split()).value_counts()[-10:]
    freq = list(freq.index)
    train['clean_text'] = train['clean_text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
    train['clean_text'].head()

    # train and validate to get optimal model
    NB(train, devtest)
    Logreg(train, devtest)

    # based results from navie bayes and logistic regression, the result from logistic regression
    Logre = Logreg(train, test)

    # used logstic regression for classify input data
    ifile = 'data/Dev/INPUT.txt'
    output = 'data/Dev/INPUT.csv'
    pre_process_for_model(ifile, output)
    inputdata = pd.read_csv("data/Dev/INPUT.csv")

    inputdata['clean_text'] = inputdata['clean_text'].apply(lambda x: stem_stop(x))

    ngram_vectorizer = CountVectorizer(analyzer='word', binary=True, lowercase=False, ngram_range=(1, 2))
    X_train = ngram_vectorizer.fit_transform(train['clean_text'])
    X_test = ngram_vectorizer.transform(inputdata['clean_text'])

    y_pred = Logre.predict(X_test)
    # List1
    ID = inputdata['tweet_id']

    # List2
    label = y_pred

    # get the list of tuples from two lists.
    # and merge them by using zip().
    list_of_tuples = list(zip(ID, label))

    # Assign data to tuples.
    list_of_tuples

    # Converting lists of tuples into
    # pandas Dataframe.
    df = pd.DataFrame(list_of_tuples, columns=['ID', 'label'])
    df.to_csv("data/Dev/Output.csv")


if __name__ == '__main__':
    main()


