import pandas as pd
import numpy as np
import spacy
from spacy.tokens import Token
from spacy.tokenizer import Tokenizer
from spacy.matcher import Matcher
from spacy.lang.tokenizer_exceptions import URL_PATTERN
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
import re
import string
from nltk.tokenize import TweetTokenizer
from nltk.corpus import words
import itertools

# clean tweet to help with statistics
def processTweet(tweet):
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # convert @username to AT_USER
    tweet = re.sub('@[^\s]+', '', tweet)
    # Remove url
    tweet = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', '', tweet)
    # Remove hashtags
    # only removing the hash # sign from the word, we believe hashtags contains useful information
    tweet = re.sub(r'#', '', tweet)
    # Remove HTML special entities (e.g. &amp;)
    tweet = re.sub(r'\&\w*;', '', tweet)
    # Remove Punctuation with a space for filter
    tweet = re.sub(r'[' + string.punctuation.replace('@', '') + ']+', ' ', tweet)
    # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
    tweet = ''.join(c for c in tweet if c <= '\uFFFF')
    return tweet


def main():

    # load input data line by line with utf8
    with open("data/Dev/INPUT.txt", encoding="utf8") as f:
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

    # report the number of tweet by checking the unique tweet id (no duplicated id found)
    print("The total number of tweets is %s" % len(tweets_df.tweet_id.unique()))

    # write all tweets into one text
    tweets_list = tweets_df['tweet_content'].tolist()

    # sum up the length of each line
    num_of_character = sum(len(list) for list in tweets_list)
    print("The total number of character in all tweets %s " % num_of_character)

    # clean dataframe text column
    tweets_df['clean_text'] = tweets_df['tweet_content'].apply(processTweet)
    tweets_df['token'] = tweets_df['clean_text'].apply(TweetTokenizer().tokenize)
    tweets_df['token count'] = tweets_df['token'].apply(len)
    tweets_df.to_csv(r'data/tweets_input.csv', index=None, header=True)


    #tokens_all = []
    #tokens_all.append(row for row in tweets_df['token'])
    # tokens_all_list = list(itertools.chain.from_iterable(tokens_all))
    #print(tokens_all)

    #vocab = sorted(set(tokens_all))
    #print(vocab)
    #num_of_distinct_words = sum((int(token in words.words() == True)) for token in tokens_all)
    #print("The total number of distinct words in all tweets %s " % num_of_distinct_words)

    #words_all_df = pd.DataFrame(words_all)
    #words_all_df.to_csv(r'data/words.csv', index=None, header=True)

    # write all tweets into one text
    #tweets_text = '\n'.join(tweets_list)

    # spacy text
    #doc_tweets = nlp(tweets_text)

    # report length of tokens in all tweets
    #print("The total number of words in all tweets %s " % len(doc_tweets))


if __name__ == '__main__':
    main()
