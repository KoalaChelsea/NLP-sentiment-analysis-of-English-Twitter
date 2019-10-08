import pandas as pd
import numpy as np
import re
import string
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import itertools
import collections
from nltk.util import ngrams
import matplotlib.pyplot as plt


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


# Check if a string has a number
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


def count_ngrams_words(start, end, x): # start = 2, end = 6, x = tweets_df['clean_text']
    for i in range(start, end):
        l_ngrams = []
        for s in x:
            s = s.lower()
            s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
            tokens = [token for token in s.split(" ") if token != ""]
            l_ngrams = l_ngrams + list(ngrams(tokens, i))
            len_ngrams = len(collections.Counter(set(l_ngrams)))
        if i == 2:
            print("the total number of distinct bigrams of words that appear is: ", len_ngrams)
        elif i == 3:
            print("the total number of distinct trigrams of words that appear is: ", len_ngrams)
        else:
            print("the total number of distinct", i, "-grams of words that appear is: ", len_ngrams)


def count_ngrams_chars(start,end, x):
    for i in range(start, end):
        l_ngrams_char = []
        for s in x:
            s = s.lower()
            s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
            for q in x:
                l_ngrams_char = l_ngrams_char + [q[p:p+i] for p in range(len(q)-i+1)]
            len_ngrams_char  = len(collections.Counter(set(l_ngrams_char)))
        if i == 2:
            print("the total number of distinct bigrams of characters that appear is: ", len_ngrams_char)
        elif i == 3:
            print("the total number of distinct trigrams of characters that appear is: ", len_ngrams_char)
        else:
            print("the total number of distinct", i, "-grams of characters that appear is: ", len_ngrams_char)


def main():
    # load input data line by line with utf8
    with open('data/Dev/INPUT.txt', encoding='utf-8') as f:
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

    # write all cleaned tweets into one text
    tweets_list_cleaned = tweets_df['clean_text'].tolist()

    # Count number of characters and words in each tweet
    tweet_character_count = []
    tweet_word_count = []
    for tweet in tweets_list_cleaned:

        # Count # of words
        temp_tweet = tweet.lower()
        # drop anything that is not a word from the string
        temp_tweet = re.sub('[^a-zA-Z]+', ' ', temp_tweet)
        tweet_word_count.append(len(temp_tweet.split()))

        # count characters
        char_count = len(temp_tweet)
        tweet_character_count.append(char_count)

    # add character count and word count to master tweets_df
    tweets_df['Character_Count'] = tweet_character_count
    tweets_df['Word_Count'] = tweet_word_count

    # split/tokenize all words
    words_in_tweet = [tweet.lower().split() for tweet in tweets_list_cleaned]

    # flatten the list to one massive string
    all_words = [item for sublist in words_in_tweet for item in sublist]

    # remove all tokenized words with values
    for word in all_words:
        if hasNumbers(word):
            all_words.remove(word)

    # remove all tokenized words with punctuation
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in all_words]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]

    # Count the number of unique words appeared
    word_counts = collections.Counter(set(words))
    print("The total number of distinct words (vocabulary) is %s" % len(word_counts))

    print("The average number of characters per tweet is " +
          str(round(tweets_df['Character_Count'].mean(axis=0), 1)) +
          " and the average number of words per tweet is " +
          str(round(tweets_df['Word_Count'].mean(axis=0), 1)))

    # Count number of characters for each token for each tweet
    tweets_token_list = tweets_df['token'].tolist()
    token_character_count = []
    for tokens in tweets_token_list:
        count = 0
        for token in tokens:
            count += len(token)
        token_character_count.append(count)

    tweets_df['Token_Character_Count'] = token_character_count/tweets_df['token count']

    # Calculate Standard Deviation for each tweet's tokens
    token_character_average = tweets_df['Token_Character_Count'].mean(axis=0)
    tweets_df['Token_Character_SD'] = np.absolute(tweets_df['Token_Character_Count'] - token_character_average) ** 2

    # The average number and standard deviation of characters per token
    print("The average number of characters per token per tweet is " +
          str(round(token_character_average, 1)) +
          " and the average standard deviation per tweet's token is " +
          str(round(np.sqrt(tweets_df['Token_Character_SD'].mean(axis=0)), 1)))

    # Top 10 most frequent words (types) in the vocabulary
    top_10_words = collections.Counter(words).most_common(10)
    print("The total number of tokens corresponding to the top 10 most frequent words (types) in the vocabulary is ",
          top_10_words)

    # Number of Token / Number of Vocab
    print("The token/type ratio in the dataset is ", tweets_df['token count'].sum(axis=0) / len(word_counts))

    # Count the total number of distinct n-grams of words
    count_ngrams_words(2, 6, tweets_df['clean_text'])

    # Count the total number of distinct n-grams of characters
    count_ngrams_chars(2, 8, tweets_df['clean_text'])

    # Plot a token log frequency
    Y = words_in_tweet.values()
    Y = sorted(Y, reverse=True)
    X = range(len(Y))
    plt.figure()
    plt.loglog(X, Y)
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

    # Write the data frame to csv for further reference
    tweets_df.to_csv(r'data/tweets_input.csv', index=None, header=True, encoding='utf-8')


if __name__ == '__main__':
    main()
