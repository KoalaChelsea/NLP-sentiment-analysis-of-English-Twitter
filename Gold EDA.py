import pandas as pd
import numpy as np
import re
import string
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import collections
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import glob



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


def count_distinct_words(list):
    # split/tokenize all words
    words_list = [token_list for token_list in list]

    # flatten the list to one massive string
    all_words = [item.lower() for sublist in words_list for item in sublist]

    # Count the number of unique words appeared
    word_counts = collections.Counter(set(all_words))

    return len(word_counts)


def main():

    files = glob.glob('data/Gold/*.txt')

    # load gold data line by line with utf8
    with open('data/gold_all.txt', 'w') as gold_all:
        for file_ in files:
            for line in open(file_, 'r'):
                gold_all.write(line)

    with open('data/gold_all.txt', encoding='utf-8') as f:
        lines_gold_all = f.readlines()
        # If there is no data in the file, don't go any further.
        if not len(lines_gold_all):
            exit

    # get a dictionary of each tweet by regex
    tweets_dict = {"tweet_id": [], "tweet_sentiment": [], "tweet_content": []}
    for line in lines_gold_all:
        matches = re.compile(r'(^\d*)(\s)([^\s]+)(\s)(.*)')
        tweets_dict["tweet_id"].append(matches.match(line).group(1))
        tweets_dict["tweet_sentiment"].append(matches.match(line).group(3))
        tweets_dict["tweet_content"].append(matches.match(line).group(5))

    # create a data frame of tweet
    tweets_gold_df = pd.DataFrame(tweets_dict)

    # report the number of tweet by checking the unique tweet id (no duplicated id found)
    print("The total number of tweets in all gold files is %s" % len(tweets_gold_df.tweet_id.unique()))

    # -------------------------------------------------------------------------------------
    # clean dataframe text column
    tweets_gold_df['clean_text'] = tweets_gold_df['tweet_content'].apply(processTweet)
    tweets_gold_df['token'] = tweets_gold_df['clean_text'].apply(TweetTokenizer().tokenize) \
        .apply(lambda x: [item for item in x if item.isalpha()])
    tweets_gold_df['token count'] = tweets_gold_df['token'].apply(len)
    stop = stopwords.words('english')
    porter_stemmer = PorterStemmer()
    tweets_gold_df['token deep clean'] = tweets_gold_df['token'].apply(lambda x: [item for item in x if item not in stop]) \
        .apply(lambda x: [porter_stemmer.stem(item) for item in x])

    # write all cleaned tweets into one text
    token_list_cleaned_gold = tweets_gold_df['token'].tolist()

    # split/tokenize all words
    words_list_in_tweet_gold = [token_list for token_list in token_list_cleaned_gold]

    # flatten the list to one massive string
    all_words_gold = [item.lower() for sublist in words_list_in_tweet_gold for item in sublist]

    # Count the number of unique words appeared
    word_counts_gold = collections.Counter(set(all_words_gold))

    print("The total number of distinct words (vocabulary) in all gold files is %s" % len(word_counts_gold))

    # Write the data frame to csv for further reference
    tweets_gold_df.to_csv(r'data/tweets_gold.csv', index=None, header=True, encoding='utf-8')

    # load input data line by line with utf8
    with open('data/Dev/INPUT.txt', encoding='utf-8') as f:
        lines_input = f.readlines()
        # If there is no data in the file, don't go any further.
        if not len(lines_input):
            exit

    # get a dictionary of each tweet by regex
    tweets_dict = {"tweet_id": [], "tweet_sentiment": [], "tweet_content": []}
    for line in lines_input:
        matches = re.compile(r'(^\d*)(\s)([^\s]+)(\s)(.*)')
        tweets_dict["tweet_id"].append(matches.match(line).group(1))
        tweets_dict["tweet_sentiment"].append(matches.match(line).group(3))
        tweets_dict["tweet_content"].append(matches.match(line).group(5))

    # create a data frame of tweet
    tweets_input_df = pd.DataFrame(tweets_dict)

    # clean dataframe text column
    tweets_input_df['clean_text'] = tweets_input_df['tweet_content'].apply(processTweet)
    tweets_input_df['token'] = tweets_input_df['clean_text'].apply(TweetTokenizer().tokenize) \
        .apply(lambda x: [item for item in x if item.isalpha()])
    tweets_input_df['token count'] = tweets_input_df['token'].apply(len)
    stop = stopwords.words('english')
    porter_stemmer = PorterStemmer()
    tweets_input_df['token deep clean'] = tweets_input_df['token'].apply(lambda x: [item for item in x if item not in stop]) \
        .apply(lambda x: [porter_stemmer.stem(item) for item in x])

    # write all cleaned tweets into one text
    token_list_cleaned_input = tweets_input_df['token'].tolist()

    # split/tokenize all words
    words_list_in_tweet_input = [token_list for token_list in token_list_cleaned_input]

    # flatten the list to one massive string
    all_words_input = [item.lower() for sublist in words_list_in_tweet_input for item in sublist]

    # Count the number of unique words appeared
    word_counts_input = collections.Counter(set(all_words_input))

    print("The total number of distinct words (vocabulary) in input file is %s" % len(word_counts_input))
    # -------------------------------------------------------------------------------------

    '''
    # -------------------------------------------------------------------------------------
    OOV = [x for x in set(all_words_input) if x not in set(all_words_gold)]
    print("The number of types that appear in the dev data but not the training data (OOV) are", len(set(OOV)))
    # -------------------------------------------------------------------------------------
    '''

    # create gold and input list for 25%, 50%, 75% and 100%
    gold_25 = token_list_cleaned_gold[:int((len(token_list_cleaned_gold) + 1) * .25)]
    gold_50 = token_list_cleaned_gold[:int((len(token_list_cleaned_gold) + 1) * .50)]
    gold_75 = token_list_cleaned_gold[:int((len(token_list_cleaned_gold) + 1) * .75)]

    input_25 = token_list_cleaned_input[:int((len(token_list_cleaned_input) + 1) * .25)]
    input_50 = token_list_cleaned_input[:int((len(token_list_cleaned_input) + 1) * .50)]
    input_75 = token_list_cleaned_input[:int((len(token_list_cleaned_input) + 1) * .75)]

    # Choose the height of the blue bars
    gold_distinct_words = [count_distinct_words(gold_25), count_distinct_words(gold_50),
                           count_distinct_words(gold_75), count_distinct_words(token_list_cleaned_gold)]

    input_distinct_words = [count_distinct_words(input_25), count_distinct_words(input_50),
                           count_distinct_words(input_75), count_distinct_words(token_list_cleaned_input)]

    # width of the bars
    barWidth = 0.3

    # The x position of bars
    r1 = np.arange(len(gold_distinct_words))
    r2 = [x + barWidth for x in r1]

    plt.bar(r1, gold_distinct_words, barWidth, color='blue', edgecolor='black', capsize=7, label='gold data sets')
    plt.bar(r2, input_distinct_words, barWidth, color='cyan', edgecolor='black', capsize=7, label='input data set')

    # general layout
    plt.xticks([r + barWidth for r in range(len(gold_distinct_words))], ['25%', '50%', '75%', '100%'])
    plt.xlabel('percentage of data used')
    plt.ylabel('count')
    plt.legend(loc='best')
    plt.title('Vocabulary Growth (types): Four Gold Data Sets vs Input Data')

    # Show graphic
    plt.show()
    plt.close()
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Class Distribution of the Gold Training Data Set
    tweets_gold_df['tweet_sentiment'].value_counts().plot(kind='bar', rot=0)
    plt.title('Class Distribution of the Gold Training Data Set')
    plt.xlabel('tweet_sentiment')
    plt.xlabel('count')
    plt.show()
    plt.close()
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------

    df_neutral = tweets_gold_df[tweets_gold_df['tweet_sentiment'] == 'neutral']
    df_positive = tweets_gold_df[tweets_gold_df['tweet_sentiment'] == 'positive']
    df_negative = tweets_gold_df[tweets_gold_df['tweet_sentiment'] == 'negative']

    # write all cleaned tweets into one text
    token_list_cleaned_neutral = df_neutral['token'].tolist()
    # split/tokenize all words
    words_list_in_tweet_neutral = [token_list for token_list in token_list_cleaned_neutral]
    # flatten the list to one massive string
    all_words_neutral = [item.lower() for sublist in words_list_in_tweet_neutral for item in sublist]

    # write all cleaned tweets into one text
    token_list_cleaned_positive = df_positive['token'].tolist()
    # split/tokenize all words
    words_list_in_tweet_positive = [token_list for token_list in token_list_cleaned_positive]
    # flatten the list to one massive string
    all_words_positive = [item.lower() for sublist in words_list_in_tweet_positive for item in sublist]

    # write all cleaned tweets into one text
    token_list_cleaned_negative = df_negative['token'].tolist()
    # split/tokenize all words
    words_list_in_tweet_negative = [token_list for token_list in token_list_cleaned_negative]
    # flatten the list to one massive string
    all_words_negative = [item.lower() for sublist in words_list_in_tweet_negative for item in sublist]

    # Set up the matplotlib figure
    plt.figure(figsize=(7, 14))

    plt.subplot(311)
    labels_neutral, values_neutral = zip(*collections.Counter(all_words_neutral).most_common(30))
    plt.bar(labels_neutral, values_neutral, color='darkblue')
    plt.xticks(labels_neutral, rotation='vertical')
    plt.ylabel('Neutral', rotation='horizontal')
    plt.title('Difference Between the Top Word Types Across These Three Classes')

    plt.subplot(312)
    labels_positive, values_positive = zip(*collections.Counter(all_words_positive).most_common(30))
    plt.bar(labels_positive, values_positive, color='darkred')
    plt.xticks(labels_positive, rotation='vertical')
    plt.ylabel('Positive', rotation='horizontal')

    plt.subplot(313)
    labels_negative, values_negative = zip(*collections.Counter(all_words_negative).most_common(30))
    plt.bar(labels_negative, values_negative, color='darkgreen')
    plt.xticks(labels_negative, rotation='vertical')
    plt.ylabel('Negative', rotation='horizontal')

    plt.tight_layout()
    plt.show()
    plt.close()
    # -------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()
