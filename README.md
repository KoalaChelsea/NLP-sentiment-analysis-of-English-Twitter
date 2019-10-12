# NLP Sentiment Analysis of English Twitter
## Team
Dance Squad: Chelsea Wang, Jiaqi Tang, Xinyi Ye, Jingjing Lin

## Objective
In this Sentiment Analysis task, we aim to predict the affective states and subjective information (Positive, Neutral and Negative) of each twitter message using Naive Bayes and Logistic Regression models. 


There are 3 parts in this project:

1)Exploratory Data Analysis: conducting exploratory data analysis (EDA) of example("INPUT.txt") twitter messages; 

2)Message polarity classification: training data with Naive Bayes Model and Logistic Regression Model and validating those two models using additional datasets;

3)Performance evaluation: using test data to demonstrate the performance of classifiers (models)


## Datasets

+ Data/Dev - This is your INPUT file (INPUT.txt); the data you will classify
+ Data Gold - train.txt. This is the data for training your model.
+ Data/Gold - dev.txt. This is the validation data set for tuning your model.
+ Data/Gold - devtest.txt. This is what you can evaluate your results with during your development time.
+ Data/Gold - test.ext.  This is what you can use to evaluate your results before you provide you a final test set.
+ Data/Test - This is the (gold) data for evaluating your model. We will provide this data set the day your assignment is due, thus the folder is empty.

## Code Instruction
There are two python files in this project

 - Exploratory Data analysis.py, for part 1)
 - XXXXXXXXXXXXXX, for part 2) and part 3)
 
 And one output file for the results of the INPUT.txt file
 
 - OUTPUT.csv (CHECK!!!!!!!!!) 
 
 It takes around 40 minutes to run the entire project.

### Task 1: Exploratory Data Analysis (EDA)

#### Input data
- [x] The total number of tweets
- 18816
- [x] The total number of characters
- 1343152
- [x] The total number of distinct words (vocabulary)
- 18816
- [x] The average number of characters and words in each tweet
- The average number of characters per tweet is 82.8 and the average number of words per tweet is 14.3
- [x] The average number and standard deviation of characters per token
- The average number of characters per token per tweet is 5.1 and the average standard deviation per tweet's token is 1.2
- [x] The total number of tokens corresponding to the top 10 most frequent words (types) in the vocabulary
- [('the', 5733), ('to', 3792), ('a', 2954), ('of', 2594), ('and', 2461), ('in', 2267), ('is', 2251), ('for', 1945), ('i', 1744), ('on', 1396)]
- [x] The token/type ratio in the dataset
- 8.73203656462585

- [x] The total number of distinct n-grams (of words) that appear in the dataset for n=2,3,4,5.
- [x] The total number of distinct n-grams of characters that appear for n=2,3,4,5,6,7
- [ ] Plot a token log frequency. Describe what this plot means and how to interpret it. Describe out it might help you understand coverage when training a model?


#### Gold
- [x] What is the number of types that appear in the dev data but not the training data (OOV).?
- 8442
- [x] Look at the vocabulary growth (types) combining your four gold data sets against your input data. Plot vocabulary growth at difference sample sizes N. 
- [Growth graphs](...)
- [x] What is the class distribution of the training data set - how many negative, neutral, positive tweets?
- [Class distribution](...)
- [x] Look at the difference between the top word types across these three classes.
- [Difference classes](...)
- [ ] What words are particularly characteristic of your training set and dev set? Are they the same? 
