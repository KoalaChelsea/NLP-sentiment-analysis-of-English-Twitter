# NLP Sentiment Analysis of English Twitter
## Team
Dance Squard: Chelsea Wang, Jiaqi Tang, Xinyi Ye, Jingjing Lin

## Objective
In this Sentiment Analysis task, we aim to predict the affective states and subjective information (Positive, Neutral and Negative) of each twitter message using Naive Bayes and Logistic Regression models. 


There are 3 parts in this project:

1)Exploratory Data Analysis: conducting exploratory data analysis (EDA) of example("input") twitter messages; 

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

## Task 1: Exploratory Data Analysis (EDA)

#### Input data
- [x] The total number of tweets
- [x] The total number of characters
- [x] The total number of distinct words (vocabulary)
- [x] The average number of characters and words in each tweet
- [x] The average number and standard deviation of characters per token
- [x] The total number of tokens corresponding to the top 10 most frequent words (types) in the vocabulary
- [x] The token/type ratio in the dataset
- [x] The total number of distinct n-grams (of words) that appear in the dataset for n=2,3,4,5.
- [x] The total number of distinct n-grams of characters that appear for n=2,3,4,5,6,7
- [ ] Plot a token log frequency. Describe what this plot means and how to interpret it. Describe out it might help you understand coverage when training a model?

#### Gold
- [ ] What is the number of types that appear in the dev data but not the training data (OOV).?
- [ ] Look at the vocabulary growth (types) combining your four gold data sets against your input data. Plot vocabulary growth at dif ference sample sizes N. 
- [ ] What is the class distribution of the training data set - how many negative, neutral, positive tweets?
- [ ] Look at the difference between the top word types across these three classes.
- [ ] What words are particularly characteristic of your training set and dev set? Are they the same? 
