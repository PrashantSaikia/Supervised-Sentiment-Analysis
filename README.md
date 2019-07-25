# Supervised-Sentiment-Analysis

This is a project to do a supervised classification of sentiments on a dataset combining customer reviews from Amazon, Yelp and IMDB (in total about 2750 records). I randomized the reviews and put them into 1 dataset. I took 2500 reviews for training, validating and testing the model, and the remaining ~250 reviews, without the sentiments, constitute the unseen dataset on which we can run the model.

Libraries required:
* nltk (for lemmatization in the pre-processing step - if you don't want to lemmatize, you can delete the relevant couple of lines)
* pandas
* sklearn
* matplotlib
* seaborn
