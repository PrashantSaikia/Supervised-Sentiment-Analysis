# Supervised Sentiment Analysis (Sentiment classification)

This is a project to do a supervised classification of sentiments on a dataset combining customer reviews from Amazon, Yelp and IMDB (in total about 2750 records). I collected and aggregated the data from [here](https://www.kaggle.com/marklvl/sentiment-labelled-sentences-data-set). I randomized the reviews and put them into 1 dataset. I took 2500 reviews for training, validating and testing the model, and the remaining ~250 reviews, without the sentiments, constitute the unseen dataset on which we can run the model.

Libraries required:
* nltk (for lemmatization in the pre-processing step - if you don't want to lemmatize, you can delete the relevant couple of lines)
* pandas
* sklearn
* matplotlib
* seaborn

The input dataset is preprocessed and saved in the `datasets` folder once, and for subsequent runs it skips the preprocessing step, to save time, and uses the existing preprocessed dataset.

![](https://user-images.githubusercontent.com/39755678/61920373-f0655b00-af8b-11e9-932d-3c96e2e7ea77.png)

After the analysis and model training is done, the confusion matrix is saved in a folder, and the classification report is appended to a file along with the time stamp.

`Validation accuracy: 80.80%`

`Test accuracy: 78.75%`

![](https://user-images.githubusercontent.com/39755678/61866163-4348ff00-af07-11e9-8fcd-c6fcc06529f3.png)
![](https://user-images.githubusercontent.com/39755678/61919839-f78b6980-af89-11e9-91a7-07cd48c7cd48.png)

The code works with more number of classes for Sentiments (like, `good`, `neutral` and `bad` for example). It will technically work for any number of classes, but of course as the number of classes increase, the accuracy will go down. 

`Validation accuracy: 76.34%`

`Test accuracy: 73.75%`

![](https://user-images.githubusercontent.com/39755678/61920108-fe66ac00-af8a-11e9-803c-b2c47eef4de0.png)
![](https://user-images.githubusercontent.com/39755678/61920186-4ede0980-af8b-11e9-9ab5-69038033d190.png)

And if the Sentiment column in the dataset has a continuous range of values (say, 0 to 1, or -1 to 1), the code will run but it will give bad results, as it will treat the Sentiment column as a categorical variable and will train the model for each unique value in Sentiments. In that case, you would want to modify the Sentiment column to convert the continuous values into discrete values. For example, if your Sentiment column ranges from 0 to 1 and if you want 2 classes, you could convert sentiments from 0 to 0.5 as  `negative` and 0.5 to 1 as `positive`. You could of course choose some other threshold based on your domain knowledge.
