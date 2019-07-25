from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

def analyse_sentiment(df):

	# Take 80% of the data as training data, and the rest as test data; experiment with the %age to be allocated to each
	train = df.Text[0:int(0.8*len(df))]
	test = df.Text[int(0.8*len(df))+1:]
	train_target = df.Sentiment[0:int(0.8*len(df))]
	test_target = df.Sentiment[int(0.8*len(df))+1:]

	# Vectorize the text; i.e., convert the text into numerical form in order to be able to do the analysis
	stop_words = ['in', 'of', 'at', 'a', 'the']
	ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=stop_words) # Taking a window size of upto 3 words
	ngram_vectorizer.fit(train)
	X = ngram_vectorizer.transform(train)
	X_test = ngram_vectorizer.transform(test)

	# Split the training data into training and validation sets; experiment with the %age to be allocated to each
	X_train, X_val, y_train, y_val = train_test_split(X, train_target, train_size = 0.75)

	# Train the model
	model = LogisticRegression() # play around with the parameters in Logisticregression() to find the optimal parameters
	model.fit(X_train, y_train)

	# Check model accuracy on the validation data
	val_acc = accuracy_score(y_val, model.predict(X_val))

	# Check model accuracy on the test data
	test_acc = accuracy_score(test_target, model.predict(X_test))

	return val_acc, test_acc, X_test, test_target, model, ngram_vectorizer