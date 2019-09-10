import pandas as pd
from core_functions.pre_process import pre_process
from core_functions.analyse import analyse_sentiment
from core_functions.evaluate import model_evaluation
import warnings
import os

warnings.filterwarnings("ignore")  

# If the pre-processed dataset is not already present in the folder, load the original dataset, pre-process it and save it
if not os.path.isfile('datasets/Pre-processed Dataset.csv'):
    import time

    df = pd.read_csv('datasets/Dataset.csv', encoding="ISO-8859-1")
    '''
    The dataset is expected to have 2 columns: 
    1. The "Text" column, whose each row contains some strings of text whose sentiment we want to anslyse.
    2. The "Sentiment" column, whose each row contains the sentiment of the text in the same row (good, bad, neutral).	
    '''

    # Pre-process the dataset to clean it and make it suitable for analysis 
    print('Preprocessing...')
    t = time.time()
    df.text= df.text.apply(pre_process)

	print('==================================================================\nTime taken for pre-processing the data = {0:.2f} seconds.\n=================================================================='.format(time.time()-t))

    df.to_csv('datasets/Pre-processed Dataset.csv', index=False)

# Otherwise load the pre-processed dataset
else:
    df = pd.read_csv('datasets/Pre-processed Dataset.csv')

# shuffle the rows of the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Do the sentiment analysis and check model accuracy
val_acc, test_acc, X_test, test_target, model, ngram_vectorizer = analyse_sentiment(df)
print('Validation accuracy: {0:.2f}%'.format(100*val_acc))
print('Test accuracy: {0:.2f}%'.format(100*test_acc))

# Evaluate the model
labels = df.Sentiment.unique()
labels = [str(i) for i in labels]
model_evaluation(X_test=X_test, test_target=test_target, target_names=labels, model=model)
# The confusion matrix and the classification report are saved in the folder.

# ===================================================== #

# Make predictions on an unseen/new dataset
new_df = pd.read_csv('datasets/New Dataset.csv', encoding="ISO-8859-1")
'''
The new dataset is expected to have just 1 column: 
The "Text" column, whose each row contains some strings of text whose sentiment we want to analyse.
'''
print('=======================================================\nAnalysing the new dataset...')

# The text column in the new dataset is going to be pre-processed for the analysis. So, preserve the original text column to put back later.
orig_text_col = new_df.Text

# Pre-process the new dataset to clean it and make it suitable for analysis
for i in range(len(new_df)):
    new_df.Text[i] = pre_process(new_df.Text[i])

# Add a header for the Sentiment column
new_df = new_df.reindex(columns=['Text', 'Sentiment'])

# Convert the text into numbers for analysis with the same vectorizer as the one used for the original dataset (to preserve the dimensions)
features = ngram_vectorizer.transform(new_df.Text).toarray()
new_df.Sentiment = model.predict(features)

new_df.to_csv('datasets/New Dataset - Results.csv', index=False)
new_df.Text = orig_text_col
print('Done.')
