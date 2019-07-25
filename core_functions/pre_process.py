import re
from nltk.stem.wordnet import WordNetLemmatizer

stop_words = ['in', 'of', 'at', 'a', 'the']

def pre_process(text):
    
    # lowercase
    text=str(text).lower()

    # remove numbers followed by dot (like, "1.", "2.", etc)
    text=re.sub('((\d+)[\.])', '', text)
    
    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    
    # correct some misspellings and/or replace some text with others that might be easier to handle
    text=text.replace('do not', "don't")
    
    # remove special characters except spaces, apostrophes and dots
    text=re.sub(r"[^a-zA-Z0-9.']+", ' ', text)
    
    # remove stopwords
    text=[word for word in text.split(' ') if word not in stop_words]
    
    # lemmatize
    lmtzr = WordNetLemmatizer()
    text = ' '.join((lmtzr.lemmatize(i)) for i in text)
    
    return text