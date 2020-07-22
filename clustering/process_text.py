import re
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def process_text(text):
    '''
    The text is parsed to transform text into a meaningful list of words for the clusterizer.
    Some noise is removed like some punctuation and what comes before an apostrophe.
    Stopwords are removed and the rest is stemmed (adding more noisy stopwords and changing the stemmer could enhance the clustering).
    '''
    text = text.replace(","," ")
    querywords = text.split()
    for idx, word in enumerate(querywords):
        if "'" in word:
            querywords[idx] = re.sub(r"^(.*?)\'", "", word)
    text = ' '.join(querywords)
    punctuation = string.punctuation + "¦,€$™«»…’"
    text = text.translate(str.maketrans('', '', punctuation))

    stemmer = PorterStemmer()
    en_stopwords = stopwords.words('english')
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t.lower() not in en_stopwords and len(stemmer.stem(t))>1]

    return tokens