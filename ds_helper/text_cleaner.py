
import re
import string
try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except ImportError:
    stopwords = None
    WordNetLemmatizer = None
class TextCleaner:
    def __init__(self, filler_words=None, use_lemmatization=False, language='english'):
        if filler_words is None:
            self.filler_words = {"uh", "um", "like"}
        else:
            self.filler_words = set(filler_words)
        self.use_lemmatization = use_lemmatization
        self.language = language
        if stopwords:
            self.stop_words = set(stopwords.words(language))
        else:
            self.stop_words = set()
        if use_lemmatization and WordNetLemmatizer:
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = None

    def clean(self, text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = re.findall(r'\b\w+\b', text)
        tokens = [
            word for word in tokens
            if word not in self.filler_words and word not in self.stop_words
        ]
        if self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
