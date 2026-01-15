## transform text
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle



class TextTransformer:
    
    def __init__(self):
        nltk.download('stopwords'); nltk.download('wordnet')
        self.stop = set(stopwords.words('english'))
        self.stop.add("u")
        self.lemmatizer = WordNetLemmatizer()
        with open('App/models/tfidf_vectorizer.pkl', 'rb') as f:
            self.tfidf = pickle.load(f)
            
    def transform(self, texts):
        
        return self.tfidf.transform(texts)
    
    def clean_text(self, s):
        s = (s or "").lower()
        s = re.sub(r"http\S+|www\S+","", s)
        s = re.sub(r"[^a-z0-9\sàâéèêçùôî\-]", " ", s)
        tokens = [t for t in s.split() if t not in self.stop]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)
    
    
    



    
    
    
    
