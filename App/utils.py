## transform text
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import json
from google.cloud import storage
from dotenv import load_dotenv





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



def download_model():
    load_dotenv(override=True)
    
    creds = json.loads(os.environ["GCP_SECRET"])
    creds_path = "/tmp/gcp_key.json"
    
    with open(creds_path, "w") as f:
        json.dump(creds, f)
        
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

    model_folder = os.environ["MODEL_FOLDER"]
    client = storage.Client()
    bucket = client.bucket(os.environ["GCS_BUCKET"])
    
    print("BUCKET NAME:", bucket)
    bloblist = bucket.list_blobs()

    for blob in bloblist:
        print("BLOB NAME:", blob.name)

        local_path = os.path.join("/tmp", blob.name)
        
        # créer les dossiers parents
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        blob.download_to_filename(local_path)
        
    return os.path.join("/tmp", model_folder)
    
    
    
    
        
    
    
    



    
    
    
    
