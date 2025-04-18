import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from data import get_final_df  # Import direct de votre DataFrame prétraité
import os

def train_word2vec(combined_df):
    """Entraîne un modèle Word2Vec sur les tokens"""
    sentences = combined_df['tokens'].tolist()
    
    model =  Word2Vec(
        sentences,
        vector_size=300,
        window=10,
        min_count=3,
        negative=10,
        hs=1,
        sample=1e-5,
        workers=8,
        epochs=20
)
    
    return model

def create_embeddings(model, combined_df):
    """Crée des embeddings moyens pour chaque document"""
    def document_vector(tokens):
        words = [word for word in tokens if word in model.wv]
        return np.mean(model.wv[words], axis=0) if words else np.zeros(model.vector_size)
    
    combined_df['w2v_embedding'] = combined_df['tokens'].apply(document_vector)
    return combined_df



if __name__ == '__main__':
    # 1. Chargement des données nettoyées
    df = get_final_df()  # Appel direct à votre fonction de data.py
    
    # 2. Entraînement du modèle
    w2v_model = train_word2vec(df)
    
    # 3. Création des embeddings
    df_with_embeddings = create_embeddings(w2v_model, df)
    
    # 4. Sauvegarde
    os.makedirs("models", exist_ok=True)
    w2v_model.save("models/w2v_fake_news.model")
    df_with_embeddings.to_pickle("processed_data/news_with_embeddings.pkl")
    
    print("Modèle Word2Vec et embeddings créés avec succès!")