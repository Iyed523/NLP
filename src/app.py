import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
from data import unified_text_processor, get_final_df
import warnings
warnings.filterwarnings('ignore')

# Fonction pour traiter un seul texte
def process_user_input(user_input):
    # Créer un DataFrame temporaire avec la même structure que vos données d'entraînement
    temp_df = pd.DataFrame({
        'text': [user_input],
        'title': [''],
        'subject': [''],
        'date': [''],
        'label': [None]  # None car c'est une prédiction
    })
    
    # Appliquer le traitement NLP
    processed_df = unified_text_processor(temp_df)
    return processed_df

# Charger les modèles
@st.cache_resource
def load_models():
    # Charger le modèle LSTM
    lstm_model = load_model("trained_models/lstm_model.keras")
    
    # Charger le modèle Word2Vec
    w2v_model = Word2Vec.load("models/w2v_fake_news.model")
    
    return lstm_model, w2v_model

# Interface Streamlit
st.title("🕵️‍♂️ Fake News Detector")
st.write("Entrez un article de news pour vérifier son authenticité")

# Zone de texte
user_input = st.text_area("Coller le texte de l'article ici:", height=200)

if st.button("Analyser"):
    if user_input:
        try:
            # Charger les modèles
            model, w2v_model = load_models()
            
            # 1. Prétraiter le texte
            processed_df = process_user_input(user_input)
            
            # 2. Créer l'embedding Word2Vec (identique à votre pipeline d'entraînement)
            def document_vector(tokens):
                words = [word for word in tokens if word in w2v_model.wv]
                return np.mean(w2v_model.wv[words], axis=0) if words else np.zeros(w2v_model.vector_size)
            
            embedding = document_vector(processed_df.iloc[0]['tokens'])
            embedding = np.array([embedding])  # Ajouter une dimension batch
            
            # 3. Faire la prédiction
            prediction = model.predict(embedding)
            proba = prediction[0][0]
            
            # 4. Afficher les résultats
            st.subheader("Résultat")
            if proba < 0.5:  # Changement de > à <
                st.error(f"⚠️ Fake News (confiance: {1-proba:.2%})")
            else:
                st.success(f"✅ News Authentique (confiance: {proba:.2%})")
            
            # Jauge de confiance
            st.progress(float(proba if proba > 0.5 else 1-proba))
            
            # Analyse détaillée
            with st.expander("Détails de l'analyse"):
                st.write("**Statistiques du texte:**")
                st.json({
                    "Mots": processed_df.iloc[0]['word_count'],
                    "Noms": processed_df.iloc[0]['noun_count'],
                    "Sentiment": "Positif" if processed_df.iloc[0]['sentiment'] > 0 
                                else "Négatif" if processed_df.iloc[0]['sentiment'] < 0 
                                else "Neutre"
                })
                
        except Exception as e:
            st.error(f"Une erreur est survenue: {str(e)}")
    else:
        st.warning("Veuillez entrer un texte à analyser")