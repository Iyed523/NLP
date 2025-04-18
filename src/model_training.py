import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape, Bidirectional, BatchNormalization
import tensorflow as tf
import os
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
import pickle


# 1. Chargement des données
df = pd.read_pickle("processed_data/news_with_embeddings.pkl")
X = np.stack(df['w2v_embedding'].values)
y = df['label'].values

# 2. Division des données
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Dictionnaire pour stocker les modèles
models = {}

### A. Modèle LSTM ###
def build_lstm_model():
    model = Sequential([
        Reshape((300, 1), input_shape=(300,)),  # Remettre au format (timesteps, features)
        
        # LSTM bidirectionnel
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.5),
        
        # Batch Normalization pour accélérer l'entraînement et stabiliser
        BatchNormalization(),
        
        # Autre LSTM pour mieux capturer les dépendances temporelles
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.5),
        
        # Dense
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        # Sortie finale
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Création du modèle
lstm_model = build_lstm_model()

# EarlyStopping pour éviter le surapprentissage et arrêter l'entraînement si la performance ne s'améliore plus
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


lstm_model = build_lstm_model()

lstm_model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)

models['LSTM'] = lstm_model


### B. Évaluation ###
def evaluate_models():
    results = {}
    for name, model in models.items():
        if isinstance(model, Sequential):  # Pour Keras
            y_pred = (model.predict(X_test) > 0.5).astype("int32")
            acc = accuracy_score(y_test, y_pred)
        else:  # Pour scikit-learn
            acc = model.score(X_test, y_test)
        results[name] = acc
        print(f"{name}: Test Accuracy = {acc:.4f}")
    return results

results = evaluate_models()


### C. Sauvegarde ###
os.makedirs("trained_models", exist_ok=True)

# Sauvegarde du modèle LSTM
lstm_model.save("trained_models/lstm_model.keras")
print("Modèle LSTM sauvegardé avec succès!")