import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, roc_curve, auc, 
                            RocCurveDisplay, ConfusionMatrixDisplay,precision_recall_curve, average_precision_score)
import shap
import lime
from lime import lime_tabular
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pickle




from tensorflow.keras.models import load_model
model = load_model("trained_models/lstm_model.keras")
df = pd.read_pickle("processed_data/news_with_embeddings.pkl")
X = np.stack(df['w2v_embedding'].values)
y = df['label'].values

y_pred = (model.predict(X) > 0.5).astype("int32")
y_proba = model.predict(X)  

# Calcul des métriques de base
metrics = {
    "Accuracy": accuracy_score(y, y_pred),
    "Precision": precision_score(y, y_pred),
    "Recall": recall_score(y, y_pred),
    "F1-score": f1_score(y, y_pred)
}

print("\nMétriques d'évaluation:")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

# Matrice de confusion
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice de confusion")
plt.savefig("evaluation/confusion_matrix.png")
plt.close()

# Courbe ROC
fpr, tpr, thresholds = roc_curve(y, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
plt.title('Courbe ROC')
plt.savefig("evaluation/roc_curve.png")
plt.close()

# Analyse des erreurs
df['prediction'] = y_pred
errors = df[df['label'] != df['prediction']]

print("\nExemple d'erreurs:")
print(errors[['clean_text', 'label', 'prediction']].sample(5))

# SHAP 
explainer_shap = shap.Explainer(
    model, 
    X[:100],  
    algorithm='permutation',
    max_evals=2*X.shape[1]+2  
)

try:
    explainer_shap = shap.Explainer(
        model,
        shap.sample(X, 100),
        feature_names=[f"embed_dim_{i}" for i in range(X.shape[1])]
    )
    shap_values = explainer_shap(X[:50])  # Premier 50 échantillons
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.savefig("evaluation/shap_summary.png", bbox_inches='tight')
except Exception as e:
    print(f"SHAP failed: {str(e)}")

def binary_predict_proba(x):
    """Wrapper pour adapter les prédictions au format LIME"""
    preds = model.predict(x).flatten()
    return np.vstack([1-preds, preds]).T

explainer_lime = lime_tabular.LimeTabularExplainer(
    X,
    feature_names=[f"dim_{i}" for i in range(X.shape[1])],
    class_names=['Real', 'Fake'],
    mode='classification',
    discretize_continuous=True,
    random_state=42
)

error_sample = errors.iloc[0]
exp = explainer_lime.explain_instance(
    error_sample['w2v_embedding'],
    binary_predict_proba,
    num_features=10,
    top_labels=1
)

def save_plot(fig, filename, folder="evaluation"):
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, filename), bbox_inches='tight')
    plt.close(fig)


def plot_embeddings_2d(embeddings, labels, title="Embeddings 2D", method='TSNE', save=False):
    """Fonction pour réduire les dimensions des embeddings et les visualiser"""
    if method == 'TSNE':
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings)
    elif method == 'PCA':
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=labels, palette='Set1', s=60, alpha=0.7, ax=ax)
    ax.set_title(f'{title} - {method}')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.legend(title='Classe', loc='upper right', labels=['True', 'Fake'])
    ax.grid(True)

    if save:
        save_plot(fig, f"embeddings_2d_{method}.png")

    plt.show() 

plot_embeddings_2d(X, y, title="Réduction des dimensions des embeddings", method='TSNE', save=True)

print("\nExplication LIME (texte):")
for feature, weight in exp.as_list():
    print(f"{feature}: {weight:.4f}")



precision, recall, _ = precision_recall_curve(y, y_proba)
avg_prec = average_precision_score(y, y_proba)

plt.figure()
plt.plot(recall, precision, label=f'AP = {avg_prec:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Courbe Précision / Rappel')
plt.legend()
plt.grid(True)
plt.savefig("evaluation/precision_recall_curve.png", bbox_inches='tight')
plt.close()

plt.figure()
plt.hist(y_proba, bins=30, alpha=0.7, color='purple')
plt.title("Distribution des probabilités prédites")
plt.xlabel("Probabilité prédite (classe 1)")
plt.ylabel("Nombre d'échantillons")
plt.grid(True)
plt.savefig("evaluation/predicted_probabilities_hist.png", bbox_inches='tight')
plt.close()






