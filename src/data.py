import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from collections import Counter
import string
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import swifter
import matplotlib.pyplot as plt
import os


#nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])



def load_data(fake_path, true_path):
    # Charger le dataset
    fake_news = pd.read_csv(fake_path)
    true_news = pd.read_csv(true_path)
    
    # Ajouter les labels
    fake_news['label'] = 0  # 0 pour les fausses news
    true_news['label'] = 1  # 1 pour les vraies news
    
    combined_df = pd.concat([true_news, fake_news], axis=0)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return combined_df


def analyze_data(df):
    class_distribution = df['label'].value_counts()
    print("Distribution des classes:")
    print(class_distribution)

    df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
    mean_length = df.groupby('label')['text_length'].mean()
    print("\nLongueur moyenne des textes (en mots):")
    print(mean_length)

    stop_words = set(stopwords.words('english'))
    fake_words = get_top_words(df[df['label'] == 0]['text'])
    true_words = get_top_words(df[df['label'] == 1]['text'])

    return fake_words, true_words



def get_top_words(text_series, n=20):
    stop_words = set(stopwords.words('english'))
    all_words = ' '.join(text_series).split()
    filtered_words = [word.lower() for word in all_words if word.lower() not in stop_words and word.isalpha()]
    return Counter(filtered_words).most_common(n)



def donnee_manquante(df):
    print("Valeurs manquantes par colonne:")
    print(df.isnull().sum())

    print(f"Nombre d'articles avant suppression des doublons: {len(df)}")
    df.drop_duplicates(subset=['text'], keep='first', inplace=True)
    print(f"Nombre d'articles après suppression: {len(df)}")



def unified_text_processor(df, text_col='text'):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    def clean_and_tokenize(text):
        # Nettoyage de base
        text = str(text).lower()
        text = re.sub(r'https?://\S+|www\.\S+|@\w+|#\w+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization 
        tokens = word_tokenize(text)
        
        # Lemmatisation et filtrage
        clean_tokens = [
            lemmatizer.lemmatize(word) 
            for word in tokens 
            if word not in stop_words and len(word) > 2
        ]
        
        return clean_tokens
    
    df['tokens'] = df[text_col].swifter.apply(clean_and_tokenize)
    df['clean_text'] = df['tokens'].apply(' '.join)
    
    df['word_count'] = df['tokens'].apply(len)
    df['char_count'] = df['clean_text'].apply(len)
    
    df['noun_count'] = df['tokens'].apply(
        lambda x: sum(1 for _, pos in nltk.pos_tag(x) if pos.startswith('NN'))
    )
    
    # Sentiment analysis
    df['sentiment'] = df['clean_text'].swifter.apply(
        lambda x: TextBlob(x).sentiment.polarity
    )
    
   
    
    return df

def save_plot(fig, filename, folder="evaluation"):
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, filename), bbox_inches='tight')
    plt.close(fig)

def plot_class_distribution(df):
    # Distribution des classes
    class_distribution = df['label'].value_counts()

    # Diagramme camembert (Pie chart)
    fig1, ax1 = plt.subplots()
    ax1.pie(class_distribution, labels=['True', 'Fake'], autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title("Répartition des Classes (Vrai vs Faux)")
    save_plot(fig1, "class_distribution_pie.png")
    
    # Bar chart
    fig2, ax2 = plt.subplots()
    sns.barplot(x=class_distribution.index, y=class_distribution.values, ax=ax2)
    ax2.set_xticklabels(['True', 'Fake'])
    ax2.set_title("Répartition des classes")
    ax2.set_ylabel("Nombre d'articles")
    save_plot(fig2, "class_distribution_bar.png")

     # Extraire les mots les plus fréquents dans les fausses nouvelles
    fake_words = get_top_words(df[df['label'] == 0]['text'])
    
    # Extraire les mots et leurs fréquences
    fake_labels, fake_counts = zip(*fake_words)

    # Création du graphique des mots les plus fréquents dans les fausses nouvelles
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(fake_counts), y=list(fake_labels), ax=ax, palette='Reds_r')
    ax.set_title("Top 20 des mots (Fake News)")
    ax.set_xlabel("Fréquence")
    ax.set_ylabel("Mots")
    
    # Affichage du graphique
    plt.show()  # Affiche le graphique
    
    # Sauvegarde du graphique
    save_plot(fig, "top_words_fake.png")

    

    

def get_final_df():

    FAKE_PATH = "../dataSet/Fake.csv"
    TRUE_PATH = "../dataSet/True.csv"
    print("Chargement des données...")
    df = load_data(FAKE_PATH, TRUE_PATH)
    print(df.head())

    print(plot_class_distribution(df))

    print("\nAnalyse des données...")
    fake_words, true_words = analyze_data(df)
    print("\nMots fréquents (Fake):", fake_words)
    print("\nMots fréquents (True):", true_words)

    donnee_manquante(df)

    print(unified_text_processor(df, text_col='text', date_col='date', visualize=True))

    # 4. Traitement NLP complet
    print("\nTraitement NLP avancé...")
    final_df = unified_text_processor(df)
    print(final_df[['clean_text', 'word_count', 'noun_count', 'sentiment']].head())
    
    return final_df
    

if __name__ == '__main__':
    
    print(get_final_df()) 
   