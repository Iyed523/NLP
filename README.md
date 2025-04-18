

## Étapes pour cloner le dépôt avec Git LFS

Avant de cloner ce dépôt, assure-vous d'avoir installé **Git LFS** et d'avoir suivi les étapes nécessaires pour gérer les fichiers volumineux. Voici les étapes à suivre :

### 1. **Installer Git LFS**
   Git LFS (Large File Storage) est nécessaire pour gérer les fichiers volumineux dans ce dépôt. Pour l'installer, choisis l'une des méthodes suivantes selon ton système d'exploitation.

   - **Sur macOS** (avec Homebrew) :
     ```bash
     brew install git-lfs
     ```

   - **Sur Windows** :
     Télécharge et installe Git LFS à partir du site officiel : [Git LFS pour Windows](https://git-lfs.github.com/).

   - **Sur Linux (Debian/Ubuntu)** :
     ```bash
     sudo apt-get install git-lfs
     ```

### 2. **Initialiser Git LFS**
   Une fois Git LFS installé, initialise-le dans votre répertoire Git :

   ```bash
   git lfs install


Pour tester ce projet en local, suivez les étapes ci-dessous :
Accéder au dossier src : Une fois le projet cloné, naviguez dans le dossier src, où se trouve l'application Streamlit, en exécutant :
cd NLP/src
Installer les bibliothèques nécessaires : Avant de pouvoir lancer l'application, vous devez installer les dépendances Python requises:
streamlit
pandas
numpy
tensorflow
gensim

Lancer l'application Streamlit : Une fois les bibliothèques installées, vous pouvez lancer l'application Streamlit en exécutant la commande suivante dans le terminal :
streamlit run app.py




