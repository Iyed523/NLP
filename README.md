

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
