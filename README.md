# RAG des descriptions de produits Amazon

Ce projet implémente un système de récupération d'informations basé sur des descriptions de produits Amazon. L'objectif est de créer une interface utilisateur qui permet de poser des questions sur les produits et obtenir des réponses pertinentes en utilisant un modèle de récupération d'information (RAG - Retrieval Augmented Generation).

## Fonctionnalités principales

- **Traitement des descriptions de produits** : Le projet charge et nettoie un fichier de descriptions de produits Amazon, en segmentant les descriptions longues en morceaux plus petits pour une meilleure gestion des données.
  
- **Génération des embeddings** : Le modèle `SentenceTransformer` est utilisé pour transformer les descriptions de produits en embeddings (représentations vectorielles) afin de permettre une recherche efficace par similarité.

- **ChromaDB pour le stockage** : Les embeddings et les segments sont stockés dans une base de données vectorielle ChromaDB, ce qui permet de récupérer rapidement les informations pertinentes lors d'une requête utilisateur.

- **Interface utilisateur avec Streamlit** : Une interface utilisateur simple est fournie via Streamlit pour permettre aux utilisateurs de poser des questions sur les produits Amazon. Le système répond en utilisant les informations extraites de la base de données.

## Prérequis

Avant de commencer, assurez-vous d'avoir installé les dépendances suivantes dans votre environnement virtuel :

- Python 3.x
- Streamlit
- OpenAI
- Sentence-Transformers
- ChromaDB
- dotenv

Vous pouvez installer les dépendances avec le fichier `requirements.txt` :

```bash
pip install -r requirements.txt
