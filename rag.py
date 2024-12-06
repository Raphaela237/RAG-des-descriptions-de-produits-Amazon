import streamlit as st
import openai
from chromadb import PersistentClient
from chromadb.api.client import SharedSystemClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import os
import json
import re
from dotenv import load_dotenv

# Désactiver la parallélisation des tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Récupérer la clé API OpenAI depuis le fichier .env
openai_api_key = os.getenv("API_KEY")

# Vérifie si la clé API est présente
if openai_api_key:
    openai.api_key = openai_api_key
else:
    raise ValueError("La clé API OpenAI n'est pas définie dans le fichier .env")

# Charger les descriptions depuis le fichier meta.jsonl
descriptions = []
with open("meta.jsonl", 'r', encoding='utf-8') as file:
    for line in file:
        record = json.loads(line)
        if 'description' in record:
            descriptions.append(record['description'])

# Nettoyage des descriptions
def cleaner(text_to_clean):
    cleaned_descriptions = []
    for desc in text_to_clean:
        if isinstance(desc, list):
            desc = ' '.join(desc)  # Joindre les éléments de la sous-liste en une seule chaîne
        desc = desc.lower()
        desc = re.sub(r'[^\w\s]', '', desc)  # Supprimer la ponctuation
        cleaned_descriptions.append(desc)
    return cleaned_descriptions

descriptions = cleaner(descriptions)

# Fonction de segmentation des descriptions
def segmentation(descriptions):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=50,
        length_function=len
    )
    segmented_descriptions = []
    for desc in descriptions:
        segmented_desc = text_splitter.split_text(desc)
        segmented_descriptions.extend(segmented_desc)
    return segmented_descriptions

segments = segmentation(descriptions)

# Création des embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(segments)

# Réinitialiser le cache de ChromaDB
SharedSystemClient.clear_system_cache()
print("Cache de ChromaDB vidé.")

# Définir le répertoire de persistance
persist_directory = "db/"

# Création d'un client persistant
try:
    client = PersistentClient(path=persist_directory)
    print("Nouvelle instance de ChromaDB créée.")
except Exception as e:
    raise RuntimeError(f"Erreur lors de la création du client ChromaDB : {e}")

# Vérification ou création de la collection
try:
    collection = client.create_collection(name="Amazon_product_descriptions")
    print("Nouvelle collection créée.")
except Exception as e:
    print(f"Erreur lors de la création de la collection : {e}")
    collection = client.get_collection(name="Amazon_product_descriptions")
    print("Collection existante récupérée.")

# Ajouter les segments à la collection
ids = [f"doc_{i}" for i in range(len(segments))]
collection.add(
    documents=segments,
    embeddings=embeddings,
    metadatas=[{"source": "Amazon"}] * len(segments),
    ids=ids
)

# Fonction pour configurer le système de récupération
def configure_retriever(collection, k=5):
    def retriever(query, model, collection):
        query_embedding = model.encode([query])[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        return results["documents"]
    return retriever

retriever = configure_retriever(collection, k=5)

# Fonction de réponse à la question
def get_answer_from_docs(query, retriever, model, collection, k=5):
    retrieved_docs = retriever(query, model, collection)
    
    if not retrieved_docs:
        return "Je ne dispose pas d'informations pouvant répondre à cette question. Pour plus de fiabilité, veuillez consulter sur internet."

    flat_docs = [item for sublist in retrieved_docs for item in sublist] if isinstance(retrieved_docs[0], list) else retrieved_docs
    documents_str = "\n".join(flat_docs)
    
    prompt = f"""
    Vous êtes un assistant conçu pour répondre aux questions des utilisateurs en utilisant uniquement les informations contenues dans une base de données de descriptions de produits. Vous ne devez pas utiliser de connaissances externes ou internes non fournies dans les documents.
    
    Veuillez répondre de manière précise et factuelle en extrayant les informations spécifiques des documents fournis. Si vous ne trouvez pas de réponse appropriée dans les documents, indiquez clairement que vous ne savez pas.

    Si vous avez trouvé des passages pertinents, fournissez les documents ou les passages exacts d'où vous avez extrait les informations.

    Votre tâche est de répondre uniquement avec les informations extraites des documents suivants :
    {documents_str}

    Question: {query}

    Réponse :
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant designed to answer questions based on the documents provided. You should not use any external or internal knowledge, only the documents given."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=200,
        top_p=1.0,
        n=1
    )

    return response['choices'][0]['message']['content'].strip()

# Création de l'interface Streamlit
 
def main():
    st.title("Assistant RAG - Recherche de produits Amazon")
    user_query = st.text_input("Posez votre question sur les produits Amazon:")

    # Si une question est posée
    if user_query:
        # Appeler la fonction pour générer une réponse dynamique
        response = get_answer_from_docs(user_query, retriever, model, collection)
        
        # Afficher la réponse
        st.markdown(
            f"""
            <div class="response-box">
            <img src="https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg" class="amazon-logo" />
            <div class="response-text">{response}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Inviter à poser une nouvelle question
        st.text_input("Posez une nouvelle question ici...", key="next_question")

if __name__ == "__main__":
    main()