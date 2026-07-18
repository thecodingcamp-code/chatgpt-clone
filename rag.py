import chromadb
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

db_client = chromadb.PersistentClient(path="vectordb")

collection = db_client.get_or_create_collection("company")

def store_document():

    with open("company.txt", "r") as f:
        text = f.read()

    embedding = embedding_model.encode(text).tolist()

    # Avoid duplicates
    try:
        collection.delete(ids=["company"])
    except:
        pass

    collection.add(
        ids=["company"],
        documents=[text],
        embeddings=[embedding]
    )

def retrieve(query):

    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )

    return results["documents"][0][0]