# Some ref materials - https://realpython.com/chromadb-vector-database/
# https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://medium.com/%40chilldenaya/vector-database-introduction-and-python-implementation-4a6ac8518c6b&ved=2ahUKEwjFo4qP3e6OAxVqUGwGHa7xM-4QFnoECBgQAQ&usg=AOvVaw2acRTR64htdhpeLAmdNemU

import chromadb #by default, Chroma uses the Sentence Transformers all-MiniLM-L6-v2 model
from rich import print


client = chromadb.PersistentClient(path="./chroma_store")

# Code to create a collection and create some documents. This is inserted to the chromadb
'''
collection = client.create_collection(
    name="all-my-documents",
    metadata={"hsnw:space":"cosine"} # HNSW (Hierarchical Navigable Small World) 
    )

documents=[
    "this is a document about food",
    "This is a document about animals", 
    "This is a document about cats and dogs"]

metadatas=[
        {"topic":"food"},
        {"topic":"animal"},
        {"topic":"animal"},
        ]

ids=["doc1","doc2","doc3"]       
collection.add(documents=documents, metadatas=metadatas,ids=ids)
results = collection.query(query_texts=["This is asking a query on pizza"], n_results=2)

for doc, metadata in zip(results['documents'][0],results['metadatas'][0]):

    print(doc)
    print(metadata)

'''
# Code to retrieve a collection and see the embeddings for a document

# Get your collection
collection = client.get_collection("all-my-documents")

results_1 = collection.query(query_texts=["This is asking a query on pizza","where are the cats?"], n_results=1, where={"topic":{"$eq":"food"}})
print(results_1)
print(results_1["distances"])

for doc, metadata in zip(results_1['documents'][0],results_1['metadatas'][0]):
    print(doc)
    print(metadata)



# Fetch the documents with their embeddings
results = collection.get(include=["documents","embeddings","metadatas"],ids=["doc1","doc2"])

for doc_id, embedding in zip(results["ids"], results["embeddings"]):
    print(f"ID: {doc_id}")
    print(f"Embedding: {len(embedding)}\n")



