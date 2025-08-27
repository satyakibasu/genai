import json
from PIL import Image
from sentence_transformers import SentenceTransformer
import chromadb

# This program will store images and text as embeddings in a vector database ie chroma db
# This will be used to query text and image search


image = Image.open("dented_car.jpg")
model = SentenceTransformer("clip-ViT-B-32")


# Initialize ChromaDB client
client = chromadb.Client()
collection = client.create_collection(name="multimodal")

# Store text embeddings
participants = ["A cute dog", "A red car", "A man with sunglasses"]
text_embeddings = model.encode(participants).tolist()
collection.add(documents=participants, embeddings=text_embeddings,ids=[f"id:{i}" for i in participants])


# Store image embeddings
img_emb = model.encode(image)
collection.add(embeddings=img_emb,ids=["participants.jpg"],metadatas=[{"filename":"dented_car.jpg"}])

# ---- Text Query ----
query = "dog with sunglasses"
q_emb = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=q_emb,
    n_results=3
)
from rich import print
print("Search Results:", results['documents'][0])

context = "\n\n".join(results['documents'][0]) 
print(context)
                                                                      




