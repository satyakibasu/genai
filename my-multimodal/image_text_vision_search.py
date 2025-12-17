import json
from PIL import Image
from sentence_transformers import SentenceTransformer
import chromadb

# This program will store images and text as embeddings in a vector database ie chroma db
# This will be used to query text and image search


#image = Image.open("dented_car.jpg")
image = Image.open("contracts/images/satyaki-page-0.png")
model = SentenceTransformer("clip-ViT-B-32")


# Initialize ChromaDB client
client = chromadb.Client()
collection = client.create_collection(name="multimodal")

# Store text embeddings
#participants = ["A cute dog", "A red car", "A man with sunglasses"]
participants = ["HDFC Securities Ltd","Arihant Capital Markets Ltd","ICICI Securities Ltd","Kotak Securities Ltd","Motilal Oswal Ltd","Prabhudas Lilladher Pvt Ltd","Reliance Securities Ltd","SBI Capital Markets Ltd","Sharekhan Ltd","Yes Securities (India) Ltd"]

text_embeddings = model.encode(participants).tolist()
collection.add(documents=participants, embeddings=text_embeddings,ids=[f"id:{i}" for i in participants])


# Store image embeddings
img_emb = model.encode(image)
collection.add(embeddings=img_emb,ids=["satyaki-page-0.png"],metadatas=[{"filename":"satyaki-page-0.png"}])

# ---- Text Query ----
query = "trader name"
q_emb = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=q_emb,
    n_results=3
)
from rich import print
print("Search Results:", results['documents'][0])

context = "\n\n".join(results['documents'][0]) 
print(context)
                                                                      




