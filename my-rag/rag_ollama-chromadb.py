
import chromadb
from ollama import chat
import os
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import os, psutil

process = psutil.Process(os.getpid())
print(f"Before Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# for file reading
import pandas as pd
from docx import Document
from pypdf import PdfReader

# Step 1: Load the documents. The content of the documents is loaded from a folder and stored in a list.
def load_documents(folder):
    documents = []
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())
        elif filename.endswith('.docx'):
            doc = Document(os.path.join(folder, filename))
            documents.append('\n'.join([para.text for para in doc.paragraphs]))
        elif filename.endswith('.pdf'):
            reader = PdfReader(os.path.join(folder, filename))
            documents.append('\n'.join([page.extract_text() for page in reader.pages if page.extract_text()]))
        elif filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder, filename))
            csv_text = df.to_string(index=False)
            documents.append(csv_text)
        else:
            print(f"Skipping unsupported file: {file}")

    return documents    

# Step 2: Chunk and embed the documents.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
all_text = "\n".join(load_documents("../../python_material"))  # Load all documents into a single string

chunks = text_splitter.split_text(all_text)

# Load the SentenceTransformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks).tolist()  # Convert embeddings to list format
print(f"Generated {len(chunks)} chunks with embeddings.")


# Step 3: Store the documents and their embeddings in ChromaDB.
client = chromadb.PersistentClient(path="./chroma_store")

# Create a collection
collection = client.get_or_create_collection(
    name="python-training-material",
    metadata={"hsnw:space": "cosine"}  # HNSW (Hierarchical Navigable Small World) 
)                           

if len(collection.get()["ids"]) == 0:
    collection.add(documents=chunks,embeddings=embeddings,metadatas=[{"topic": "general"} for _ in chunks],  # Example metadata
    ids=[f"doc_{i}" for i in range(len(chunks))])  # Unique IDs for each chunk


    print(f"Added {len(chunks)} chunks to the collection.")

print(f"After Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# Step 4: Query the collection + Ask Mistral for information.
def query_chroma(query_text, n_results=5):
    q_embeddings = model.encode([query_text]).tolist()  # Embed the query text
    results = collection.query(query_embeddings=q_embeddings,n_results=n_results)
    print(f"Query results {results}")

    return results["documents"][0]

def ask_llm(context,question):
    prompt = f"""You are an expert assistant. You must answer ONLY using the following context.
    If the answer is not in the context, say "I don't know" — do not guess.

    ### CONTEXT:
    {context}

    ### QUESTION:
    {question}

    ### ANSWER:
    """
    response = chat(model="mistral",messages=[
        {"role": "system", "content": "Answer ONLY using the context provided. Say 'I don't know' if not available."},
        {"role": "user", "content": prompt}
        ])
    
    return response['message']['content']



#Fetch embeddings and documents
def print_embeddings(collection):
    # Step 1: Retrieve all IDs
    ids = collection.get()["ids"]
    data = collection.get(include=["embeddings", "documents"])

    print(f"Total documents: {len(data['documents'])}")
    print(f"Total embeddings: {len(data['embeddings'])}")
    # Step 3: Print each embedding and its text
    for idx, (doc, embedding) in enumerate(zip(data["documents"], data["embeddings"])):
        print(f"\n🔹 Document {idx+1}:")
        print(f"Text: {doc[:200]}...")  # first 200 chars
        print(f"Vector length: {len(embedding)}")
        print(f"Embedding (first 10 dims): {embedding[:10]}")


# Step 5: Answer the Q&A
while True:
    user_query = input("Enter your question (or 'exit' to quit): ")
    if user_query.lower() == 'exit':
        break

    # Query ChromaDB and ask Mistral
    print("Querying ChromaDB...")
    context = "\n\n".join(query_chroma(user_query)) 

    answer = ask_llm(context, user_query)
    print(f"Answer: {answer}")






