import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from rich import print
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv
import json
import pandas as pd


load_dotenv(override=True)


#client = openai.OpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"])
#MODEL_NAME = os.getenv("GITHUB_MODEL", "openai/gpt-4o")

pdf_location = "documents"
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name="gpt-4o", chunk_size=500, chunk_overlap=125)

# Load embedding model (if not using the Git Hub hosted models)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
all_chunks = []

# Step 1: Load the documents. The content of the documents is loaded from a folder and stored in a list.
for filename in os.listdir(pdf_location):
    if filename.endswith('.pdf'):
        print("processing file:", filename)
        texts = pymupdf4llm.to_markdown(os.path.join(pdf_location, filename))
        texts_list = [texts]  # Wrap in a list to match the expected input format

        # Split the text into smaller chunks
        chunks = splitter.create_documents(texts_list, metadatas=[{"source": filename}])
        #print(chunks)

        #Add chunk_id to metadata
        [d.metadata.__setitem__("chunk_id", i) for i, d in enumerate(chunks)]
        #embeddings = client.embeddings.create(model="text-embedding-3-small", input=d.page_content)
        #embeddings = embedder.encode(d.page_content) .tolist()
        
        
        file_chunks = [{"id": f"{filename}-{(i + 1)}", "text": d.page_content} for i, d in enumerate(chunks)]

        # Generate embeddings using openAI SDK for each text
        for file_chunk in file_chunks:
            file_chunk["embedding"] = (embedder.encode(file_chunk['text']) .tolist())
    
        all_chunks.extend(file_chunks)

# Save the documents with embeddings to a JSON file
with open("pdf_ingested_chunks.json", "w") as f:
    json.dump(all_chunks, f, indent=4)

# Load the JSON file into a DataFrame for further processing
df = pd.DataFrame(all_chunks)
print(df.head())

embeddings = df["embedding"].apply(pd.Series).values  # Convert to 2D numpy array

# use the same logic as defined in my_rag_csv.py from the search till the end.





