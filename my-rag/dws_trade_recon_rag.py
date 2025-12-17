# This code will read pdf files and extract text and convert to images.
# The images will be sent to GPT along with the text for structured extraction.
# This uses RAG to find relevant pages using FAISS and text embeddings.
# Steps:
# 1. Extract text + images from PDF
# 2. Build FAISS index using text embeddings
# 3. Query FAISS to find relevant pages
# 4. Send both text + image to LLM for structured extraction

import fitz  # PyMuPDF
import os
import base64
import openai
import faiss
import numpy as np
from dotenv import load_dotenv

from pydantic import BaseModel
from typing import Optional, List
import json
from rich import print

load_dotenv()

#---------------------------------------
# Step 1. Extract text + images from PDF
#---------------------------------------

def extract_text_and_images(pdf_path, output_dir="pdf_pages",password=""):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)

    if doc.authenticate(password):  
    
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            image_path = os.path.join(output_dir, f"page_{i}.png")
            pix = page.get_pixmap()
            pix.save(image_path)
            
            pages.append({
                "page_number": i,
                "text": text,
                "image_path": image_path
            })
        return pages

# Send images to LLM as base64
def open_image_as_base64(filename):
    with open(filename, "rb") as image_file:
        image_data = image_file.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    
    return f"data:image/png;base64,{image_base64}"


API_HOST = os.getenv("API_HOST", "github")
MODEL_NAME = os.getenv("GITHUB_MODEL", "openai/gpt-4o")

if API_HOST == "github":
    client = openai.OpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"])



#---------------------------------------
# Step 2: Build FAISS index using text embeddings
#---------------------------------------
def embed_texts(texts, model="text-embedding-3-small"):
    embeddings = [
        client.embeddings.create(input=t, model=model).data[0].embedding
        for t in texts
    ]
    return np.array(embeddings, dtype="float32")

def build_faiss_index(pages):
    texts = [p["text"] for p in pages]
    embeddings = embed_texts(texts)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    return index, texts

#---------------------------------------
# Step 3: Query FAISS to find relevant pages
#---------------------------------------
def search_index(query, index, texts, k=2):
    query_emb = embed_texts([query])
    distances, indices = index.search(query_emb, k)
    return [(texts[i], i) for i in indices[0]]

#---------------------------------------
# Step 4: Send both text + image to LLM for structured extraction
#---------------------------------------
class TradeExtractionMessage(BaseModel):
    trade_id: str
    trade_date: str
    symbol: str
    quantity: int
    price: float
    counterparty: str
    status: str
    notes: Optional[str] = None

class TradeExtractionResponse(BaseModel):
    trades: List[TradeExtractionMessage]


def extract_trades_with_llm(page_text, page_image):
    image_base64 = open_image_as_base64(page_image)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # vision-capable model
        messages=[
            {"role": "system", "content": "You are an assistant that extracts trade details into structured JSON."},
            {"role": "user", "content": [
                {"type": "text", "text": f"Extract all trade details from this page text:\n{page_text}\n\nReturn as JSON."},
                {"type": "image_url", "image_url":{"url": f"{image_base64}"}}
            ]}
        ],
        response_format={ "type": "json_object" }
    )

    json_text = response.choices[0].message.content
    raw = json.loads(json_text)

    print("Raw extraction:", raw)
    return raw



# Main function to process PDF and extract trades

def process_pdf(pdf_path, query,password=""):
    # Step 1: Extract text + images
    pages = extract_text_and_images(pdf_path,password=password)


    # Step 2: Build FAISS index
    index, texts = build_faiss_index(pages)

    # Step 3: Search relevant pages
    results = search_index(query, index, texts, k=2)
    print("Relevant pages:", results)

    # Step 4: Extract trades using LLM with both text + image
    all_trades = []
    for _, idx in results:
        print(idx)
        trades = extract_trades_with_llm(pages[idx]["text"], pages[idx]["image_path"])
        all_trades.extend(trades)

    return all_trades


pdf_path = "contracts/satyaki.pdf"
query = "Extract equity trade reconciliation details"
trades = process_pdf(pdf_path, query,password="SAT0808")
print("Extracted trades:",trades)

