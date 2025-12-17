import os
import pandas as pd
import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from lunr import lunr
from rich import print

load_dotenv()

API_HOST = os.getenv("API_HOST", "github")
MODEL_NAME = os.getenv("GITHUB_MODEL", "openai/gpt-4o")

if API_HOST == "github":
    client = openai.OpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"])
    model = SentenceTransformer('all-MiniLM-L6-v2') # encoding model for text


# Load the datafrome
df = pd.read_csv('hybrid.csv')
df = df.reset_index()
print(df)

# create a combined text column
df['combined_text'] = df.drop(columns=['index']).astype(str).agg(' '.join, axis=1)


# Create an emdedding of 'combined text' column and add to the dataframe
df['embedding'] = df['combined_text'].apply(lambda x: model.encode(x, convert_to_numpy=True).tolist())
embeddings = df['embedding'].apply(pd.Series).values  # Convert to 2D numpy array
print(f"Created embbeddings for: {len(embeddings)} rows")


# Lets create a vector-search function without using ChromaDB
def vector_search(query, top_k=2):
    q_embedding = model.encode([query], convert_to_numpy=True) # This is already in 2D
    q_embedding = q_embedding.reshape(1, -1)  # Reshape to 2D array for cosine similarity
    sim = cosine_similarity(q_embedding, embeddings).flatten() # flatten to get a 1D array
    top_idx = sim.argsort()[::-1][:top_k]  # # argsort gives the position, then sort descending
    print("Best match row from vector search:", top_idx)
    retrieved_document = df.iloc[top_idx].to_dict(orient="records")

    #print(f"Retrieved documents from vector search: {retrieved_document}")
    return retrieved_document

def text_search(query, top_k=2):
    # Convert the dataframe to a lunr index
    #documents = df.reset_index().to_dict(orient='records') #list of dictionaries
    documents = df.to_dict(orient='records') #list of dictionaries

    index = lunr(ref="index", fields=["combined_text"], documents=documents)
    results = index.search(query)
    #print(f"Text search results for '{query}': {results}")
    print("Best match row from text search:", [int(result['ref']) for result in results[:top_k]])
    retrieved_document = [documents[int(result['ref'])] for result in results[:top_k]]

    #print(f"Retrieved documents from text search: {retrieved_document}")
    return retrieved_document

def reciprocal_rank_fusion(text_results, vector_results, k=60):
    """
    Perform Reciprocal Rank Fusion (RRF) on the results from text and vector searches,
    based on algorithm described here:
    https://learn.microsoft.com/azure/search/hybrid-search-ranking#how-rrf-ranking-works
    """
    scores = {}

    for i, doc in enumerate(text_results):
        if doc["index"] not in scores:
            scores[doc["index"]] = 0
        scores[doc["index"]] += 1 / (i + k)
        
    for i, doc in enumerate(vector_results):
        if doc["index"] not in scores:
            scores[doc["index"]] = 0
        scores[doc["index"]] += 1 / (i + k)

    scored_documents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(f"Scored documents: {scored_documents}")

    # This is to get the records from the dataframe
    ids = [doc_id for doc_id, _ in scored_documents]
    #print(df.iloc[ids])

    retrieved_documents = [df.iloc[int(doc_id)].to_dict() for doc_id, _ in scored_documents]
    #print(f"Retrieved documents after RRF: {retrieved_documents}")
    
    return retrieved_documents

def rerank(query, retrieved_documents):
    """
    Rerank the results using a cross-encoder model.
    """
    encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = encoder.predict([(query, doc["combined_text"]) for doc in retrieved_documents])
    print(f"Scores from reranking: {scores}")


    scored_documents = [v for _, v in sorted(zip(scores, retrieved_documents), reverse=True)]
    #print(f"Scored documents after reranking: {scored_documents}")

    return scored_documents

def hybrid_search(query, limit):
    """
    Perform a hybrid search using both full-text and vector search.
    """
    text_results = text_search(query, limit)
    vector_results = vector_search(query, limit)
    fused_results = reciprocal_rank_fusion(text_results, vector_results)
    reranked_results = rerank(query, fused_results)

    return reranked_results[:limit]

if __name__ == "__main__":

    query = "speed of prius"

    retrieved_documents = hybrid_search(query, limit=5)

    print(f"Retrieved {len(retrieved_documents)} matching documents.")



    #context = "\n".join([f"{doc['index']}: {doc['combined_text']}" for doc in retrieved_documents[0:5]])
    context = "\n".join([f"{doc['combined_text']}" for doc in retrieved_documents[0:5]])
    print(f"Context: {context}")
    print(f"Model: {MODEL_NAME}")

    # Now we can use the matches to generate a response
    SYSTEM_MESSAGE = """
    You are a helpful assistant that answers questions about hybrid cars.
    the data contains vehicle,year,msrp,acceleration,mpg and class.
    You must use the data set to answer the questions,
    you should not provide any info that is not in the provided sources.
    Cite the sources you used to answer the question inside square brackets.
    The sources are in the format: <id>: <text>.
    """
    response = client.chat.completions.create(
    model=MODEL_NAME,
    temperature=0.3,
    messages=[
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": f"{query}\nSources: {context}"},
    ],
    )   
    
    print(f"\nResponse from {MODEL_NAME} on {API_HOST}: \n")
    print(response.choices[0].message.content)


    
