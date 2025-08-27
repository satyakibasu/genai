import pandas as pd
import os
from dotenv import load_dotenv
import openai 
from sentence_transformers import SentenceTransformer  
from sklearn.metrics.pairwise import cosine_similarity
from ollama import chat

# Load environment variables
load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")

print(API_HOST)

if API_HOST == "ollama":
    print("Using Ollama model on local...")
    #client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="nokeyneeded")
    MODEL_NAME = os.getenv("OLLAMA_MODEL", "mistral")
elif API_HOST == "github": 
    print("Using GitHub models...")
    client = openai.OpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"])
    MODEL_NAME = os.getenv("GITHUB_MODEL", "openai/gpt-4o")
else:
    raise ValueError(f"Unsupported API_HOST: {API_HOST}")



# Load the CSV file
df = pd.read_csv("hybrid.csv")
print(df.shape)

# Let's create emdeddings for the text in the CSV file. Turn each row into a string
#df["combined_text"] = df.astype(str).agg(" ".join, axis=1)
df["combined_text"] = df.apply(lambda row: " | ".join([f"{col}: {row[col]}" for col in df.columns]),axis=1)


# Load the SentenceTransformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# The below code are different ways to generate embeddings
# Option 1: Use the model to encode the entire column at once
#embeddings = model.encode(df["combined_text"].tolist()).tolist()  # Convert embeddings to list format of storing in ChromaDB
#embeddings = model.encode(df["combined_text"].tolist(), convert_to_numpy=True)  # Keep as numpy array for Ollama

# Option 2: Add embeddings directly to the DataFrame
# Add embeddings to the DataFrame and convert embeddings column to 2D array
df["embedding"] = df["combined_text"].apply(lambda x: model.encode(x,convert_to_numpy=True).tolist())
embeddings = df["embedding"].apply(pd.Series).values  # Convert to 2D numpy array

print(f"Generated {len(embeddings)} embeddings.")


# Lets create a search function without using ChromaDB
def search(query, top_k=5):
    q_embedding = model.encode([query], convert_to_numpy=True) # This is already in 2D
    q_embedding = q_embedding.reshape(1, -1)  # Reshape to 2D array for cosine similarity
    
    #print(q_embedding)
    
    sim = cosine_similarity(q_embedding, embeddings).flatten() # flatten to get a 1D array
    #print(f"Similarity scores: {sim}")
    #print(sim.argsort())
    
    top_idx = sim.argsort()[::-1][:top_k]  # # argsort gives the position, then sort descending
    print("Best match row:", top_idx)

    return top_idx


#user_query = "how fast is the prius v?"

#top_idx = search(user_query)
#print("Top results:\n",top_idx)
#retrieved_context = "\n".join(df["combined_text"].iloc[top_idx].tolist())
#print(retrieved_context)

QUERY_REWRITE_SYSTEM_MESSAGE = """
You are a helpful assistant that rewrites user questions into good keyword queries
for an index of CSV rows with these columns: vehicle, year, msrp, acceleration, mpg, class.
Good keyword queries don't have any punctuation, and are all lowercase.
You will be given the user's new question and the conversation history.
Respond with ONLY the suggested keyword query and no other text.
"""

SYSTEM_MESSAGE = """
You are a helpful assistant that answers questions about cars based off a hybrid car data set.
You must use the data set to answer the questions, you should not provide any info that is not in the provided sources.
Use the context provided to answer the question.
"""

messages = [{"role": "system", "content": SYSTEM_MESSAGE},]


while True:
    user_query = input("Enter your question (or type 'exit' to quit): ")
    if user_query.lower() == 'exit':
        break
    

    # Rewrite the query to fix typos and incorporate past context
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.05,
        messages=[
            {"role": "system", "content": QUERY_REWRITE_SYSTEM_MESSAGE},
            {"role": "user", "content": f"New user question: {user_query}\n\nConversation history: {messages}"},
        ],
    )
    search_query = response.choices[0].message.content.strip()
    print(f"Rewritten query: {search_query}")


    #top_idx = search(user_query)
    top_idx = search(search_query)
    
    print("Top results:\n", top_idx)
    retrieved_context = "\n".join(df["combined_text"].iloc[top_idx].tolist())
    print(f"Retrieved context:\n{retrieved_context}\n")
    
    # Create the prompt for the model
    # Use the retrieved context to answer the question  
    prompt = f"""
    You are a helpful assistant that answers questions about cars based off a hybrid car data set.
    You must use the data set to answer the questions, you should not provide any info that is not in the provided sources.".
    Question: {user_query}
    Context: {retrieved_context}    
    """
    messages.append({"role": "user", "content": prompt})
 


    # Test using the GIT Hub to answer the question
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=150,
        top_p=0.95,
        n=1,
        stop=None,
    )
    bot_response = response.choices[0].message.content.strip()
    messages.append({"role": "user", "content": bot_response})

    print(f"\nResponse from {API_HOST} using {MODEL_NAME}\n")
    print(bot_response)

"""
# Test using the Ollama model to answer the question
print("\nTesting Ollama model...\n")
response = chat(model="mistral",
    messages=[
        {"role": "system", "content": "Only answer using the given context. If not found, say 'Not in context'."},
        {"role": "user", "content": prompt},
    ])
print(response['message']['content'])

"""
