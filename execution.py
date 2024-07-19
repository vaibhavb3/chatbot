import os
import string
import openai
from sentence_transformers import SentenceTransformer, util

# Load the OpenAI API key
openai.api_key = 'your_openai_api_key'

# Initialize the SentenceTransformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load and preprocess the document
def load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        document = file.read()
    return document

def chunk_document(document, chunk_size=100):
    words = document.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Generate a response from OpenAI
def generate_response(prompt, model="text-davinci-003"):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# Find the most relevant chunk using embeddings
def find_relevant_chunk(query, chunk_embeddings, chunks, model):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)
    most_similar_chunk_idx = similarities.argmax().item()
    return chunks[most_similar_chunk_idx]

# Query the relevant chunk
def query_relevant_chunk(query, relevant_chunk, model="text-davinci-003"):
    prompt = f"Context: {relevant_chunk}\n\nQuestion: {query}\nAnswer:"
    response = generate_response(prompt, model)
    return response

# Main function to run the chatbot
def main():
    print("Welcome to the Document Query Chatbot!")
    file_path = 'path/to/your/document.txt'
    document = load_document(file_path)
    chunks = chunk_document(document, chunk_size=100)
    chunk_embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
    
    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        relevant_chunk = find_relevant_chunk(query, chunk_embeddings, chunks, embedding_model)
        response = query_relevant_chunk(query, relevant_chunk)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
