import spacy
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import subprocess
import time
import warnings
import pdfplumber

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Initialize spaCy model for sentence tokenization (if needed)
nlp = spacy.load("en_core_web_sm")

# Function to load data from an Excel sheet and create chunks
def load_excel_data(excel_path, sheet_name="Sheet1"):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    chunks = []
    
    for index, row in df.iterrows():
        # Convert the entire row into a string and treat it as a chunk
        chunk = " ".join([f"{col}: {row[col]}" for col in df.columns])
        chunks.append(chunk.strip())
    
    return chunks, df.columns.tolist()

# Function to load data from a PDF file and create chunks
def load_pdf_data(pdf_path):
    chunks = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            doc = nlp(text)
            for sent in doc.sents:  # Split the text into sentences or chunks
                chunks.append(sent.text.strip())
    
    return chunks

# Function to generate a response with the LLaMA 3 model
def generate_response_with_llama3(context, query):
    prompt = f"{context}\n\nQuery: {query}"
    try:
        process = subprocess.Popen(
            ['ollama', 'run', 'llama3'],  # Adjust this command to match your environment
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'  # Ensures proper handling of special characters
        )
        stdout, stderr = process.communicate(input=prompt)
        
        if process.returncode != 0:
            return f"Error: {stderr}"
        
        return stdout.strip()
    except Exception as e:
        return f"An exception occurred: {str(e)}"

# Function to retrieve relevant chunks based on a query
def retrieve_relevant_chunks(query, k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

# Function to generate a response based on the query and combined document content
def respond_to_query(query):
    retrieved_chunks = retrieve_relevant_chunks(query)
    if not retrieved_chunks:
        return "No relevant information found in the documents."
    
    # Combine the retrieved chunks into a context string
    context = " ".join(retrieved_chunks)
    response = generate_response_with_llama3(context, query)
    
    return response

# Main execution
def main():
    excel_path = r"C:\Users\hp\OneDrive\Pictures\Documents\ACCENTURE.xlsx"  # Path to your Excel file
    sheet_name = "Batch 1 to 4"  # Name of the sheet in the Excel file
    pdf_path = r"C:\Users\hp\OneDrive\Pictures\Documents\AIEndSemNotes.pdf"  # Path to your PDF file

    print("Loading data from Excel and PDF...")
    start_time = time.time()
    
    # Load data from Excel and PDF files
    global chunks, columns
    chunks_excel, columns = load_excel_data(excel_path, sheet_name)
    chunks_pdf = load_pdf_data(pdf_path)
    
    # Combine both Excel and PDF chunks
    chunks = chunks_excel + chunks_pdf

    print("Data loading completed.")
    print(f"Time taken for loading data: {time.time() - start_time:.2f} seconds")

    # Initialize SentenceTransformer model for embeddings
    global model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create FAISS index for text retrieval
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    global index
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # Interactive query loop
    print("Interactive mode is now active. Type 'exit' to quit.")
    while True:
        query = input("Enter your query: ")
        if query.lower() == 'exit':
            break
        response = respond_to_query(query)
        print("Response:", response)
        print()  # Print a blank line for better readability

if __name__ == "__main__":
    main()
