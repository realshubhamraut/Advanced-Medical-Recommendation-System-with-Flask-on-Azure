import os
import time
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

def main():
    start_time = time.time()
    print("=" * 50)
    print("Starting Vector Database Creation Process")
    print("=" * 50)
    
    print("Loading environment variables...")
    load_dotenv(find_dotenv())
    
    DATA_PATH = "datasets/pdf/"
    DB_FAISS_PATH = "models/vectorstore/db_faiss"
    
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    
    print(f"\n[Step 1/4] Loading PDF files from {DATA_PATH}...")
    step_start = time.time()
    documents = load_pdf_files(data=DATA_PATH)
    step_time = time.time() - step_start
    print(f"✓ Loaded {len(documents)} documents in {step_time:.2f} seconds")
    
    print(f"\n[Step 2/4] Splitting documents into chunks...")
    step_start = time.time()
    text_chunks = create_chunks(extracted_data=documents)
    step_time = time.time() - step_start
    print(f"✓ Created {len(text_chunks)} text chunks in {step_time:.2f} seconds")
    
    print(f"\n[Step 3/4] Initializing embedding model...")
    step_start = time.time()
    embedding_model = get_embedding_model()
    step_time = time.time() - step_start
    print(f"✓ Embedding model initialized in {step_time:.2f} seconds")
    
    print(f"\n[Step 4/4] Creating FAISS vector database...")
    step_start = time.time()
    
    print("Processing chunks: Generating embeddings and building vector store")
    
    batch_size = 100 
    batches = [text_chunks[i:i+batch_size] for i in range(0, len(text_chunks), batch_size)]
    
    if len(batches) > 1:
        print(f"Processing {len(text_chunks)} chunks in {len(batches)} batches of size {batch_size}")
        
        print(f"Processing batch 1/{len(batches)} ({len(batches[0])} chunks)")
        db = FAISS.from_documents(batches[0], embedding_model)
        
        for i, batch in enumerate(tqdm(batches[1:], desc="Processing batches", unit="batch")):
            print(f"\nProcessing batch {i+2}/{len(batches)} ({len(batch)} chunks)")
            batch_start = time.time()
            db.add_documents(batch)
            print(f"Batch {i+2} completed in {time.time() - batch_start:.2f} seconds")
            # Show overall progress
            elapsed = time.time() - step_start
            progress = (i+2) / len(batches)
            eta = elapsed / progress - elapsed if progress > 0 else 0
            print(f"Overall progress: {progress*100:.1f}% - ETA: {eta:.1f} seconds")
    else:
        # If only one batch, process normally
        print(f"Processing all {len(text_chunks)} chunks in a single batch")
        db = FAISS.from_documents(text_chunks, embedding_model)
    
    # Save the database
    print(f"Saving vector database to {DB_FAISS_PATH}...")
    save_start = time.time()
    db.save_local(DB_FAISS_PATH)
    print(f"✓ Database saved in {time.time() - save_start:.2f} seconds")
    
    step_time = time.time() - step_start
    print(f"✓ Vector database created and saved in {step_time:.2f} seconds")
    
    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"Process completed successfully in {total_time:.2f} seconds")
    print(f"Documents processed: {len(documents)}")
    print(f"Text chunks created: {len(text_chunks)}")
    print(f"Vector database saved to: {DB_FAISS_PATH}")
    print("=" * 50)

def load_pdf_files(data):
    """Load PDF files from directory"""
    print(f"Scanning directory for PDF files...")
    
    if not os.path.exists(data):
        print(f"Directory {data} does not exist. Creating it...")
        os.makedirs(data)
        print(f"Please place PDF files in the {data} directory and run again.")
        return []
    
    loader = DirectoryLoader(data,
                            glob='*.pdf',
                            loader_cls=PyPDFLoader)
    
    pdf_files = [f for f in os.listdir(data) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files: {', '.join(pdf_files)}")
    
    print(f"Loading PDF contents...")
    documents = loader.load()
    
    for i, doc in enumerate(documents):
        print(f"  - Document {i+1}: {doc.metadata.get('source', 'Unknown')} " + 
              f"({len(doc.page_content)} chars)")
    
    return documents

def create_chunks(extracted_data):
    """Split documents into smaller chunks"""
    if not extracted_data:
        print("No documents to split!")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                  chunk_overlap=50)
    
    print(f"Splitting {len(extracted_data)} documents with chunk size=500, overlap=50...")
    
    chunks = []
    for doc in tqdm(extracted_data, desc="Splitting documents", unit="doc"):
        doc_chunks = text_splitter.split_documents([doc])
        chunks.extend(doc_chunks)
    
    if chunks:
        first_chunk_preview = chunks[0].page_content[:100] + "..." if len(chunks[0].page_content) > 100 else chunks[0].page_content
        print(f"Sample chunk: \"{first_chunk_preview}\"")
    
    return chunks

def get_embedding_model():
    """Initialize and return the embedding model"""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Initializing embedding model: {model_name}")
    
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    
    print(f"Embedding model initialized successfully")
    return embedding_model

if __name__ == "__main__":
    main()