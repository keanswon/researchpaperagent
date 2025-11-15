"""

ingest.py - PDF Processing and Storage Pipeline

Run this once per PDF to process and store in vector database.
Usage: python ingest.py paper.pdf --strategy fixed

"""

import pymupdf
from sentence_transformers import SentenceTransformer
import chromadb
import argparse
import re

print("loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2') # maybe pick a better model later but should b ok
chroma_client = chromadb.PersistentClient(path="./vectordb")


def parse_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    page_chunks = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        page_chunks.append({
            'text': text,
            'page': page_num + 1,
            'source': pdf_path
        })
    
    print(f"Extracted {len(page_chunks)} pages from PDF")
    return page_chunks


# horrid chunk strategy as a baseline (fixed chunk size)
def chunk_fixed_size(page_chunks, chunk_size=500, overlap=100):
    all_chunks = []
    
    for page_data in page_chunks:
        text = page_data['text']
        words = text.split()
        
        # Sliding window over words
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) < 50:  # Skip very small chunks
                continue
                
            chunk = ' '.join(chunk_words)
            all_chunks.append({
                'text': chunk,
                'page': page_data['page'],
                'source': page_data['source']
            })
    
    print(f"Created {len(all_chunks)} fixed-size chunks")
    return all_chunks

# split by headers (still not great)
def chunk_by_section(page_chunks):
    """
    CHUNKING STRATEGY 2: Split by section headers
    
    Pros: Preserves document structure, semantic coherence
    Cons: Sections can be very different sizes
    """
    # Common section patterns in papers
    section_patterns = [
        r'^#+\s+',  # Markdown headers
        r'^\d+\.?\s+[A-Z]',  # "1. Introduction" or "1 Introduction"
        r'^[A-Z][A-Z\s]+$',  # "INTRODUCTION" (all caps)
    ]
    
    all_chunks = []
    current_chunk = ""
    current_page = None
    
    for page_data in page_chunks:
        lines = page_data['text'].split('\n')
        
        for line in lines:
            # Check if line is a section header
            is_header = any(re.match(pattern, line.strip()) for pattern in section_patterns)
            
            if is_header and current_chunk.strip():
                # Save previous section
                all_chunks.append({
                    'text': current_chunk.strip(),
                    'page': current_page,
                    'source': page_data['source']
                })
                current_chunk = line + "\n"
                current_page = page_data['page']
            else:
                current_chunk += line + "\n"
                if current_page is None:
                    current_page = page_data['page']
    
    # Add final chunk
    if current_chunk.strip():
        all_chunks.append({
            'text': current_chunk.strip(),
            'page': current_page,
            'source': page_data['source']
        })
    
    print(f"Created {len(all_chunks)} section-based chunks")
    return all_chunks


# better chunking strategy, may add agentic chunking or more advanced strategy later if needed
def chunk_semantic(page_chunks, similarity_threshold=0.7):
    """
    CHUNKING STRATEGY 3: Group semantically similar sentences
    
    Pros: Natural semantic boundaries
    Cons: Computationally expensive, variable sizes
    """
    from sentence_transformers import util
    
    all_chunks = []
    
    for page_data in page_chunks:
        # Split into sentences
        sentences = re.split(r'[.!?]+', page_data['text'])
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            continue
        
        # Embed all sentences
        embeddings = embedding_model.encode(sentences)
        
        # Group similar consecutive sentences
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # Calculate similarity with previous sentence
            similarity = util.cos_sim(embeddings[i-1], embeddings[i]).item()
            
            if similarity > similarity_threshold:
                current_chunk.append(sentences[i])
            else:
                # Start new chunk
                if len(current_chunk) > 0:
                    all_chunks.append({
                        'text': ' '.join(current_chunk),
                        'page': page_data['page'],
                        'source': page_data['source']
                    })
                current_chunk = [sentences[i]]
        
        # Add final chunk
        if current_chunk:
            all_chunks.append({
                'text': ' '.join(current_chunk),
                'page': page_data['page'],
                'source': page_data['source']
            })
    
    print(f"Created {len(all_chunks)} semantic chunks")
    return all_chunks

# choose chunking strategy
def chunk_text(page_chunks, strategy="fixed"):
    if strategy == "fixed":
        return chunk_fixed_size(page_chunks, chunk_size=500, overlap=100)
    elif strategy == "section":
        return chunk_by_section(page_chunks)
    elif strategy == "semantic":
        return chunk_semantic(page_chunks, similarity_threshold=0.7)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


def embed_and_store(chunks, collection_name="papers"):
    # get or create collection
    try:
        collection = chroma_client.get_collection(collection_name)
        print(f"Using existing collection: {collection_name}")
    except:
        collection = chroma_client.create_collection(collection_name)
        print(f"Created new collection: {collection_name}")
    
    # extract texts for embedding
    texts = [c['text'] for c in chunks]
    
    print("Generating embeddings...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    
    # store in ChromaDB
    print("Storing in vector database...")
    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                'page': str(c['page']),  # ChromaDB needs strings
                'source': c['source']
            } for c in chunks
        ],
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    
    print(f"✓ Stored {len(chunks)} chunks in vector DB")


def process_pdf(pdf_path, chunking_strategy="fixed", collection_name="papers"):
    """
    Main ingestion pipeline
    
    Steps:
    1. Parse PDF → extract text
    2. Chunk text → break into smaller pieces (CHUNKING HAPPENS HERE)
    3. Embed chunks → convert to vectors
    4. Store in DB → save for retrieval
    """
    
    
    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path}")
    print(f"Chunking strategy: {chunking_strategy}")
    print(f"{'='*60}\n")
    
    # Step 1: Parse PDF
    page_chunks = parse_pdf(pdf_path)
    
    # Step 2: Chunk text (DIFFERENT STRATEGIES HERE)
    chunks = chunk_text(page_chunks, strategy=chunking_strategy)
    
    # Step 3: Embed and store
    embed_and_store(chunks, collection_name=collection_name)
    
    print(f"\n✓ Processing complete!")
    print(f"You can now query this paper using query.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDF and store in vector DB")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--strategy", 
                       choices=["fixed", "section", "semantic"],
                       default="fixed",
                       help="Chunking strategy to use")
    parser.add_argument("--collection", 
                       default="papers",
                       help="ChromaDB collection name")
    
    args = parser.parse_args()
    
    process_pdf(args.pdf_path, 
                chunking_strategy=args.strategy,
                collection_name=args.collection)