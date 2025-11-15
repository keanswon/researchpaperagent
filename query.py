"""
query.py - Question Answering Pipeline

Run this to ask questions about processed papers.
Usage: python query.py "What is the main contribution?"
"""

from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from openai import OpenAI
import argparse
import os
from dotenv import load_dotenv

import json
from datetime import datetime

load_dotenv()

# Initialize models
print("loading models...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Initialize OpenAI client (or use Anthropic)
llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./vectordb")

# basic semantic search
def retrieve_basic(collection, query, top_k=5):
    # Embed query
    query_embedding = embedding_model.encode(query)
    
    # Search vector DB
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    
    chunks = results['documents'][0]
    metadatas = results['metadatas'][0]
    
    print(f"Retrieved {len(chunks)} chunks (basic search)")
    return chunks, metadatas

# retrieval with re-ranking
def retrieve_with_rerank(collection, query, top_k=5, candidate_multiplier=4):
    """
    Process:
    1. Get top_k * candidate_multiplier candidates (e.g., 20 chunks)
    2. Re-rank using cross-encoder (more accurate but slower model)
    3. Return top_k after re-ranking
    """
    # Step 1: Get more candidates than we need
    num_candidates = top_k * candidate_multiplier
    query_embedding = embedding_model.encode(query)
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=num_candidates
    )
    
    chunks = results['documents'][0]
    metadatas = results['metadatas'][0]
    
    # Step 2: Re-rank using cross-encoder
    print(f"Re-ranking {len(chunks)} candidates...")
    pairs = [[query, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)
    
    # Step 3: Sort by score and take top_k
    ranked_indices = scores.argsort()[::-1][:top_k]
    
    reranked_chunks = [chunks[i] for i in ranked_indices]
    reranked_metadatas = [metadatas[i] for i in ranked_indices]
    
    print(f"Retrieved {len(reranked_chunks)} chunks (with re-ranking)")
    return reranked_chunks, reranked_metadatas

# hybrid semantic + keyword search
def retrieve_hybrid(collection, query, top_k=5):
    """
    RETRIEVAL STRATEGY 3: Hybrid semantic + keyword search
    
    Pros: Combines semantic understanding with exact matching
    Cons: More complex, needs keyword index
    
    Note: ChromaDB doesn't have built-in BM25, so this is a simplified version.
    In production, you'd use a hybrid search engine like Weaviate or Qdrant.
    """
    # For now, just do semantic search with query expansion
    # A real implementation would combine BM25 + semantic scores
    
    # Expand query with synonyms/variations
    expanded_queries = [
        query,
        query.replace("?", ""),  # Remove question mark
        query.lower()
    ]
    
    all_chunks = []
    all_metadatas = []
    seen_texts = set()
    
    for q in expanded_queries:
        query_embedding = embedding_model.encode(q)
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Deduplicate
        for chunk, meta in zip(results['documents'][0], results['metadatas'][0]):
            if chunk not in seen_texts:
                all_chunks.append(chunk)
                all_metadatas.append(meta)
                seen_texts.add(chunk)
    
    print(f"Retrieved {len(all_chunks[:top_k])} chunks (hybrid search)")
    return all_chunks[:top_k], all_metadatas[:top_k]


def retrieve_chunks(collection, query, strategy="rerank", top_k=5):
    if strategy == "basic":
        return retrieve_basic(collection, query, top_k)
    elif strategy == "rerank":
        return retrieve_with_rerank(collection, query, top_k)
    elif strategy == "hybrid":
        return retrieve_hybrid(collection, query, top_k)
    else:
        raise ValueError(f"Unknown retrieval strategy: {strategy}")

def generate_answer(query, chunks, metadatas, model="gpt-4o-mini"):
    """
    Call LLM with retrieved context to generate answer
    """
    # Assemble context with citations
    context = ""
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
        page = meta.get('page', 'Unknown')
        source = meta.get('source', 'Unknown')
        context += f"[{i+1}] (Source: {source}, Page {page}):\n{chunk}\n\n"
    
    # System prompt
    system_prompt = """You are a research assistant analyzing academic papers. 
    
Your task:
1. Answer the user's question based ONLY on the provided context
2. Always cite sources using [number] format (e.g., "According to [1], ...")
3. If the context doesn't contain enough information, explicitly say so
4. Be concise but thorough
5. Include page numbers when citing

Remember: Only use information from the provided context chunks."""
    
    user_message = f"""Context from research paper(s):

{context}

Question: {query}

Please provide a detailed answer with citations."""
    
    # LLM call
    print(f"\nGenerating answer using {model}...")
    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.3  # Lower temperature for more factual responses
    )
    
    token_stats = {
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': response.usage.completion_tokens,
        'total_tokens': response.usage.total_tokens,
        'model': model
    }
    
    return response.choices[0].message.content, token_stats


def answer_question(query, collection_name="papers", retrieval_strategy="rerank", top_k=5):
    """
    Main query pipeline
    
    Steps:
    1. Retrieve relevant chunks from vector DB (RETRIEVAL HAPPENS HERE)
    2. Assemble context with metadata
    3. Call LLM to generate answer
    4. Return answer with citations
    """
    print(f"\n{'='*60}")
    print(f"Question: {query}")
    print(f"Retrieval strategy: {retrieval_strategy}")
    print(f"{'='*60}\n")
    
    # Get collection
    try:
        collection = chroma_client.get_collection(collection_name)
    except Exception as e:
        print(f"Error: Collection '{collection_name}' not found!")
        print("Did you run ingest.py first?")
        return None
    
    # Step 1: Retrieve chunks (DIFFERENT STRATEGIES HERE)
    chunks, metadatas = retrieve_chunks(collection, query, strategy=retrieval_strategy, top_k=top_k)
    
    if not chunks:
        return "No relevant information found in the database."
    
    answer, token_info = generate_answer(query, chunks, metadatas)
    retrieval_stats = {
        'num_chunks': len(chunks),
        'strategy': retrieval_strategy,
        'top_k': top_k,
        'total_context_chars': sum(len(c) for c in chunks)
    }
    
    log_benchmark(query, answer, token_info, retrieval_stats)
    
    # Step 2: Generate answer
    return answer, token_info
    

def log_benchmark(query, answer, token_stats, retrieval_stats, benchmark_file="benchmarks.jsonl"):
    """Log query results for benchmarking"""
    entry = {
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'answer_length': len(answer),
        'tokens': token_stats,
        'retrieval': retrieval_stats
    }
    
    with open(benchmark_file, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def interactive_mode(collection_name="papers", retrieval_strategy="rerank"):
    """Interactive Q&A session"""
    print("\n" + "="*60)
    print("Research Paper Q&A - Interactive Mode")
    print("="*60)
    print(f"Collection: {collection_name}")
    print(f"Retrieval strategy: {retrieval_strategy}")
    print("\nType 'quit' or 'exit' to stop")
    print("Type 'strategy <name>' to change retrieval strategy")
    print("Available strategies: basic, rerank, hybrid")
    print("="*60 + "\n")
    
    current_strategy = retrieval_strategy
    
    while True:
        query = input("\n💬 Your question: ").strip()
        
        if not query:
            continue
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if query.lower().startswith('strategy '):
            new_strategy = query.split()[1]
            if new_strategy in ['basic', 'rerank', 'hybrid']:
                current_strategy = new_strategy
                print(f"✓ Switched to {current_strategy} retrieval")
            else:
                print(f"Unknown strategy: {new_strategy}")
            continue
        
        # Answer the question
        answer, token_info = answer_question(query, 
                                collection_name=collection_name,
                                retrieval_strategy=current_strategy)
        
        # token info in the form:
        #     token_stats = {
        #     'prompt_tokens': response.usage.prompt_tokens,
        #     'completion_tokens': response.usage.completion_tokens,
        #     'total_tokens': response.usage.total_tokens,
        #     'model': model
        # }
        
        if answer:
            print(f"\n📝 Answer:\n{answer}")
            
        if token_info:
            print("prompt tokens: ", token_info["prompt_tokens"])
            print("completion tokens: ", token_info["completion_tokens"])
            print("total tokens: ", token_info["total_tokens"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query research papers")
    parser.add_argument("query", 
                       nargs='?',
                       help="Question to ask (or omit for interactive mode)")
    parser.add_argument("--strategy", 
                       choices=["basic", "rerank", "hybrid"],
                       default="rerank",
                       help="Retrieval strategy")
    parser.add_argument("--collection", 
                       default="papers",
                       help="ChromaDB collection name")
    parser.add_argument("--top-k",
                       type=int,
                       default=5,
                       help="Number of chunks to retrieve")
    parser.add_argument("--interactive",
                       action="store_true",
                       help="Start interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive or not args.query:
        # Interactive mode
        interactive_mode(collection_name=args.collection,
                        retrieval_strategy=args.strategy)
    else:
        # Single query mode
        answer = answer_question(args.query,
                                collection_name=args.collection,
                                retrieval_strategy=args.strategy,
                                top_k=args.top_k)
        
        if answer:
            print(f"\n📝 Answer:\n{answer}")