"""
query.py - Question Answering Pipeline

Run this to ask questions about processed papers.
Usage: python query.py "What is the main contribution?"
"""

from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
import argparse
import os
from dotenv import load_dotenv
import time
import json
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

# Initialize models
print("Loading models...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize OpenAI client
llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./vectordb")


def retrieve_hybrid(collection, query, top_k=5):
    """
    Hybrid semantic + keyword search with query expansion
    
    Note: ChromaDB doesn't have built-in BM25, so this is a simplified version.
    In production, use a hybrid search engine like Weaviate or Qdrant.
    """
    # Expand query with variations
    expanded_queries = [
        query,
        query.replace("?", ""),
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


def deduplicate_chunks(chunks, metadatas, similarity_threshold=0.85):
    """Remove semantically similar chunks"""
    if len(chunks) <= 1:
        return chunks, metadatas
    
    # Embed all chunks
    embeddings = embedding_model.encode(chunks)
    
    unique_chunks = [chunks[0]]
    unique_metadatas = [metadatas[0]]
    unique_embeddings = [embeddings[0]]
    
    for i in range(1, len(chunks)):
        # Check similarity with all kept chunks
        from sentence_transformers import util
        similarities = [util.cos_sim(embeddings[i], emb).item() 
                       for emb in unique_embeddings]
        
        if max(similarities) < similarity_threshold:
            unique_chunks.append(chunks[i])
            unique_metadatas.append(metadatas[i])
            unique_embeddings.append(embeddings[i])
    
    print(f"Kept {len(unique_chunks)}/{len(chunks)} chunks after deduplication")
    return unique_chunks, unique_metadatas


def retrieve_chunks(collection, query, top_k=5):
    """Main retrieval function with deduplication"""
    chunks, metadatas = retrieve_hybrid(collection, query, top_k)
    chunks, metadatas = deduplicate_chunks(chunks, metadatas)
    return chunks, metadatas


def generate_answer(query, chunks, metadatas, model="gpt-4o-mini"):
    """Generate answer with streaming"""
    
    # Build context
    context = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks)])
    
    messages = [
        {
            "role": "system", 
            "content": "You are a research paper Q&A assistant. Answer questions based only on the provided context. Cite sources as [N]. If information is missing, say so clearly."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }
    ]
    
    print(f"\n🤖 Answer:\n")
    start_time = time.time()
    
    stream = llm_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=350,
        stream=True,
        stream_options={"include_usage": True}
    )
    
    full_answer = ""
    usage_data = None
    
    for chunk in stream:
        # Collect answer text
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end='', flush=True)
            full_answer += content
        
        # The last chunk contains usage data
        if hasattr(chunk, 'usage') and chunk.usage is not None:
            usage_data = chunk.usage
    
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    print("\n")
    
    # Extract token counts
    if not usage_data:
        print("⚠️  Warning: Usage data not available")
        token_stats = {
            'prompt_tokens': 0,
            'completion_tokens': len(full_answer.split()),
            'total_tokens': 0,
            'model': model,
            'latency_ms': latency_ms,
            'streaming': True
        }
    else:
        token_stats = {
            'prompt_tokens': usage_data.prompt_tokens,
            'completion_tokens': usage_data.completion_tokens,
            'total_tokens': usage_data.total_tokens,
            'model': model,
            'latency_ms': latency_ms,
            'streaming': True
        }
    
    return full_answer, token_stats


def answer_question(query, collection_name="papers", top_k=5):
    """Main query pipeline"""
    print(f"\n{'='*60}")
    print(f"Question: {query}")
    print(f"{'='*60}\n")
    
    # Get collection
    try:
        collection = chroma_client.get_collection(collection_name)
    except Exception as e:
        print(f"Error: Collection '{collection_name}' not found!")
        print("Did you run ingest.py first?")
        return None, None
    
    # Retrieve chunks
    chunks, metadatas = retrieve_chunks(collection, query, top_k=top_k)
    
    if not chunks:
        print("No relevant information found in the database.")
        return None, None
    
    # Generate answer
    answer, token_info = generate_answer(query, chunks, metadatas)
    
    # Prepare retrieval stats
    retrieval_stats = {
        'num_chunks': len(chunks),
        'top_k': top_k,
        'total_context_chars': sum(len(c) for c in chunks)
    }
    
    # Log benchmark
    log_benchmark(query, answer, token_info, retrieval_stats)
    
    return answer, token_info


def log_benchmark(query, answer, token_stats, retrieval_stats, benchmark_file="benchmarks.jsonl"):
    """Log metrics for later analysis"""
    
    # Calculate cost (OpenAI pricing)
    # gpt-4o-mini: $0.15/1M input, $0.60/1M output
    input_cost = (token_stats['prompt_tokens'] / 1_000_000) * 0.15
    output_cost = (token_stats['completion_tokens'] / 1_000_000) * 0.60
    total_cost = input_cost + output_cost
    
    # Tokens per second
    if token_stats.get('latency_ms', 0) > 0:
        tokens_per_sec = (token_stats['completion_tokens'] / token_stats['latency_ms']) * 1000
    else:
        tokens_per_sec = 0
    
    entry = {
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'answer': answer,
        'answer_length_chars': len(answer),
        'answer_length_words': len(answer.split()),
        
        # Token metrics
        'tokens': {
            'prompt': token_stats['prompt_tokens'],
            'completion': token_stats['completion_tokens'],
            'total': token_stats['total_tokens'],
            'model': token_stats['model']
        },
        
        # Performance metrics
        'performance': {
            'latency_ms': token_stats.get('latency_ms', 0),
            'tokens_per_sec': round(tokens_per_sec, 2),
            'streaming': token_stats.get('streaming', False)
        },
        
        # Cost metrics
        'cost': {
            'input_usd': round(input_cost, 6),
            'output_usd': round(output_cost, 6),
            'total_usd': round(total_cost, 6)
        },
        
        # Retrieval metrics
        'retrieval': {
            'num_chunks': retrieval_stats['num_chunks'],
            'top_k': retrieval_stats['top_k'],
            'total_context_chars': retrieval_stats['total_context_chars'],
            'avg_chunk_size': retrieval_stats['total_context_chars'] // retrieval_stats['num_chunks'] if retrieval_stats['num_chunks'] > 0 else 0
        }
    }
    
    with open(benchmark_file, 'a') as f:
        f.write(json.dumps(entry) + '\n')
    
    # Print summary
    print(f"\n📊 Stats:")
    print(f"  Tokens: {token_stats['total_tokens']} ({token_stats['prompt_tokens']} prompt + {token_stats['completion_tokens']} completion)")
    print(f"  Latency: {token_stats.get('latency_ms', 0):.0f}ms ({tokens_per_sec:.1f} tokens/sec)")
    print(f"  Cost: ${total_cost:.6f}")
    print(f"  Retrieved: {retrieval_stats['num_chunks']} chunks ({retrieval_stats['total_context_chars']} chars)")


def interactive_mode(collection_name="papers"):
    """Interactive Q&A session"""
    print("\n" + "="*60)
    print("Research Paper Q&A - Interactive Mode")
    print("="*60)
    print(f"Collection: {collection_name}")
    print("\nType 'quit' or 'exit' to stop")
    print("="*60 + "\n")
    
    while True:
        query = input("\n💬 Your question: ").strip()
        
        if not query:
            continue
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        answer_question(query, collection_name=collection_name, top_k=5)


def analyze_benchmarks(benchmark_file="benchmarks.jsonl"):
    """Analyze benchmark logs"""
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas required for analysis. Install with: pip install pandas")
        return
    
    if not os.path.exists(benchmark_file):
        print(f"No benchmark file found at {benchmark_file}")
        return
    
    # Load all benchmark entries
    entries = []
    with open(benchmark_file, 'r') as f:
        for line in f:
            entries.append(json.loads(line))
    
    if not entries:
        print("No benchmark entries found")
        return
    
    df = pd.DataFrame(entries)
    
    print("\n" + "="*60)
    print("BENCHMARK ANALYSIS")
    print("="*60 + "\n")
    
    # Overall stats
    print(f"Total queries: {len(df)}")
    print(f"Total cost: ${df['cost'].apply(lambda x: x['total_usd']).sum():.4f}")
    print(f"Avg latency: {df['performance'].apply(lambda x: x['latency_ms']).mean():.0f}ms")
    print(f"Avg tokens/query: {df['tokens'].apply(lambda x: x['total']).mean():.0f}")
    
    # Most expensive queries
    print("\n" + "-"*60)
    print("TOP 5 MOST EXPENSIVE QUERIES")
    print("-"*60)
    
    df['total_cost'] = df['cost'].apply(lambda x: x['total_usd'])
    top_expensive = df.nlargest(5, 'total_cost')
    
    for _, row in top_expensive.iterrows():
        print(f"\n{row['query'][:60]}...")
        print(f"  Cost: ${row['total_cost']:.6f}")
        print(f"  Tokens: {row['tokens']['total']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query research papers")
    parser.add_argument("query", 
                       nargs='?',
                       help="Question to ask (or omit for interactive mode)")
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
    parser.add_argument("--analyze",
                       action="store_true",
                       help="Analyze benchmark logs")
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_benchmarks()
    elif args.interactive or not args.query:
        interactive_mode(collection_name=args.collection)
    else:
        answer_question(args.query,
                       collection_name=args.collection,
                       top_k=args.top_k)